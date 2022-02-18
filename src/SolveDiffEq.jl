module SolveDiffEq

const VecI = AbstractVector # Input  Vector
const MatI = AbstractMatrix # Input  Matrix
const MatB = AbstractMatrix # Buffer Matrix
const ArrI = AbstractArray  # Input  Array
const ArrO = AbstractArray  # Output Array

const ATOL = 1.0e-12
const RTOL = 1.0e-12

import LinearAlgebra: BLAS

if haskey(ENV, "BLAS_THREAD_NUM")
    BLAS.set_num_threads(parse(Int, ENV["BLAS_THREAD_NUM"]))
else
    BLAS.set_num_threads(4)
end

export BulirschStoer, apply_step!

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

macro def(genre::Symbol, ex::Union{Expr,Symbol}, vars::Symbol...)
    o = genre == :prop ? :(::) : genre == :vars ? :(=) : error("@def(genre = $genre, ...) is invalid.")
    n = length(vars)
    e = Vector{Expr}(undef, n)
    @inbounds for i in 1:n
        e[i] = Expr(o, vars[i], ex)
    end
    return Expr(:escape, Expr(:block, e...))
end

macro get(obj::Symbol, vars::Symbol...)
    n = length(vars)
    e = Vector{Expr}(undef, n)
    @inbounds for i in 1:n
        vari = vars[i]
        e[i] = :($vari = $obj.$vari)
    end
    return Expr(:escape, Expr(:block, e...))
end

function unsafe_cpy!(des::AbstractArray, src::AbstractArray)
    @simd for i in eachindex(des)
        @inbounds des[i] = src[i]
    end
    return nothing
end

derivative!(ηn::ArrO, ηm::ArrI, tn::Real, g!::Function) = g!(ηn, ηm, tn)

# @code_warntype ✓
# Don't use `:tuple` here, NTuple will cause
# unexpected allocations during the extrapolation.
function get_stepinv2(ns::NTuple{LMAX,Int}) where LMAX
    if @generated
        a = Vector{Expr}(undef, LMAX-1)
        for k in 1:LMAX-1
            b = Vector{Expr}(undef, LMAX-k)
            @inbounds for i in 1:LMAX-k
                b[i] = :(abs2(ns[$i] / ns[$(i+k)]) - 1.0)
            end
            @inbounds a[k] = Expr(:vect, b...)
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:vect, a...))
        end
    else
        xs = Vector{Vector{Float64}}(undef, LMAX-1)
        for k in 1:LMAX-1
            a = Vector{Float64}(undef, LMAX-k)
            @inbounds for i in 1:LMAX-k
                a[i] = abs2(ns[i] / ns[i+k]) - 1.0
            end
            xs[k] = a
        end
        return xs
    end
end

# @code_warntype ✓
function get_stepsize(ns::NTuple{LMAX,Int}, h::Real) where LMAX
    if @generated
        a = Vector{Expr}(undef, LMAX)
        @inbounds for k in 1:LMAX
            a[i] = :(h / ns[$i])
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:vect, a...))
        end
    else
        hs = Vector{Float64}(undef, LMAX)
        @simd for i in eachindex(hs)
            @inbounds hs[i] = h / ns[i]
        end
        return hs
    end
end

#=
Extrapolate a polynomial τ0 + τ1 * h² + τ2 * h⁴ + ...
by h[i] = h / n[i] where n[1] < n[2] < n[3] < ...
as
    f(h[1]²) = T[11] → T[12] → T[13]
    f(h[2]²) = T[22] → T[23] /
    f(h[3]²) = T[33] /
=#
function extrap2zero!(ys::ArrO, yr::ArrO, ym::MatI, bm::MatB, xs::VecI, ℓ::Int)
    one2n = eachindex(ys)

    @inbounds for i in one2n
        yr[i] = ym[i,ℓ]
    end

    @inbounds for j in 1:ℓ, i in one2n
        bm[i,j] = ym[i,j] - yr[i]
    end

    for k in 1:ℓ-1
        @inbounds xk = xs[k]
        for j in 1:ℓ-k
            @inbounds xkj = xk[j]
            @inbounds for i in one2n
                bm[i,j] += (bm[i,j] - bm[i,j+1]) / xkj
            end
        end
    end

    @inbounds for i in one2n
        ys[i]  = bm[i,1] + yr[i]
        yr[i] += bm[i,2]
    end

    return nothing
end

# @code_warntype ✓
function estimate_err(ys::ArrI, yr::ArrI, y0::ArrI)
    err = 0.0 # error
    @inbounds for i in eachindex(y0)
        scal = ATOL + RTOL * max(abs(y0[i]), abs(yr[i])) # scale
        err += abs2((ys[i] - yr[i]) / scal)
    end
    return sqrt(err / length(y0))
end

struct BulirschStoer{LMAX}
    @def prop Vector{Float64} hs ηt ηm ηn yr # vector of reference
    @def prop Matrix{Float64} ym bm # matrix of sequences, buffer of matrix
    xs::Vector{Vector{Float64}}
    ns::NTuple{LMAX,Int}

    BulirschStoer{13}(H::Real, ydim::Int) = BulirschStoer(H, ydim, (2,  6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50))
    BulirschStoer{ 9}(H::Real, ydim::Int) = BulirschStoer(H, ydim, (2,  8, 14, 20, 26, 32, 38, 44, 50))
    BulirschStoer{ 7}(H::Real, ydim::Int) = BulirschStoer(H, ydim, (2, 10, 18, 26, 34, 42, 50))

    function BulirschStoer(H::Real, ydim::Int, ns::NTuple{LMAX,Int}) where LMAX
        @def vars Vector{Float64}(undef, ydim)       ηt ηm ηn yr
        @def vars Matrix{Float64}(undef, ydim, LMAX) ym bm
        xs = get_stepinv2(ns)
        hs = get_stepsize(ns, H)
        return new{LMAX}(hs, ηt, ηm, ηn, yr, ym, bm, xs, ns)
    end
end

# @code_warntype ✓
function apply_step!(ys::ArrO, y0::ArrI, t0::Real, g!::Function, o::BulirschStoer{LMAX}) where LMAX
    @get o ns xs hs ηt ηm ηn ym yr bm

    prevE = Inf # previous error
    one2n = eachindex(ηt)

    for ℓ in 1:LMAX
        @inbounds hℓ = hs[ℓ]
        @inbounds nℓ = ns[ℓ]
        h2 = 2.0 * hℓ

        unsafe_cpy!(ηm, y0)
        derivative!(ηn, ηm, t0, g!)
        BLAS.axpby!(1., ηm, hℓ, ηn)
    
        tj = t0 + hℓ; j = 1
        while j < nℓ
            derivative!(ηt, ηn, tj, g!)
            BLAS.axpby!(1., ηm, h2, ηt)
            @inbounds for i in one2n
                ηm[i], ηn[i] = ηn[i], ηt[i]
            end
            tj += hℓ; j += 1
        end
    
        derivative!(ηt, ηn, tj, g!)
        @inbounds for i in one2n
            ηt[i] = 0.5 * (ηn[i] + ηm[i] + hℓ * ηt[i])
        end

        if isone(ℓ)
            @inbounds for i in one2n
                ys[i] = ym[i,1] = ηt[i]
                yr[i] = ym[i,2] = 0.0
            end
        else
            @inbounds for i in one2n
                ym[i,ℓ] = ηt[i]
            end
            extrap2zero!(ys, yr, ym, bm, xs, ℓ)
        end

        thisE = estimate_err(ys, yr, y0)
        if thisE > prevE || iszero(thisE)
            break
        end
        prevE = thisE
    end

    return nothing
end

end # module
