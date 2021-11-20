module SolveDiffEq

const VecI  = AbstractVector # Input  Vector
const VecO  = AbstractVector # Output Vector
const VecB  = AbstractVector # Buffer Vector
const VecIO = AbstractVector # In/Out Vector
const MatI  = AbstractMatrix # Input  Matrix
const MatO  = AbstractMatrix # Output Matrix
const MatB  = AbstractMatrix # Buffer Matrix
const MatIO = AbstractMatrix # In/Out Matrix

function unsafe_cpy!(des::AbstractArray, src::AbstractArray)
    @simd for i in eachindex(des)
        @inbounds des[i] = src[i]
    end
    return nothing
end

apply_grad!(ηn::VecO, ηm::VecI, tn::Real, g!::Function) = g!(ηn, ηm, tn)

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

#=
Extrapolate a polynomial τ0 + τ1 * h² + τ2 * h⁴ + ...
by h[i] = h / n[i] where n[1] < n[2] < n[3] < ...
as
    f(h[1]²) = T[11] → T[12] → T[13]
    f(h[2]²) = T[22] → T[23] /
    f(h[3]²) = T[33] /
=#
function extrap2zero!(ys::VecO, yr::VecO, ym::MatI, bm::MatB, xs::VecI, ℓ::Int)
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

struct BulirschStoer{LMAX}
    ns::NTuple{LMAX,Int}
    hs::Vector{Float64}
    xs::Vector{Vector{Float64}}
    ηt::Vector{Float64}
    ηm::Vector{Float64}
    ηn::Vector{Float64}
    yr::Vector{Float64} # vector of reference
    ym::Matrix{Float64} # matrix of sequences
    bm::Matrix{Float64} # buffer of matrix

    BulirschStoer{13}(H::Real, ydim::Int) = BulirschStoer(H, ydim, (2,  6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50))
    BulirschStoer{ 9}(H::Real, ydim::Int) = BulirschStoer(H, ydim, (2,  8, 14, 20, 26, 32, 38, 44, 50))
    BulirschStoer{ 7}(H::Real, ydim::Int) = BulirschStoer(H, ydim, (2, 10, 18, 26, 34, 42, 50))

    function BulirschStoer(H::Real, ydim::Int, ns::NTuple{LMAX,Int}) where LMAX
        hs = get_stepsize(ns, H)
        xs = get_stepinv2(ns)
        ηt = Vector{Float64}(undef, ydim)
        ηm = Vector{Float64}(undef, ydim)
        ηn = Vector{Float64}(undef, ydim)
        yr = Vector{Float64}(undef, ydim)
        ym = Matrix{Float64}(undef, ydim, LMAX)
        bm = Matrix{Float64}(undef, ydim, LMAX)
        return new{LMAX}(ns, hs, xs, ηt, ηm, ηn, yr, ym, bm)
    end
end

end # module
