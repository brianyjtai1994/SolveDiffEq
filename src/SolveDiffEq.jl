module SolveDiffEq

const VecI  = AbstractVector # Input  Vector
const VecO  = AbstractVector # Output Vector
const VecB  = AbstractVector # Buffer Vector
const VecIO = AbstractVector # In/Out Vector
const MatI  = AbstractMatrix # Input  Matrix
const MatO  = AbstractMatrix # Output Matrix
const MatB  = AbstractMatrix # Buffer Matrix
const MatIO = AbstractMatrix # In/Out Matrix

# @code_warntype ✓
function get_stepsize(ns::NTuple{KMAX,Int}, h::Real) where KMAX
    if @generated
        a = Vector{Expr}(undef, KMAX)
        @inbounds for k in 1:KMAX
            a[i] = :(h / ns[$i])
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:vect, a...))
        end
    else
        hs = Vector{Float64}(undef, KMAX)
        @simd for i in eachindex(hs)
            @inbounds hs[i] = h / ns[i]
        end
        return hs
    end
end

# @code_warntype ✓
# Don't use `:tuple` here, NTuple will cause
# unexpected allocations during the extrapolation.
function get_stepinv2(ns::NTuple{KMAX,Int}) where KMAX
    if @generated
        a = Vector{Expr}(undef, KMAX-1)
        for k in 1:KMAX-1
            b = Vector{Expr}(undef, KMAX-k)
            @inbounds for i in 1:KMAX-k
                b[i] = :(abs2(ns[$i] / ns[$(i+k)]) - 1.0)
            end
            @inbounds a[k] = Expr(:vect, b...)
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:vect, a...))
        end
    else
        xs = Vector{Vector{Float64}}(undef, KMAX-1)
        for k in 1:KMAX-1
            a = Vector{Float64}(undef, KMAX-k)
            @inbounds for i in 1:KMAX-k
                a[i] = abs2(ns[i] / ns[i+k]) - 1.0
            end
            xs[k] = a
        end
        return xs
    end
end

#=
Extrapolate a polynomial τ0 + τ1 * h² + τ2 * h⁴ + ...
by h[i] = h / n[i] where n[1] < n[2] < n[3] < ... as

    f(h[1]²) = T[11] → T[12] → T[13]
    f(h[2]²) = T[22] → T[23] /
    f(h[3]²) = T[33] /
=#
function extrap2zero(xv::VecI, yv::VecI, bv::VecB, xs::VecI, n::Int)
    one2n = eachindex(1:n)
    @inbounds for i in one2n
        iszero(xv[i]) && return yv[i]
    end
    Δx = Inf
    yp = 0.0
    @inbounds for i in one2n
        δx = abs(xv[i])
        δx < Δx && (Δx = δx; yp = yv[i])
    end
    @simd for i in one2n
        @inbounds bv[i] = yv[i] - yp
    end
    for k in 1:n-1
        @inbounds xk = xs[k]
        @inbounds for i in 1:n-k
            bv[i] += (bv[i] - bv[i+1]) / xk[i]
        end
    end
    @inbounds return bv[1] + yp
end

end # module
