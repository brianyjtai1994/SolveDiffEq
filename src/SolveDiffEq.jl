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
function get_stepsize(ns::VecI{Int}, H::Real)
    hs = similar(ns, Float64)
    @simd for i in eachindex(hs)
        @inbounds hs[i] = H / ns[i]
    end
    return hs
end

# @code_warntype ✓
function get_stepinv2(ns::VecI{Int})
    xs = similar(ns, Float64)
    @simd for i in eachindex(xs)
        @inbounds xs[i] = 4 / abs2(ns[i])
    end
    return xs
end

end # module
