using Test, SolveDiffEq

print_head(s::String) = println("\033[1m\033[32mTesting Task\033[0m \033[33m$s\033[0m")
print_body(s::String) = println("             $s")

@testset "Test Bulirsch-Stoer Method" begin
    print_head("y'' - 6y' + 15y = 2sin(3t), y(0) = -1, y'(0) = -4")

    function test_func(t::Real)
        temp = sqrt(6.0) * t
        arg1 = cos(3.0t) + sin(3.0t) / 3.0
        arg2 = 11.0 * exp(3.0t) * cos(temp)
        arg3 = 8.0 * exp(3.0t) * sin(temp) / sqrt(6.0)
        return 0.1 * (arg1 - arg2 - arg3)
    end

    function test_diff!(du::AbstractVector, un::AbstractVector, tn::Real)
        @inbounds du[1] = un[2]
        @inbounds du[2] = -15.0 * un[1] + 6.0 * un[2] + 2.0 * sin(3.0tn)
        return nothing
    end

    Δt = 0.05; ans = test_func(1.0)

    obj = BulirschStoer{7}(Δt, 2)                # stepper
    arr = collect(range(0.0, 1.0; step=Δt))      # t vector
    mat = Matrix{Float64}(undef, 2, length(arr)) # y matrix

    # initial values
    @inbounds mat[1, 1], mat[2, 1] = -1.0, -4.0
    # iterations
    @inbounds for i in 1:length(arr)-1
        apply_step!(view(mat,:,i+1), view(mat,:,i), arr[i], test_diff!, obj)
    end

    print_body("answer = $ans")
    print_body("approx = $(mat[1, end])")
    @test mat[1, end] ≈ ans
end
