import Pkg
println("# Activate"); flush(stdout)
Pkg.activate(".")
println("# using Test"); flush(stdout)
using Test
println("# using CuArrays"); flush(stdout)
using CuArrays

println("# Try cufill"); flush(stdout)
N = 128
x_d = cufill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = cufill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

println("# Try y_d .+= x_d"); flush(stdout)
y_d .+= x_d

println("# Test"); flush(stdout)
@test all(Array(y_d) .== 3.0f0)
println("# Completed"); flush(stdout)

