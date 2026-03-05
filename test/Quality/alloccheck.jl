using Test
using AllocCheck

@check_allocs mul_noalloc(x, y) = x * y

target = isempty(ARGS) ? "." : ARGS[1]
println("Running AllocCheck smoke test (TARGET=$(target))")
@test mul_noalloc(3.0, 2.0) == 6.0
println("AllocCheck smoke test passed")
