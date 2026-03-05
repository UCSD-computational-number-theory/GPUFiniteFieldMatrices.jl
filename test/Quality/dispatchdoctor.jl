using Test
using DispatchDoctor: @stable

@stable stable_add(x, y) = x + y
@stable unstable_relu(x) = x > 0 ? x : 0.0

target = isempty(ARGS) ? "." : ARGS[1]
println("Running DispatchDoctor checks (TARGET=$(target))")
@test stable_add(1, 2) == 3

did_throw = false
try
    unstable_relu(0)
catch
    did_throw = true
end
@test did_throw
println("DispatchDoctor checks passed")
