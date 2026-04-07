using GPUEnv
using Test

GPUEnv.activate(backends_to_test = [:JLArrays])

backends = GPUEnv.gpu_backends()
if isempty(backends)
    error("No functional backend found for test scenario.")
else
    backend = first(backends)
    x_cpu = Float32.(1:64)
    x_gpu = backend(x_cpu)
    @test isapprox(x_cpu .* x_cpu, Array(x_gpu .* x_gpu); atol = 1.0e-6)
end

# JSON is a dependency of the workspace root, and should NOT be available:
@test Base.find_package("JSON") === nothing
# YAML is a dependency of the SecondLevel project, and should be available:
@test Base.find_package("YAML") !== nothing
# IniFile is a dependency of the ThirdLevel project, and should be available:
@test Base.find_package("IniFile") === nothing
# CSV is a dependency of the workspace root test project, and should NOT be available in this package:
@test Base.find_package("CSV") === nothing
# JSON2 is a dependency of the SecondLevel2 project, and should NOT be available in this package:
@test Base.find_package("JSON2") === nothing
