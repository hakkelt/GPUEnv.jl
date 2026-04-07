using GPUEnv

GPUEnv.activate(backends_to_test = [:JLArrays])

backends = GPUEnv.gpu_backends()
if isempty(backends)
    error("No functional backend found for benchmark scenario.")
else
    backend = first(backends)
    x_cpu = Float32.(1:64)
    x_gpu = backend(x_cpu)
    isapprox(x_cpu .* x_cpu, Array(x_gpu .* x_gpu); atol = 1.0e-6) || error("GPU smoke test failed")
end

# Test if a dependency from parent project is available (JSON is a dependency of the root project, but not of this package)
try
    using JSON
catch e
    error("JSON dependency was not available through the parent project")
end
