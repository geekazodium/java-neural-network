package com.geekazodium.javaneuralnetwork.neuralnetwork;

import com.geekazodium.javaneuralnetwork.GPUComputeContext;

public interface RunnableKernel {
    void run(GPUComputeContext context);
}
