package com.geekazodium.javaneuralnetwork.neuralnetwork;

import com.geekazodium.javaneuralnetwork.GPUComputeContext;

public interface CostFunction {
    float[] cost(float[] outs,Object trainingDataObj);

    float[] derivative(float[] outs,Object trainingDataObj);

    RunnableKernel createKernel(GPUComputeContext context);
}
