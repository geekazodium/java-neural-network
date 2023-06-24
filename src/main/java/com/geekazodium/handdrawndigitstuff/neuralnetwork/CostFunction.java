package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.geekazodium.handdrawndigitstuff.GPUComputeContext;

public interface CostFunction {
    float[] cost(float[] outs,Object trainingDataObj);

    float[] derivative(float[] outs,Object trainingDataObj);

    RunnableKernel createKernel(GPUComputeContext context);
}
