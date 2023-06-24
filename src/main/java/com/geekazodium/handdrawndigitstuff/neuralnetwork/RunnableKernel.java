package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.geekazodium.handdrawndigitstuff.GPUComputeContext;

public interface RunnableKernel {
    void run(GPUComputeContext context);
}
