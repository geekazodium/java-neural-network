package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public interface CostFunction {
    float[] cost(float[] outs,Object trainingDataObj);

    float[] derivative(float[] outs,Object trainingDataObj);
}
