package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public interface EvaluateLayer extends NonInputLayer{
    String name();

    void evaluate(ActivationFunction activationFunction);

    float[][] trainingEvaluate(ActivationFunction activationFunction, float[] previousLayerNodes);
}
