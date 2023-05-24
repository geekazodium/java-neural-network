package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public interface EvaluateLayer extends NonInputLayer{
    String name();

    void setActivationFunction(ActivationFunction activationFunction);

    float[][] trainingEvaluate(ActivationFunction activationFunction, float[] previousLayerNodes);

}
