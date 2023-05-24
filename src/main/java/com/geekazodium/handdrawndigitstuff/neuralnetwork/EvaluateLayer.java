package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public interface EvaluateLayer extends NonInputLayer{
    String name();

    void setActivationFunction(ActivationFunction activationFunction);

    float[] evaluate(float[] in);

    float[] evaluateSelf(float[] in);

    float[] backpropagate(float[] in,CostFunction costFunction,Object trainingDataObject);
}
