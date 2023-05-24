package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public interface EvaluateLayer extends NonInputLayer{
    String name();

    void setActivationFunction(ActivationFunction activationFunction);

    float[][] trainingEvaluate(ActivationFunction activationFunction, float[] previousLayerNodes);

    float[] evaluate(float[] in);

    float[] evaluateSelf(float[] in);

    float[] backpropagate(float[] in,CostFunction costFunction,Object trainingDataObject);
}
