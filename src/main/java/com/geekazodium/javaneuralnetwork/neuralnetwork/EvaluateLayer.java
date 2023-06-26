package com.geekazodium.javaneuralnetwork.neuralnetwork;

import com.geekazodium.javaneuralnetwork.neuralnetwork.activationfunctions.ActivationFunction;

public interface EvaluateLayer extends NonInputLayer{
    String name();

    void setActivationFunction(ActivationFunction activationFunction);

    float[] evaluate(float[] in, Object[] args);

    float[] evaluateSelf(float[] in, Object[] args);

    float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject, Object[] args);

    float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject);

    void init();
}
