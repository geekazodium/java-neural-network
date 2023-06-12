package com.geekazodium.handdrawndigitstuff.neuralnetwork;

@FunctionalInterface
public interface TrainingFunction {
    void trainOnData(Object trainingDataObject, NeuralNetwork neuralNetwork);
}
