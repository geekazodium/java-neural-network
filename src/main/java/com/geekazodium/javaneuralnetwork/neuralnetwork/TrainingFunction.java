package com.geekazodium.javaneuralnetwork.neuralnetwork;

@FunctionalInterface
public interface TrainingFunction {
    void trainOnData(Object trainingDataObject, NeuralNetwork neuralNetwork);
}
