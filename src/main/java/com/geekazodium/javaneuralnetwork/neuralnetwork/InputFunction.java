package com.geekazodium.javaneuralnetwork.neuralnetwork;

@FunctionalInterface
public interface InputFunction {
    float[] createInputs(Object trainingDataObject);
}
