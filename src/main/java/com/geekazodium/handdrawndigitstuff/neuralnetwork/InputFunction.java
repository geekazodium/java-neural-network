package com.geekazodium.handdrawndigitstuff.neuralnetwork;

@FunctionalInterface
public interface InputFunction {
    float[] createInputs(Object trainingDataObject);
}
