package com.geekazodium.javaneuralnetwork.neuralnetwork;

public interface NonFinalLayer {
    void setNextLayer(EvaluateLayer layer);
    EvaluateLayer getNextLayer();
}
