package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public interface NonFinalLayer {
    void setNextLayer(EvaluateLayer layer);
    EvaluateLayer getNextLayer();
}
