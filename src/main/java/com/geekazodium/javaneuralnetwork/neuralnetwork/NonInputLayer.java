package com.geekazodium.javaneuralnetwork.neuralnetwork;

public interface NonInputLayer {
    void setPreviousLayer(AbstractLayer layer);
    AbstractLayer getPreviousLayer();

    AbstractLayer getEnd();
}
