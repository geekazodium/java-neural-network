package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public interface NonInputLayer {
    void setPreviousLayer(AbstractLayer layer);
    AbstractLayer getPreviousLayer();

    AbstractLayer getEnd();
}
