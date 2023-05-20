package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class OutputLayer extends AbstractLayer{
    private final AbstractLayer previousLayer;
    public OutputLayer(int nodes,AbstractLayer previousLayer) {
        super(nodes);
        this.previousLayer = previousLayer;
    }
}
