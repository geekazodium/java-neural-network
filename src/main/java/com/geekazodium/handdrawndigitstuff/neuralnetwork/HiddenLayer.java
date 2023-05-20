package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class HiddenLayer extends AbstractLayer{
    private final AbstractLayer nextLayer;
    private final AbstractLayer previousLayer;
    public HiddenLayer(int nodes,AbstractLayer previousLayer,AbstractLayer nextLayer) {
        super(nodes);
        this.previousLayer = previousLayer;
        this.nextLayer = nextLayer;
    }
}
