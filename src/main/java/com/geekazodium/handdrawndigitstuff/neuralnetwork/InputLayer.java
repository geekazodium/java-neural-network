package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class InputLayer extends AbstractLayer{
    private final AbstractLayer nextLayer;

    public InputLayer(int nodes, AbstractLayer nextLayer) {
        super(nodes);
        this.nextLayer = nextLayer;
    }

    public void setInputs(int[] values){
        if (this.nodeCount >= 0) System.arraycopy(values, 0, this.nodes, 0, this.nodeCount);
    }
}
