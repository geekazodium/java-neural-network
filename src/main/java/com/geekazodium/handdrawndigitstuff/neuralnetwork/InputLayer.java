package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class InputLayer extends AbstractLayer implements NonFinalLayer{
    private AbstractLayer nextLayer;

    public InputLayer(int nodes) {
        super(nodes);
    }
    @Override
    public void setNextLayer(AbstractLayer nextLayer){
        this.nextLayer = nextLayer;
    }

    public void setInputs(float[] values){
        if (this.nodeCount >= 0) System.arraycopy(values, 0, this.nodes, 0, Math.min(values.length, this.nodeCount));
    }
}
