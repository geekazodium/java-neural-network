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
        if (this.nodeCount >= 0) System.arraycopy(values, 0, this.nodes, 0, this.nodeCount);
    }

    @Override
    public void evaluate() {
        throw new RuntimeException("can not call evaluate on input layer");
    }
}
