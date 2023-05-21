package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public abstract class AbstractEvaluateLayer extends AbstractLayer{
    protected AbstractLayer previousLayer;

    protected float[] weights;
    protected float[] biases;

    public AbstractEvaluateLayer(int nodes) {
        super(nodes);
    }

    public void setPreviousLayer(AbstractLayer previousLayer){
        this.previousLayer = previousLayer;
    }
    @Override
    public void evaluate() {

    }
}
