package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public abstract class AbstractLayer {
    protected final float[] nodes;
    protected final int nodeCount;
    public AbstractLayer(int nodes){
        this.nodeCount = nodes;
        this.nodes = new float[nodes];
    }

    public abstract void evaluate();
}
