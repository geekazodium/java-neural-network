package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public abstract class AbstractLayer {
    protected final int[] nodes;
    protected final int nodeCount;
    public AbstractLayer(int nodes){
        this.nodeCount = nodes;
        this.nodes = new int[nodes];
    }
}
