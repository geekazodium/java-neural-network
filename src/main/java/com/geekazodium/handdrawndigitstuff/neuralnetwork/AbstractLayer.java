package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public abstract class AbstractLayer {
    public final float[] nodes;
    public final int nodeCount;
    public AbstractLayer(int nodes){
        this.nodeCount = nodes;
        this.nodes = new float[nodes];
    }


    public void evaluate(ActivationFunction activationFunction) {

    }

}
