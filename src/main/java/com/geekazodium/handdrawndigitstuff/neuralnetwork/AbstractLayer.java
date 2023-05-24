package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public abstract class AbstractLayer {
    public final int nodeCount;
    protected ActivationFunction activationFunction;

    public AbstractLayer(int nodes){
        this.nodeCount = nodes;
    }

    public float[] evaluate(float[] in){
        return new float[0];
    }

    public float[] evaluateSelf(float[] in){
        return new float[0];
    }
}
