package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public abstract class AbstractLayer {
    public final float[] nodes;
    public final int nodeCount;
    protected ActivationFunction activationFunction;

    public AbstractLayer(int nodes){
        this.nodeCount = nodes;
        this.nodes = new float[nodes];
    }


    /**
     * @deprecated modify the way neural network evaluates
     * @param activationFunction
     */
    @Deprecated
    public void evaluate(ActivationFunction activationFunction) {

    }

    public float[] evaluate(float[] in){
        return new float[0];
    }

    public void useActivationFunction(ActivationFunction activationFunction){
        this.activationFunction = activationFunction;
    }
}
