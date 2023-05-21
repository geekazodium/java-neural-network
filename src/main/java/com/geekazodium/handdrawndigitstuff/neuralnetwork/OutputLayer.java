package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class OutputLayer extends AbstractEvaluateLayer{
    public OutputLayer(int nodes) {
        super(nodes);
    }


    public float[] getOutputs() {
        float[] out = new float[this.nodeCount];
        System.arraycopy(this.nodes,0,out,0,this.nodeCount);
        return out;
    }
}
