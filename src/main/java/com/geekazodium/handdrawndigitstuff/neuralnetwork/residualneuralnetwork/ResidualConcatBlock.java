package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

public class ResidualConcatBlock extends ResidualBlockFrame.ResidualMergeOperation{
    public ResidualConcatBlock(int nodes,int input) {
        super(nodes+input,input);
    }

    @Override
    public float[] merge(float[] lastLayer, float[] in) {
        float[] out = new float[this.nodeCount];
        System.arraycopy(lastLayer,0,out,0,inputLength);
        System.arraycopy(in,0,out,inputLength,in.length);
        return out;
    }

    @Override
    public float[] trim(float[] activationChanges) {
        float[] out = new float[inputLength];
        System.arraycopy(activationChanges,0,out,0,inputLength);
        return out;
    }

    @Override
    public String name() {
        return "ResidualConcat";
    }
}
