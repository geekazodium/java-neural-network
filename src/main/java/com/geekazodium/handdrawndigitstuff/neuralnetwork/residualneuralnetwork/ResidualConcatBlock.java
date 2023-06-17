package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.google.gson.JsonObject;

import static com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork.ResidualBlockFrame.RESIDUAL_CONCAT_ID;

public class ResidualConcatBlock extends ResidualBlockFrame.ResidualMergeOperation{
    public ResidualConcatBlock(int nodes,int input) {
        super(nodes,input);
    }

    public ResidualConcatBlock(JsonObject object){
        super(object);
    }

    public static ResidualConcatBlock instantiate(int nodes, int input){
        return new ResidualConcatBlock(nodes+input,input);
    }

    @Override
    public float[] merge(float[] lastLayer, float[] in) {
        float[] out = new float[this.nodeCount];
        System.arraycopy(lastLayer,0,out,0,inputLength);
        System.arraycopy(in,0,out,inputLength,in.length);
        return out;
    }

    @Override
    public float[][] trim(float[] activationChanges) {
        float[] out = new float[inputLength];
        float[] leftover = new float[activationChanges.length-inputLength];
        System.arraycopy(activationChanges,0,out,0,inputLength);
        System.arraycopy(activationChanges,this.inputLength,leftover,0,leftover.length);
        return new float[][]{out, leftover};
    }

    @Override
    public String name() {
        return "ResidualConcat";
    }

    @Override
    public int getType() {
        return RESIDUAL_CONCAT_ID;
    }
}
