package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.google.gson.JsonObject;

import static com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork.ResidualBlockFrame.RESIDUAL_ADD_ID;

public class ResidualAddBlock  extends ResidualBlockFrame.ResidualMergeOperation{
    public final int startPosition;

    public ResidualAddBlock(int nodes, int input, int startPosition) {
        super(nodes,input);
        this.startPosition = startPosition;
    }

    public ResidualAddBlock(JsonObject object){
        super(object);
        this.startPosition = object.get("startPosition").getAsInt();
    }

    @Override
    public void serialize(JsonObject serialized) {
        super.serialize(serialized);
        serialized.addProperty("startPosition",this.startPosition);
    }

    @Override
    public float[] merge(float[] lastLayer, float[] in) {
        float[] out = new float[this.nodeCount];
        System.arraycopy(in,0,out,0,this.nodeCount);
        for (int i = 0; i < this.inputLength; i++) {
            int index = i + this.startPosition;
            out[index] += lastLayer[index];
        }
        return out;
    }

    @Override
    public float[][] trim(float[] activationChanges) {
        float[] out = new float[inputLength];
        System.arraycopy(activationChanges,this.startPosition,out,0,inputLength);
        return new float[][]{out, activationChanges};
    }

    @Override
    public String name() {
        return "ResidualAdd";
    }

    @Override
    public int getType() {
        return RESIDUAL_ADD_ID;
    }
}
