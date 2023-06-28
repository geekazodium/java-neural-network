package com.geekazodium.javaneuralnetwork.neuralnetwork;

public class OutputLayer extends AbstractEvaluateLayer{
    public static final int OUTPUT_LAYER_ID = 763;
    public OutputLayer(int nodes) {
        super(nodes);
    }

    @Override
    public long getLayerSizeBuffer() {
        return this.layerNodeCountBuffer;
    }

    @Override
    public String name() {
        return "OutputLayer";
    }

    @Override
    public int getId() {
        return OUTPUT_LAYER_ID;
    }
}
