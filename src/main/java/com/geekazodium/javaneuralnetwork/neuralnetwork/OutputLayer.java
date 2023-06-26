package com.geekazodium.javaneuralnetwork.neuralnetwork;

public class OutputLayer extends AbstractEvaluateLayer{
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
}
