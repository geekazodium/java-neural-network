package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class OutputLayer extends AbstractEvaluateLayer{
    public OutputLayer(int nodes) {
        super(nodes);
    }

    @Override
    public String name() {
        return "OutputLayer";
    }
}
