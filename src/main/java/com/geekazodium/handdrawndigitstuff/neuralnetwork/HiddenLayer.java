package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class HiddenLayer extends AbstractEvaluateLayer implements NonFinalLayer{
    private AbstractLayer nextLayer;

    public HiddenLayer(int nodes) {
        super(nodes);
    }

    @Override
    protected String name() {
        return "HiddenLayer";
    }

    @Override
    public void setNextLayer(AbstractLayer nextLayer){
        this.nextLayer = nextLayer;
    }
}
