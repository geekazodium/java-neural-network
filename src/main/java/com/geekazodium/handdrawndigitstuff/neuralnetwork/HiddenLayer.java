package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class HiddenLayer extends AbstractEvaluateLayer implements NonFinalLayer{
    private EvaluateLayer nextLayer;

    public HiddenLayer(int nodes) {
        super(nodes);
    }

    @Override
    public String name() {
        return "HiddenLayer";
    }

    @Override
    public void setNextLayer(EvaluateLayer nextLayer){
        this.nextLayer = nextLayer;
    }

    @Override
    public EvaluateLayer getNextLayer(){
        return this.nextLayer;
    }
}
