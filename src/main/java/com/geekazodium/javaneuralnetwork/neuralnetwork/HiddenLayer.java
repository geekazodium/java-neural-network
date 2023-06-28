package com.geekazodium.javaneuralnetwork.neuralnetwork;

public class HiddenLayer extends AbstractEvaluateLayer implements NonFinalLayer{

    public static final int HIDDEN_LAYER_ID = 178;
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

    @Override
    public int getId() {
        return HIDDEN_LAYER_ID;
    }
}
