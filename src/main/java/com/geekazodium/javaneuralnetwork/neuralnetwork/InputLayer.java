package com.geekazodium.javaneuralnetwork.neuralnetwork;

public class InputLayer extends AbstractLayer implements NonFinalLayer{
    private static final int INPUT_LAYER_ID = 607;
    private EvaluateLayer nextLayer;

    public InputLayer(int nodes) {
        super(nodes);
    }
    @Override
    public void setNextLayer(EvaluateLayer nextLayer){
        this.nextLayer = nextLayer;
    }

    @Override
    public float[] evaluate(float[] in, Object[] args) {
        float[] thisLayer = new float[this.nodeCount];
        System.arraycopy(in,0,thisLayer,0,Math.min(this.nodeCount,in.length));
        return this.nextLayer.evaluate(thisLayer, null);
    }

    @Override
    public float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject, Object[] args) {
        return nextLayer.backpropagate(in,costFunction,trainingDataObject);
    }

    @Override
    public EvaluateLayer getNextLayer(){
        return this.nextLayer;
    }

    @Override
    public int getId() {
        return INPUT_LAYER_ID;
    }
}
