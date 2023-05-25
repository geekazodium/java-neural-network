package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class InputLayer extends AbstractLayer implements NonFinalLayer{
    private EvaluateLayer nextLayer;

    public InputLayer(int nodes) {
        super(nodes);
    }
    @Override
    public void setNextLayer(EvaluateLayer nextLayer){
        this.nextLayer = nextLayer;
    }

//    public void setInputs(float[] values){
//        if (this.nodeCount >= 0) System.arraycopy(values, 0, this.nodes, 0, Math.min(values.length, this.nodeCount));
//    }

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
}
