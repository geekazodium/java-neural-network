package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public abstract class AbstractLayer {
    public final int nodeCount;
    protected ActivationFunction activationFunction;
    private int index;

    public AbstractLayer(int nodes){
        this.nodeCount = nodes;
    }

    public float[] evaluate(float[] in, Object[] args){
        return new float[0];
    }

    public float[] evaluateSelf(float[] in, Object[] args){
        return new float[0];
    }

    public float[] evaluateSelf(float[] in){
        return evaluateSelf(in,null);
    }

    public abstract float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject, Object[] args);


    public float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject){
        return backpropagate(in,costFunction,trainingDataObject,null);
    }

    public AbstractLayer getEnd(){
        return this;
    }

    public void printStructure(){
        System.out.println(this.nodeCount +" "+ this.getClass().getName());
        if(this instanceof NonFinalLayer nonFinalLayer) ((AbstractLayer) nonFinalLayer.getNextLayer()).printStructure();
    }

    public int layerDepth(){
        return 1;
    }

    public void assignIndex(int depth) {
        this.setIndex(depth);
    }

    public LayerBuffers createBuffer(long gpuContext) {
        return new LayerBuffers(new long[0],new long[0]);
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public record LayerBuffers(long[] weights, long[] biases){}
}
