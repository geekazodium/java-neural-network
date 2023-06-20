package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.geekazodium.handdrawndigitstuff.GPUComputeContext;

import static org.lwjgl.opencl.CL30.*;

public abstract class AbstractLayer {
    public static final int ABSTRACT_LAYER_ID = 742;
    public static final int EVALUATE_LAYER_ID = 880;
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
        return new LayerBuffers(0,0,ABSTRACT_LAYER_ID);
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public AbstractLayer[] getAsLayerArray() {
        return new AbstractLayer[]{this};
    }

    public float[] getWeights() {
        return null;
    }

    public float[] getBiases() {
        return null;
    }

    public String getEvaluateKernelSrc() {
        return null;
    }

    public void setKernelArgs(long layerEvaluateKernel, GPUComputeContext context, float[][] layerData, int index) {
        //throw new RuntimeException("can not set kernel args for layer without evaluate kernel");
    }

    public void createLayerBuffer(long[] layerDataBuffers, float[][] layerStackedData, long gpuContext, int stackSize, int index) {
        int layerStackedSize = nodeCount*stackSize;
        float[] layerStacked = new float[layerStackedSize];
        layerStackedData[index] = layerStacked;
        layerDataBuffers[index] = clCreateBuffer(gpuContext,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,layerStacked,null);
    }

    public record LayerBuffers(long weights, long biases, int types){}
}
