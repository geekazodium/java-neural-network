package com.geekazodium.javaneuralnetwork.neuralnetwork;

import com.geekazodium.javaneuralnetwork.GPUComputeContext;
import com.geekazodium.javaneuralnetwork.neuralnetwork.activationfunctions.ActivationFunction;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import static com.geekazodium.javaneuralnetwork.utils.NetworkFileFormatHelper.getIntBytes;
import static org.lwjgl.opencl.CL30.*;

public abstract class AbstractLayer {
    public static final int ABSTRACT_LAYER_ID = 742;

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

    public static long[] pointerOf(long pointer) {
        if(pointer == 0)throw new RuntimeException("null pointers can lead to undefined behavior, for the sake of everyone's sanity PLEASE DO NOT HECKING DO THIS...");
        return new long[]{pointer};
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

    public RunnableKernel getEvaluateKernel(GPUComputeContext context, int index) {
        return null;
    }

    public void createLayerBuffer(long[] layerDataBuffers, float[][] layerStackedData, GPUComputeContext gpuContext, int stackSize, int index) {
        int layerStackedSize = nodeCount*stackSize;
        float[] layerStacked = new float[layerStackedSize];
        layerStackedData[index] = layerStacked;
        layerDataBuffers[index] = clCreateBuffer(gpuContext.getGPUContext(),CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,layerStacked,null);
        gpuContext.preActivationBuffers[index] = clCreateBuffer(gpuContext.getGPUContext(),CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,new float[stackSize*this.nodeCount],null);
        gpuContext.layerGradientBuffers[index] = clCreateBuffer(gpuContext.getGPUContext(),CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,new float[stackSize*this.nodeCount],null);
    }

    public RunnableKernel createBackpropagationKernels(GPUComputeContext context, int index){

        return null;
    }

    public long getLayerSizeBuffer() {
        return 0;
    }

    public int getId(){
        return ABSTRACT_LAYER_ID;
    }

    public void writeToOutputStream(FileOutputStream outputStream) throws IOException {
        outputStream.write(getIntBytes(getId()));
        outputStream.write(getIntBytes(nodeCount));
    }

    public void readFileInputStream(FileInputStream inputStream) throws IOException {

    }
//    private static float[] readAndLogArray(FileInputStream inputStream) throws IOException {
//        IntBuffer buffer = IntBuffer.allocate(1);
//        float[] floats = readFloatArray(buffer, inputStream);
//        System.out.println(Arrays.toString(floats));
//        return floats;
//    }

    public record LayerBuffers(long weights, long biases, int types){}
}
