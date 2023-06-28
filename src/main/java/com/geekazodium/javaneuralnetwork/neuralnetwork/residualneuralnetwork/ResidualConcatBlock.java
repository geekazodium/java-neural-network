package com.geekazodium.javaneuralnetwork.neuralnetwork.residualneuralnetwork;

import com.geekazodium.javaneuralnetwork.GPUComputeContext;
import com.geekazodium.javaneuralnetwork.neuralnetwork.AbstractLayer;
import com.geekazodium.javaneuralnetwork.neuralnetwork.RunnableKernel;
import com.geekazodium.javaneuralnetwork.utils.NetworkFileFormatHelper;
import com.google.gson.JsonObject;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.io.FileInputStream;
import java.io.IOException;

import static com.geekazodium.javaneuralnetwork.neuralnetwork.NeuralNetwork.*;
import static com.geekazodium.javaneuralnetwork.neuralnetwork.residualneuralnetwork.ResidualBlockFrame.RESIDUAL_CONCAT_ID;
import static org.lwjgl.opencl.CL30.*;

public class ResidualConcatBlock extends ResidualBlockFrame.ResidualMergeOperation{
    public ResidualConcatBlock(int nodes,int input) {
        super(nodes,input);
    }

    public ResidualConcatBlock(JsonObject object){
        super(object);
    }

    @Override
    public int getId() {
        return RESIDUAL_CONCAT_ID;
    }

    public static ResidualConcatBlock instantiate(int nodes, int input){
        return new ResidualConcatBlock(nodes+input,input);
    }

    @Override
    public float[] merge(float[] lastLayer, float[] in) {
        float[] out = new float[this.nodeCount];
        System.arraycopy(lastLayer,0,out,0,inputLength);
        System.arraycopy(in,0,out,inputLength,in.length);
        return out;
    }

    @Override
    public float[][] trim(float[] activationChanges) {
        float[] out = new float[inputLength];
        float[] leftover = new float[activationChanges.length-inputLength];
        System.arraycopy(activationChanges,0,out,0,inputLength);
        System.arraycopy(activationChanges,this.inputLength,leftover,0,leftover.length);
        return new float[][]{out, leftover};
    }

    @Override
    public String name() {
        return "ResidualConcat";
    }

    @Override
    public int getType() {
        return RESIDUAL_CONCAT_ID;
    }
    @Override
    public RunnableKernel getEvaluateKernel(GPUComputeContext context, int index) {
        String kernelSrc = """
            __kernel void evaluate(
                    __global float *residualBlockInput,
                    __global float *previousLayer,
                    __global float *output,
                    __constant int *residualInputSizePointer,
                    __constant int *previousLayerSizePointer,
                    __constant int *layerSizePointer
                    ){
                int neuron = get_global_id(0);
                int stackLayer = get_global_id(1);
                
                int layerSize = layerSizePointer[0];
                int previousLayerSize = previousLayerSizePointer[0];
                
                int residualInputSize = residualInputSizePointer[0];
                
                int stackOffset = layerSize * stackLayer;
                int previousLayerStackOffset = previousLayerSize * stackLayer;
                int residualBlockStackOffset = residualInputSize * stackLayer;
                
                int resultLocation = neuron + stackOffset;
                
                float result;
                
                if(neuron < previousLayerSize){
                    result = previousLayer[neuron + previousLayerStackOffset];
                }else{
                    result = residualBlockInput[residualBlockStackOffset + neuron - previousLayerSize];
                }
                
                output[resultLocation] = result;
            }
            """;
        long layerEvaluateKernel = context.getKernel(kernelSrc,"evaluate");

        if(prevLayerNodeCountBuffer == 0) {
            int[] residualInputSize = {this.residualBlockFrame.nodeCount};
            int[] previousLayerNodeCount = {this.internalPreviousLayer.nodeCount};
            int[] layerNodeCount = {this.nodeCount};

            prevLayerNodeCountBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayerNodeCount, null);
            layerNodeCountBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, layerNodeCount, null);

            residualInputSizeBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, residualInputSize, null);
        }

        clSetKernelArg(layerEvaluateKernel,0,pointerOf(context.layerDataBuffers[index-residualBlockFrame.layerDepth()+1]));
        clSetKernelArg(layerEvaluateKernel,1,pointerOf(context.layerDataBuffers[index-1]));
        clSetKernelArg(layerEvaluateKernel,2,pointerOf(context.layerDataBuffers[index]));
        clSetKernelArg(layerEvaluateKernel,3,pointerOf(residualInputSizeBuffer));
        clSetKernelArg(layerEvaluateKernel,4,pointerOf(prevLayerNodeCountBuffer));
        clSetKernelArg(layerEvaluateKernel,5,pointerOf(layerNodeCountBuffer));
        return new RunnableKernel() {
            @Override
            public void run(GPUComputeContext context) {
                PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(2);
                globalWorkSize.put(nodeCount);
                globalWorkSize.put(context.getStackSize());
                globalWorkSize.rewind();

                clEnqueueNDRangeKernel(
                        context.getCommandQueue(), layerEvaluateKernel, 2,
                        null, globalWorkSize,null,null,null
                );
            }
        };
    }

    private long prevLayerNodeCountBuffer;
    private long layerNodeCountBuffer;
    private long residualInputSizeBuffer;

    @Override
    public RunnableKernel createBackpropagationKernels(GPUComputeContext context, int index) {
        String prevLayerGradientsSrc = """
                __kernel void getResidualGradients(
                        __global float *previousLayerActivationGradients,
                        __global float *residualBlockGradients,
                        __constant float *blockActivationGradients,
                        __constant int *residualInputSizePointer,
                        __constant int *previousLayerSizePointer,
                        __constant int *layerSizePointer
                        ){
                    int neuron = get_global_id(0);
                    int stackLayer = get_global_id(1);

                    int layerSize = layerSizePointer[0];
                    int previousLayerSize = previousLayerSizePointer[0];

                    int residualInputSize = residualInputSizePointer[0];

                    int stackOffset = layerSize * stackLayer;
                    int previousLayerStackOffset = previousLayerSize * stackLayer;
                    int residualBlockStackOffset = residualInputSize * stackLayer;

                    int resultLocation = neuron + stackOffset;

                    float gradient = blockActivationGradients[resultLocation];
                    
                    if(neuron >= previousLayerSize){
                        int addIndex = neuron - previousLayerSize;
                        previousLayerActivationGradients[addIndex + previousLayerStackOffset] = gradient;
                    }else{
                        residualBlockGradients[residualBlockStackOffset + neuron] = gradient;
                    }
                }
                """;

        long residualGradientsKernel = context.getKernel(prevLayerGradientsSrc, "getResidualGradients");

        clSetKernelArg(residualGradientsKernel,0,pointerOf(context.layerGradientBuffers[index-1]));
        clSetKernelArg(residualGradientsKernel,1,pointerOf(this.residualBlockFrame.getMergeGradientBuffer()));
        clSetKernelArg(residualGradientsKernel,2,pointerOf(context.layerGradientBuffers[index]));
        clSetKernelArg(residualGradientsKernel,3,pointerOf(residualInputSizeBuffer));
        clSetKernelArg(residualGradientsKernel,4,pointerOf(prevLayerNodeCountBuffer));
        clSetKernelArg(residualGradientsKernel,5,pointerOf(layerNodeCountBuffer));
        return new RunnableKernel() {

            @Override
            public void run(GPUComputeContext context) {

                PointerBuffer workSize = BufferUtils.createPointerBuffer(2);
                workSize.put(nodeCount);
                workSize.put(context.getStackSize());
                workSize.rewind();

                clEnqueueNDRangeKernel(context.getCommandQueue(), residualGradientsKernel,2,null,workSize,null,null,null);

            }
        };
    }

    public static class InitializationHelper implements LayerInitializationHelper {
        @Override
        public AbstractLayer instantiateLayer(FileInputStream inputStream, int nodeCount) throws IOException {
            int inputSize = NetworkFileFormatHelper.readNextInt(inputStream);
            return new ResidualConcatBlock(nodeCount,inputSize);
        }
    }
}
