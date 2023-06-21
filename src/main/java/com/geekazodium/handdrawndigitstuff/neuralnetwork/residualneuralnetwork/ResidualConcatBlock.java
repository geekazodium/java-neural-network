package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.geekazodium.handdrawndigitstuff.GPUComputeContext;
import com.google.gson.JsonObject;

import static com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork.ResidualBlockFrame.RESIDUAL_CONCAT_ID;
import static org.lwjgl.opencl.CL30.*;

public class ResidualConcatBlock extends ResidualBlockFrame.ResidualMergeOperation{
    public ResidualConcatBlock(int nodes,int input) {
        super(nodes,input);
    }

    public ResidualConcatBlock(JsonObject object){
        super(object);
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
    public String getEvaluateKernelSrc() {
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
        //System.out.println(kernelSrc);
        return kernelSrc;
    }

    private long prevLayerNodeCountBuffer;
    private long layerNodeCountBuffer;
    private long residualInputSizeBuffer;
    @Override
    public void setEvaluateKernelArgs(long layerEvaluateKernel, GPUComputeContext context, float[][] layerData, int index) {
        if(prevLayerNodeCountBuffer == 0) {
            int[] residualInputSize = {this.residualBlockFrame.nodeCount};
            int[] previousLayerNodeCount = {this.internalPreviousLayer.nodeCount};
            int[] layerNodeCount = {this.nodeCount};

            prevLayerNodeCountBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayerNodeCount, null);
            layerNodeCountBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, layerNodeCount, null);

            residualInputSizeBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, residualInputSize, null);
        }

        clSetKernelArg(layerEvaluateKernel,0,new long[]{context.layerDataBuffers[index-residualBlockFrame.layerDepth()+1]});
        clSetKernelArg(layerEvaluateKernel,1,new long[]{context.layerDataBuffers[index-1]});
        clSetKernelArg(layerEvaluateKernel,2,new long[]{context.layerDataBuffers[index]});
        clSetKernelArg(layerEvaluateKernel,3,new long[]{residualInputSizeBuffer});
        clSetKernelArg(layerEvaluateKernel,4,new long[]{prevLayerNodeCountBuffer});
        clSetKernelArg(layerEvaluateKernel,5,new long[]{layerNodeCountBuffer});
    }
}
