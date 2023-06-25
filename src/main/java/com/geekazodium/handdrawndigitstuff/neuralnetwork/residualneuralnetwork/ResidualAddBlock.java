package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.geekazodium.handdrawndigitstuff.GPUComputeContext;
import com.google.gson.JsonObject;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import static com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork.ResidualBlockFrame.RESIDUAL_ADD_ID;
import static org.lwjgl.opencl.CL30.*;

public class ResidualAddBlock  extends ResidualBlockFrame.ResidualMergeOperation{
    public final int startPosition;

    public ResidualAddBlock(int nodes, int input, int startPosition) {
        super(nodes,input);
        this.startPosition = startPosition;
    }

    public ResidualAddBlock(JsonObject object){
        super(object);
        this.startPosition = object.get("startPosition").getAsInt();
    }

    @Override
    public void serialize(JsonObject serialized) {
        super.serialize(serialized);
        serialized.addProperty("startPosition",this.startPosition);
    }

    @Override
    public float[] merge(float[] lastLayer, float[] in) {
        float[] out = new float[this.nodeCount];
        System.arraycopy(in,0,out,0,this.nodeCount);
        for (int i = 0; i < this.inputLength; i++) {
            int index = i + this.startPosition;
            out[index] += lastLayer[i];
        }
        return out;
    }

    @Override
    public float[][] trim(float[] activationChanges) {
        float[] out = new float[inputLength];
        System.arraycopy(activationChanges,this.startPosition,out,0,inputLength);
        return new float[][]{out, activationChanges};
    }

    @Override
    public String name() {
        return "ResidualAdd";
    }

    @Override
    public int getType() {
        return RESIDUAL_ADD_ID;
    }

    @Override
    public String getEvaluateKernelSrc() {
        String kernelSrc = """
            __kernel void evaluate(
                    __global float *residualBlockInput,
                    __global float *previousLayer,
                    __global float *output,
                    __constant int *residualInputSizePointer,
                    __constant int *additionStartPositionPointer,
                    __constant int *previousLayerSizePointer,
                    __constant int *layerSizePointer
                    ){
                int neuron = get_global_id(0);
                int stackLayer = get_global_id(1);
                
                int layerSize = layerSizePointer[0];
                int previousLayerSize = previousLayerSizePointer[0];
                
                int residualInputSize = residualInputSizePointer[0];
                int additionStartPosition = additionStartPositionPointer[0];
                
                int stackOffset = layerSize * stackLayer;
                int previousLayerStackOffset = previousLayerSize * stackLayer;
                int residualBlockStackOffset = residualInputSize * stackLayer;
                
                int resultLocation = neuron + stackOffset;
                
                float result = residualBlockInput[residualBlockStackOffset + neuron];
                
                int addIndex = neuron - additionStartPosition;
                
                if(addIndex >= 0 && addIndex < previousLayerSize){
                    result += previousLayer[addIndex + previousLayerStackOffset];
                }
                
                output[resultLocation] = result;
            }
            """;
        //System.out.println(kernelSrc);
        return kernelSrc;
    }

    @Override
    public GPUComputeContext.BackPropagateKernels createBackpropagationKernels(GPUComputeContext context, int index) {
        String prevLayerGradientsSrc = """
                __kernel void getResidualGradients(
                        __global float *previousLayerActivationGradients,
                        __global float *residualBlockGradients,
                        __constant float *blockActivationGradients,
                        __constant int *residualInputSizePointer,
                        __constant int *additionStartPositionPointer,
                        __constant int *previousLayerSizePointer,
                        __constant int *layerSizePointer
                        ){
                    int neuron = get_global_id(0);
                    int stackLayer = get_global_id(1);

                    int layerSize = layerSizePointer[0];
                    int previousLayerSize = previousLayerSizePointer[0];

                    int residualInputSize = residualInputSizePointer[0];
                    int additionStartPosition = additionStartPositionPointer[0];

                    int stackOffset = layerSize * stackLayer;
                    int previousLayerStackOffset = previousLayerSize * stackLayer;
                    int residualBlockStackOffset = residualInputSize * stackLayer;

                    int resultLocation = neuron + stackOffset;

                    float gradient = blockActivationGradients[resultLocation];

                    int addIndex = neuron - additionStartPosition;

                    if(addIndex >= 0 && addIndex < previousLayerSize){
                        previousLayerActivationGradients[addIndex + previousLayerStackOffset] = gradient;
                    }

                    residualBlockGradients[residualBlockStackOffset + neuron] = gradient;
                }
                """;

        long residualGradientsKernel = context.getKernel(prevLayerGradientsSrc, "getResidualGradients");
        clSetKernelArg(residualGradientsKernel,0,pointerOf(context.layerGradientBuffers[index-1]));
        clSetKernelArg(residualGradientsKernel,1,pointerOf(this.residualBlockFrame.getMergeGradientBuffer()));
        clSetKernelArg(residualGradientsKernel,2,pointerOf(context.layerGradientBuffers[index]));
        clSetKernelArg(residualGradientsKernel,3,pointerOf(residualInputSizeBuffer));
        clSetKernelArg(residualGradientsKernel,4,pointerOf(blockStartPositionBuffer));
        clSetKernelArg(residualGradientsKernel,5,pointerOf(prevLayerNodeCountBuffer));
        clSetKernelArg(residualGradientsKernel,6,pointerOf(layerNodeCountBuffer));
        return new AddBlockBackpropagateKernel(residualGradientsKernel);
    }

    private long prevLayerNodeCountBuffer;
    private long layerNodeCountBuffer;
    private long blockStartPositionBuffer;
    private long residualInputSizeBuffer;
    @Override
    public void setEvaluateKernelArgs(long layerEvaluateKernel, GPUComputeContext context, float[][] layerData, int index) {
        if(prevLayerNodeCountBuffer == 0) {
            int[] residualInputSize = {this.residualBlockFrame.nodeCount};
            int[] blockStartPosition = {this.startPosition};
            int[] previousLayerNodeCount = {this.internalPreviousLayer.nodeCount};
            int[] layerNodeCount = {this.nodeCount};

            prevLayerNodeCountBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayerNodeCount, null);
            layerNodeCountBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, layerNodeCount, null);

            residualInputSizeBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, residualInputSize, null);
            blockStartPositionBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, blockStartPosition, null);
        }

        clSetKernelArg(layerEvaluateKernel,0,pointerOf(context.layerDataBuffers[index-residualBlockFrame.layerDepth()+1]));
        clSetKernelArg(layerEvaluateKernel,1,pointerOf(context.layerDataBuffers[index-1]));
        clSetKernelArg(layerEvaluateKernel,2,pointerOf(context.layerDataBuffers[index]));
        clSetKernelArg(layerEvaluateKernel,3,pointerOf(residualInputSizeBuffer));
        clSetKernelArg(layerEvaluateKernel,4,pointerOf(blockStartPositionBuffer));
        clSetKernelArg(layerEvaluateKernel,5,pointerOf(prevLayerNodeCountBuffer));
        clSetKernelArg(layerEvaluateKernel,6,pointerOf(layerNodeCountBuffer));
    }

    private class AddBlockBackpropagateKernel extends GPUComputeContext.BackPropagateKernels {
        private final long residualGradientsKernel;

        public AddBlockBackpropagateKernel(long residualGradientsKernel) {
            this.residualGradientsKernel = residualGradientsKernel;
        }

        @Override
        public long[] getKernels() {
            return new long[]{residualGradientsKernel};
        }

        @Override
        public void run(GPUComputeContext context) {
            PointerBuffer workSize = BufferUtils.createPointerBuffer(2);
            workSize.put(nodeCount);
            workSize.put(context.getStackSize());
            workSize.rewind();

            clEnqueueNDRangeKernel(context.getCommandQueue(), residualGradientsKernel,2,null,workSize,null,null,null);
        }
    }
}
