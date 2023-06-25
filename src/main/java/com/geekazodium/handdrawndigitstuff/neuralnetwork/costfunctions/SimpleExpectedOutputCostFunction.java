package com.geekazodium.handdrawndigitstuff.neuralnetwork.costfunctions;

import com.geekazodium.handdrawndigitstuff.GPUComputeContext;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.CostFunction;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.RunnableKernel;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import static com.geekazodium.handdrawndigitstuff.neuralnetwork.AbstractLayer.pointerOf;
import static org.lwjgl.opencl.CL30.*;

public class SimpleExpectedOutputCostFunction implements CostFunction {

    private final long expectedResultsBuffer;
    private final long costBuffer;
    public SimpleExpectedOutputCostFunction(GPUComputeContext context,int inputSize, int stackSize){
        this.expectedResultsBuffer = clCreateBuffer(context.getGPUContext(),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,new float[inputSize * stackSize],null);
        this.costBuffer = clCreateBuffer(context.getGPUContext(),CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,new float[inputSize * stackSize],null);
    }

    @Override
    public float[] cost(float[] outs, Object trainingDataObj) {
        return new float[0];
    }

    @Override
    public float[] derivative(float[] outs, Object trainingDataObj) {
        return new float[0];
    }


    public void setKernelExpectedResults(GPUComputeContext context, float[][] expectedResultsStack){
        int outputLength = expectedResultsStack[0].length;
        float[] stacked = new float[expectedResultsStack.length * outputLength];
        for (int i = 0; i < expectedResultsStack.length; i++) {
            System.arraycopy(expectedResultsStack[i],0,stacked,outputLength*i,outputLength);
        }
        clEnqueueWriteBuffer(context.getCommandQueue(),expectedResultsBuffer,true,0,stacked,null,null);
        clFinish(context.getCommandQueue());
    }

    @Override
    public RunnableKernel createKernel(GPUComputeContext context) {
        String src = """
                __kernel void cost(
                        __constant float *expectedResults,
                        __constant float *results,
                        __global float *costGradient,
                        __global float *cost,
                        __constant int *layerSizePointer
                        ){
                    int layerSize = layerSizePointer[0];
                    
                    int index = get_global_id(0) + get_global_id(1) * layerSize;
                    
                    float diff = results[index] - expectedResults[index];
                    
                    costGradient[index] = 2.0f * diff;
                    cost[index] = diff * diff;
                }
                """;

        long costKernel = context.getKernel(src, "cost");
        clSetKernelArg(costKernel,0,pointerOf(expectedResultsBuffer));
        clSetKernelArg(costKernel,1,pointerOf(context.layerDataBuffers[context.neuralNetworkLayers.length-1]));
        clSetKernelArg(costKernel,2,pointerOf(context.layerGradientBuffers[context.neuralNetworkLayers.length-1]));
        clSetKernelArg(costKernel,3,pointerOf(this.costBuffer));
        long layerSizeBuffer = context.neuralNetworkLayers[context.neuralNetworkLayers.length - 1].getLayerSizeBuffer();
        if(layerSizeBuffer == 0)throw new RuntimeException("null pointer arg can create undefined behavior.");
        clSetKernelArg(costKernel,4,pointerOf(layerSizeBuffer));
        return new RunnableKernel(){

            @Override
            public void run(GPUComputeContext context) {
                PointerBuffer costFunctionWorkDim = BufferUtils.createPointerBuffer(2);
                costFunctionWorkDim.put(context.neuralNetworkLayers[context.neuralNetworkLayers.length-1].nodeCount);
                costFunctionWorkDim.put(context.getStackSize());
                costFunctionWorkDim.rewind();
                clEnqueueNDRangeKernel(context.getCommandQueue(),costKernel,2,null,costFunctionWorkDim,null,null,null);

                int outputLength = context.neuralNetworkLayers[context.neuralNetworkLayers.length-1].nodeCount * context.getStackSize();
                float[] ptr = new float[outputLength];
                clEnqueueReadBuffer(context.getCommandQueue(),costBuffer,true,0, ptr,null,null);
                clFinish(context.getCommandQueue());
                Thread thread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        int stackSize = context.getStackSize();
                        float cost = 0;
                        for (float v : ptr) {
                            cost += v / (float) stackSize;
                        }
                        System.out.println("Average cost: "+cost);
                    }
                });
                thread.start();
            }
        };
    }

}
