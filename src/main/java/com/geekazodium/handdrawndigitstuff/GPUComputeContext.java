package com.geekazodium.handdrawndigitstuff;

import com.geekazodium.handdrawndigitstuff.neuralnetwork.AbstractLayer;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.NeuralNetwork;
import com.geekazodium.handdrawndigitstuff.utils.ConsoleStylizer;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.geekazodium.handdrawndigitstuff.neuralnetwork.AbstractLayer.EVALUATE_LAYER_ID;
import static org.lwjgl.opencl.CL30.*;

public class GPUComputeContext {


    private final long gpuComputeDevice;
    public long[] weightBuffers;
    public long[] biasBuffers;
    private int[] layerTypes;
    private long[] layerEvaluateKernels;
    public long layerTypeBuffer;

    public AbstractLayer[] neuralNetworkLayers;

    private final long commandQueue;
    private final long gpuContext;
    private final long[] deviceLocalMaxWorkSize = new long[3];
    private int neuralNetworkDepth;

    public GPUComputeContext() {
        long[] computeDevices = getPlatformDevices(CL_DEVICE_TYPE_GPU);
        gpuComputeDevice = computeDevices[0];
        gpuContext = getContext(gpuComputeDevice);
        commandQueue = getCommandQueue(gpuComputeDevice, gpuContext);

        long[] workDimLongs = getDeviceInfoLongs(gpuComputeDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES);
        System.arraycopy(workDimLongs,0,deviceLocalMaxWorkSize,0,3);
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork){
        neuralNetworkDepth = neuralNetwork.getDepth();
        this.neuralNetworkLayers = new AbstractLayer[neuralNetworkDepth];
        for (AbstractLayer layer : neuralNetwork.layers) {
            int index = layer.getIndex();
            AbstractLayer[] array = layer.getAsLayerArray();
            System.arraycopy(array, 0, this.neuralNetworkLayers, index, array.length);
        }
    }

    public void createNetworkBuffers() {
        this.layerTypes = new int[this.neuralNetworkDepth];
        this.biasBuffers = new long[this.neuralNetworkDepth];
        this.weightBuffers = new long[this.neuralNetworkDepth];

        for (int i = 0; i < this.neuralNetworkLayers.length; i++) {
            AbstractLayer.LayerBuffers buffer = this.neuralNetworkLayers[i].createBuffer(gpuContext);
            layerTypes[i] = buffer.types();
            weightBuffers[i] = buffer.weights();
            biasBuffers[i] = buffer.biases();
        }

        layerTypeBuffer = clCreateBuffer(gpuContext,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, layerTypes,null);
    }

    public void uploadNetworkToGPU(){
        clEnqueueWriteBuffer(commandQueue,layerTypeBuffer,true,0,layerTypes,null,null);
        for (int i = 0; i < this.neuralNetworkDepth; i++) {
            if(layerTypes[i] == EVALUATE_LAYER_ID){
                AbstractLayer neuralNetworkLayer = neuralNetworkLayers[i];

                long weightBuffer = weightBuffers[i];
                float[] weights = neuralNetworkLayer.getWeights();

                long biasBuffer = biasBuffers[i];
                float[] biases = neuralNetworkLayer.getBiases();

                clEnqueueWriteBuffer(commandQueue, weightBuffer,true,0, weights,null,null);
                clEnqueueWriteBuffer(commandQueue, biasBuffer,true,0, biases,null,null);
            }
        }
        System.out.println("uploading network to GPU...");
        clFinish(commandQueue);
        System.out.println("successfully uploaded network to GPU!");
    }

    public void compileNetworkLayerKernels(){
        this.layerEvaluateKernels = new long[this.neuralNetworkDepth];
        for (int i = 0; i < this.neuralNetworkDepth; i++) {
            String evaluateKernelSrc = this.neuralNetworkLayers[i].getEvaluateKernelSrc();
            if(evaluateKernelSrc == null)continue;
            layerEvaluateKernels[i] = getKernel(gpuComputeDevice, evaluateKernelSrc, "evaluate");
        }
    }

    private int stackSize = 0;

    public long[] layerDataBuffers;
    public float[][] layerStackedData;

    public void setStackSize(int stackSize){
        this.stackSize = stackSize;
        if(stackSize>1024)System.out.println("stack size is above 1024, be careful not to put too much into the gpu memory");
    }

    public float[] stackInput(float[][] inputs){
        int inputLayerSize = this.neuralNetworkLayers[0].nodeCount;
        float[] stackedInputs = new float[inputs.length* inputLayerSize];
        for (int layer = 0; layer < stackSize; layer++) {
            float[] input = inputs[layer];
            System.arraycopy(input,0,stackedInputs,layer*inputLayerSize,inputLayerSize);
        }
        return stackedInputs;
    }

    /**
     * *remember, this doesn't actually complete the set operation, you have to call clFinish to wait until this all is completed
     * @param inputStack
     */
    public void setInputs(float[] inputStack){
        System.arraycopy(inputStack,0,layerStackedData[0],0,inputStack.length);
        clEnqueueWriteBuffer(commandQueue,layerDataBuffers[0],true, 0,layerStackedData[0],null,null);
        clFinish(commandQueue);
    }

    public void createStackedLayerBuffers() {
        layerDataBuffers = new long[this.neuralNetworkDepth];
        layerStackedData = new float[this.neuralNetworkDepth][];
        AbstractLayer[] networkLayers = this.neuralNetworkLayers;
        for (int i = 0; i < networkLayers.length; i++) {
            AbstractLayer neuralNetworkLayer = networkLayers[i];
            neuralNetworkLayer.createLayerBuffer(layerDataBuffers, layerStackedData, gpuContext, stackSize, i);
        }
    }

    public void setKernelArgs(){
        long[] evaluateKernels = this.layerEvaluateKernels;
        for (int i = 0, evaluateKernelsLength = evaluateKernels.length; i < evaluateKernelsLength; i++) {
            long layerEvaluateKernel = evaluateKernels[i];
            neuralNetworkLayers[i].setKernelArgs(layerEvaluateKernel,this, this.layerStackedData, i);
        }
    }
    public void evaluate(){
        long[] evaluateKernels = this.layerEvaluateKernels;
        for (int i = 0, evaluateKernelsLength = evaluateKernels.length; i < evaluateKernelsLength; i++) {
            long layerEvaluateKernel = evaluateKernels[i];//todo evaluate neural network on gpu
            AbstractLayer layer = this.neuralNetworkLayers[i];

            if(layerEvaluateKernel == 0)continue;

            PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(2);
            globalWorkSize.put(layer.nodeCount);
            globalWorkSize.put(stackSize);
            globalWorkSize.rewind();

            clEnqueueNDRangeKernel(
                    this.commandQueue, layerEvaluateKernel, 2,
                    null, globalWorkSize,null,null,null
                    );

        }

        clEnqueueReadBuffer(this.commandQueue,this.layerDataBuffers[this.neuralNetworkDepth - 1],true,0,this.layerStackedData[this.neuralNetworkDepth - 1],null,null);
        clFinish(this.commandQueue);
        System.out.println(Arrays.toString(this.layerStackedData[this.neuralNetworkDepth - 1]));
    }

    private long getKernel(long gpuComputeDevice, String src, String kernelName) {
        long program = compileProgram(gpuComputeDevice, gpuContext, src);
        IntBuffer result = BufferUtils.createIntBuffer(1);
        final long kernel = clCreateKernel(program, kernelName, result);
        checkIfSuccess(result, "create kernel");
        return kernel;
    }

    private static long compileProgram(long device, long context,String src) {
        long program = getProgram(context,src);
        int programBuildResult = clBuildProgram(program, device,"",null,0);
        if(programBuildResult!=CL_SUCCESS){
            ByteBuffer byteBuffer = BufferUtils.createByteBuffer(2048);
            PointerBuffer logSize = BufferUtils.createPointerBuffer(1);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, byteBuffer, logSize);
            int len = (int) logSize.get()-1;
            byte[] log = new byte[len];
            byteBuffer.get(log,0,Math.min(len,2048));
            System.out.println(new String(log));
        }
        return program;
    }

    private static long getProgram(long context, String src) {
        IntBuffer resultBuffer = BufferUtils.createIntBuffer(1);
        long program = clCreateProgramWithSource(context, src, resultBuffer);
        checkIfSuccess(resultBuffer,"create program");
        return program;
    }

    private static long getCommandQueue(long computeDevice, long context) {
        IntBuffer resultBuffer = BufferUtils.createIntBuffer(1);
        long commandQueue = clCreateCommandQueue(context, computeDevice,CL_NONE,resultBuffer);
        checkIfSuccess(resultBuffer,"create command queue");
        return commandQueue;
    }

    private static long getContext(long computeDevice) {
        IntBuffer resultBuffer = BufferUtils.createIntBuffer(1);
        long context = clCreateContext(null, computeDevice, null, 0, resultBuffer);
        checkIfSuccess(resultBuffer,"get context");
        return context;
    }

    private static void checkIfSuccess(IntBuffer resultBuffer,String action){
        int result = resultBuffer.get();
        if(result != CL_SUCCESS) throw new RuntimeException("failed to "+action+", error code:"+result);
    }

    private static long[] getPlatformDevices(int deviceType) {
        PointerBuffer platformsBuffer = BufferUtils.createPointerBuffer(64);
        IntBuffer platformCountBuffer = BufferUtils.createIntBuffer(1);
        int platformResult = clGetPlatformIDs(platformsBuffer,platformCountBuffer);
        if (platformResult != CL_SUCCESS) throw new RuntimeException("Failed to get Platform ID error code:"+platformResult);
        long[] platforms = new long[64];
        int platformCount = platformCountBuffer.get();
        platformsBuffer.get(platforms);

        List<Long> devicesList = new ArrayList<>();

        for(int i = 0; i<platformCount; i++){
            devicesList.addAll(getDevices(platforms[i],deviceType));
        }

        long[] devices = new long[devicesList.size()];
        for (int i = 0; i < devicesList.size(); i++) {
            devices[i] = devicesList.get(i);
        }
        return devices;
    }

    private static List<Long> getDevices(long platform,int deviceType) {
        PointerBuffer devicesBuffer = BufferUtils.createPointerBuffer(64);
        IntBuffer deviceCountBuffer = BufferUtils.createIntBuffer(1);
        int deviceResult = clGetDeviceIDs(platform, deviceType, devicesBuffer, deviceCountBuffer);
        if(deviceResult != CL_SUCCESS){
            if(deviceResult == CL_INVALID_PLATFORM)throw new RuntimeException("Invalid platform");
            System.out.println("device incompatible #");
            return new ArrayList<>();
        }
        long[] devices = new long[64];
        int deviceCount = deviceCountBuffer.get();
        devicesBuffer.get(devices);

        List<Long> devicesList = new ArrayList<>();

        for (int j = 0; j < deviceCount; j++) {
            System.out.println(
                    ConsoleStylizer.lineDivider(60, getDeviceInfoString(devices[j],CL_DEVICE_VENDOR)) +
                            "\n"+
                            getDeviceInfoString(devices[j],CL_DEVICE_NAME)+
                            "\n"+
                            ConsoleStylizer.formatByteSize(getDeviceInfoLong(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE))+
                            " global mem size\n"+
                            ConsoleStylizer.formatByteSize(getDeviceInfoLong(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE))+
                            " global mem cache size\n"+
                            getDeviceWorkItemSizes(devices[j]) +
                            " max work item sizes\n"+
                            ConsoleStylizer.lineDivider(60, "platform:"+platform)
            );
            devicesList.add(devices[j]);
        }
        return devicesList;
    }

    private static String getDeviceWorkItemSizes(long device) {
        long[] workDimLongs = getDeviceInfoLongs(device, CL_DEVICE_MAX_WORK_ITEM_SIZES);
        return workDimLongs[0]+","+workDimLongs[1]+","+workDimLongs[2];
    }

    private static String getDeviceInfoString(long deviceID, int deviceInfo) {
        ByteBuffer attribBuffer = BufferUtils.createByteBuffer(256);
        PointerBuffer lengthBuffer = BufferUtils.createPointerBuffer(1);
        clGetDeviceInfo(deviceID,deviceInfo,attribBuffer,lengthBuffer);
        int length = (int) lengthBuffer.get();

        byte[] bytes = new byte[length-1];
        attribBuffer.get(bytes,0,length-1);

        return new String(bytes);
    }

    private static byte[] getDeviceInfoBytes(long deviceID, int deviceInfo) {
        ByteBuffer attribBuffer = BufferUtils.createByteBuffer(256);
        PointerBuffer lengthBuffer = BufferUtils.createPointerBuffer(1);
        clGetDeviceInfo(deviceID,deviceInfo,attribBuffer,lengthBuffer);
        int length = (int) lengthBuffer.get();

        byte[] bytes = new byte[length];
        attribBuffer.get(bytes,0,length);

        return bytes;
    }
    private static long[] getDeviceInfoLongs(long deviceID, int deviceInfo) {
        LongBuffer attribBuffer = BufferUtils.createLongBuffer(32);
        PointerBuffer lengthBuffer = BufferUtils.createPointerBuffer(1);
        clGetDeviceInfo(deviceID,deviceInfo,attribBuffer,lengthBuffer);
        int length = (int) lengthBuffer.get();

        long[] longs = new long[length];
        attribBuffer.get(longs,0,length);

        return longs;
    }

    private static long getDeviceInfoLong(long deviceID, int deviceInfo){
        LongBuffer attribBuffer = BufferUtils.createLongBuffer(1);
        PointerBuffer lengthBuffer = BufferUtils.createPointerBuffer(1);
        clGetDeviceInfo(deviceID,deviceInfo,attribBuffer,lengthBuffer);
        return attribBuffer.get();
    }

    public void delete() {
        clReleaseContext(this.gpuContext);
        clReleaseCommandQueue(this.commandQueue);
    }

    public long getGPUContext() {
        return this.gpuContext;
    }

    public long getCommandQueue() {
        return commandQueue;
    }
}
