package com.geekazodium.handdrawndigitstuff;

import com.geekazodium.handdrawndigitstuff.utils.ConsoleStylizer;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.lwjgl.opencl.CL30.*;

public class GPUMatrixComputeContext {
    private static final String listAddSrc = """
            __kernel void array_sum(__constant float *a,__constant float *b,__global float *c){
                int i = get_global_id(0);
                float sum = a[i] + b[i];
                c[i] = sum;
            }""";

    private static final String matrixMultiplySrc = """
            __kernel void matrix_multiply(
                    __constant float *rightMatrix,
                    __constant float *leftMatrix,
                    __global float *resultMatrix,
                    __constant int *dimensions
                    ){
                int x = get_global_id(0);
                int y = get_global_id(1);
                if(x>=dimensions[2] || y>=dimensions[0]){
                    return;
                }
                
                int commonVH = dimensions[1];
                int rightMatrixH = dimensions[2];
                
                for(int i = 0;i<commonVH;i++){
                    int leftMatrixIndex = i + y * commonVH;//i*1 (implicit)
                    int rightMatrixIndex = x + i * rightMatrixH;
                    resultMatrix[x+y*rightMatrixH] += leftMatrix[leftMatrixIndex]*rightMatrix[rightMatrixIndex];
                }
            }
            
            """;

    public static void main(String[] args){
        FloatMatrix matrix1 = new FloatMatrix(1024*8, 1024*8);
        matrix1.fillWithRandom(index -> (float) (Math.random()/5f));
        //System.out.println(matrix1);

        FloatMatrix matrix2 = new FloatMatrix(1024*8, 1024*8);
        matrix2.fillWithRandom(index -> (float) (Math.random()/5f));
        //System.out.println(matrix2);

        GPUMatrixComputeContext computeContext = new GPUMatrixComputeContext();

        System.out.println(Arrays.toString(computeContext.deviceLocalMaxWorkSize));
        MatrixMultiplyCLBuffers clMatrixMultiplyBuffers = computeContext.createCLMatrixMultiplyBuffers(matrix1, matrix2);
        long time1 = System.currentTimeMillis();
        FloatMatrix resultMatrix = computeContext.vectorMatrixMul(matrix1, matrix2, clMatrixMultiplyBuffers);
        long time2 = System.currentTimeMillis();
        clMatrixMultiplyBuffers.release();
        System.out.println(resultMatrix+","+(time2-time1));
    }

    private final long commandQueue;
    private final long computeMatrixMultiplyKernel;
    private final long computeArrayAddKernel;
    private final long gpuContext;
    private final long[] deviceLocalMaxWorkSize = new long[3];

    public GPUMatrixComputeContext() {
        long[] computeDevices = getPlatformDevices(CL_DEVICE_TYPE_GPU);
        long gpuComputeDevice = computeDevices[0];
        gpuContext = getContext(gpuComputeDevice);
        commandQueue = getCommandQueue(gpuComputeDevice, gpuContext);

        computeMatrixMultiplyKernel = getMatrixMultiplyKernel(gpuComputeDevice);
        computeArrayAddKernel = getComputeArrayAddKernel(gpuComputeDevice);

        long[] workDimLongs = getDeviceInfoLongs(gpuComputeDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES);
        System.arraycopy(workDimLongs,0,deviceLocalMaxWorkSize,0,3);
    }

    private long getComputeArrayAddKernel(long gpuComputeDevice) {
        final long computeArrayAddKernel;
        long arrayAddProgram = compileProgram(gpuComputeDevice, gpuContext, listAddSrc);
        IntBuffer result = BufferUtils.createIntBuffer(1);
        computeArrayAddKernel = clCreateKernel(arrayAddProgram,"array_sum",result);
        checkIfSuccess(result,"create kernel");
        return computeArrayAddKernel;
    }

    private long getMatrixMultiplyKernel(long gpuComputeDevice) {
        long computeMatrixMultiplyProgram = compileProgram(gpuComputeDevice, gpuContext, matrixMultiplySrc);
        IntBuffer result = BufferUtils.createIntBuffer(1);
        final long computeMatrixMultiplyKernel = clCreateKernel(computeMatrixMultiplyProgram, "matrix_multiply", result);
        checkIfSuccess(result, "create kernel");
        return computeMatrixMultiplyKernel;
    }

    public FloatMatrix vectorMatrixMul(FloatMatrix leftMatrix, FloatMatrix rightMatrix, MatrixMultiplyCLBuffers buffers){
        if(leftMatrix.width !=rightMatrix.height)throw new RuntimeException("matrix dimension mismatch");

        FloatMatrix resultMatrix = new FloatMatrix(leftMatrix.height,rightMatrix.width);

        clEnqueueWriteBuffer(commandQueue, buffers.rightMatrixCLBuffer(),true,0,rightMatrix.data,null,null);
        clEnqueueWriteBuffer(commandQueue, buffers.leftMatrixCLBuffer(),true,0,leftMatrix.data,null,null);
        clEnqueueWriteBuffer(commandQueue, buffers.dimensionsCLBuffer(),true,0, buffers.dimensions,null,null);

        clSetKernelArg(computeMatrixMultiplyKernel,0,new long[]{buffers.rightMatrixCLBuffer()});
        clSetKernelArg(computeMatrixMultiplyKernel,1,new long[]{buffers.leftMatrixCLBuffer()});
        clSetKernelArg(computeMatrixMultiplyKernel,2,new long[]{buffers.resultMatrixCLBuffer()});
        clSetKernelArg(computeMatrixMultiplyKernel,3,new long[]{buffers.dimensionsCLBuffer()});

        PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(2);
        PointerBuffer localWorkSize = BufferUtils.createPointerBuffer(2);

        int localWorkKernelSize = (int) Math.floor(Math.sqrt(deviceLocalMaxWorkSize[0]));

        int width = resultMatrix.width;
        int height = resultMatrix.height;

//        int localWidth = Math.min(localWorkKernelSize, width);
//        int localHeight = Math.min(localWorkKernelSize, height);

//        if(width%localWidth > 0){
//            width += localWidth-width%localWidth;
//        }
//        if(height%localHeight > 0){
//            height += localHeight-height%localHeight;
//        }
//
        globalWorkSize.put(width);
        globalWorkSize.put(height);
//        localWorkSize.put(localWidth);
//        localWorkSize.put(localHeight);

        globalWorkSize.rewind();
        localWorkSize.rewind();

        clEnqueueNDRangeKernel(
                commandQueue,computeMatrixMultiplyKernel,2,
                null,globalWorkSize,null,
                null, null
        );

        clEnqueueReadBuffer(commandQueue, buffers.resultMatrixCLBuffer(),true,0,resultMatrix.data,null,null);

        clFinish(commandQueue);

        return resultMatrix;
    }

    public MatrixMultiplyCLBuffers createCLMatrixMultiplyBuffers(FloatMatrix leftMatrix, FloatMatrix rightMatrix) {

        int[] dimensions = new int[]{leftMatrix.height,rightMatrix.height, rightMatrix.width};

        long rightMatrixCLBuffer = clCreateBuffer(gpuContext,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, new float[rightMatrix.size()],null);
        long leftMatrixCLBuffer = clCreateBuffer(gpuContext,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, new float[leftMatrix.size()],null);
        long dimensionsCLBuffer = clCreateBuffer(gpuContext,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dimensions,null);
        long resultMatrixCLBuffer = clCreateBuffer(gpuContext,CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, new float[leftMatrix.height*rightMatrix.width],null);
        return new MatrixMultiplyCLBuffers(rightMatrixCLBuffer, leftMatrixCLBuffer, dimensionsCLBuffer, resultMatrixCLBuffer, dimensions);
    }

    public record MatrixMultiplyCLBuffers(long rightMatrixCLBuffer, long leftMatrixCLBuffer, long dimensionsCLBuffer, long resultMatrixCLBuffer,
                                          int[] dimensions) {
        private void release(){
            clReleaseMemObject(rightMatrixCLBuffer);
            clReleaseMemObject(leftMatrixCLBuffer);
            clReleaseMemObject(dimensionsCLBuffer);
            clReleaseMemObject(resultMatrixCLBuffer);
        }
    }
//
//    private static void benchmarkCL(int vectorSize) {
//        long[] computeDevices = getPlatformDevices(CL_DEVICE_TYPE_GPU);
//        long gpuComputeDevice = computeDevices[0];
//        long context = getContext(gpuComputeDevice);
//        long commandQueue = getCommandQueue(gpuComputeDevice, context);
//        long program = compileProgram(gpuComputeDevice, context, listAddSrc);
////
////        IntBuffer result = BufferUtils.createIntBuffer(1);
////        long kernel = clCreateKernel(program,"vector_sum",result);
////        checkIfSuccess(result,"create kernel");
//
//        long timeStart = System.currentTimeMillis();
//
//        float[] vecAData = new float[vectorSize];
//        float[] vecBData = new float[vectorSize];
//        float[] vecCData = new float[vectorSize];
//        for (int i = 0; i < vectorSize; i++) {
//            vecAData[i] = (float) Math.random();
//            vecBData[i] = (float) Math.random();
//        }
//
//        long vecA = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vecAData,null);
//        long vecB = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vecBData,null);
//        long vecC = clCreateBuffer(context,CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, vecCData,null);
//
//        clEnqueueWriteBuffer(commandQueue,vecA,true,0,vecAData,null,null);
//        clEnqueueWriteBuffer(commandQueue,vecB,true,0,vecBData,null,null);
//
//        clSetKernelArg(kernel,0,new long[]{vecA});
//        clSetKernelArg(kernel,1,new long[]{vecB});
//        clSetKernelArg(kernel,2,new long[]{vecC});
//
//        PointerBuffer globalWorkOffset = BufferUtils.createPointerBuffer(1);
//        PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(1);
//        PointerBuffer localWorkSize = BufferUtils.createPointerBuffer(1);
//
//        globalWorkOffset.put(0);
//        globalWorkSize.put(vectorSize);
//        localWorkSize.put(256);
//
//        globalWorkSize.rewind();
//        globalWorkOffset.rewind();
//        localWorkSize.rewind();
//
//        clEnqueueNDRangeKernel(
//                commandQueue,kernel,1,
//                globalWorkOffset.rewind(),globalWorkSize.rewind(),localWorkSize.rewind(),
//                null, null
//            );
//
//
//        clEnqueueReadBuffer(commandQueue,vecC,true,0,vecCData,null,null);
//
//        clFinish(commandQueue);
//
//        long timeComplete = System.currentTimeMillis();
//
//        System.out.println(Arrays.toString(vecCData));
//
//        System.out.println(timeComplete-timeStart);
//    }

    private static long compileProgram(long device, long context,String src) {
        long program = getProgram(context,src);
        int programBuildResult = clBuildProgram(program, device,"",null,0);
        if(programBuildResult!=CL_SUCCESS){
            ByteBuffer byteBuffer = BufferUtils.createByteBuffer(256);
            PointerBuffer logSize = BufferUtils.createPointerBuffer(1);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, byteBuffer, logSize);
            int len = (int) logSize.get();
            byte[] log = new byte[len];
            byteBuffer.get(log,0,len);
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
}
