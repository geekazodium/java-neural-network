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

public class Test {
    private static final String src = """
            __kernel void vector_sum(__constant float *a,__constant float *b,__global float *c){
                int i = get_global_id(0);
                float sum = a[i] + b[i];
                c[i] = sum;
            }""";

    private static final String matrixMultiplySrc = """
            __kernel void matrix_multiply(
                    __constant float *leftMatrix,
                    __constant float *rightMatrix,
                    __global float *resultMatrix,
                    __constant int *dimensions
                    ){
                int v = get_global_id(0);
                int h = get_global_id(1);
                
                int commonVH = dimensions[1];
                int rightMatrixH = dimensions[2];
                                
                for(int i = 0;i<commonVH;i++){
                    int leftMatrixIndex = i + v * commonVH;//i*1 (implicit)
                    int rightMatrixIndex = h + i * rightMatrixH;
                    resultMatrix[v+commonVH*h] += leftMatrix[leftMatrixIndex]*rightMatrix[rightMatrixIndex];
                }
            }
            
            """;

//    __constant int *leftMatrixV,
//    __constant int *rightMatrixH,
//    __constant int *commonVH

    public static void main(String[] args){

        int vectorSize = 1024*16;
//        benchmarkCL(vectorSize);

        long timeStart = System.currentTimeMillis();
        float[] vecAData = new float[vectorSize];
        float[] vecBData = new float[vectorSize];
        float[] vecCData = new float[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            vecAData[i] = (float) Math.random();
            vecBData[i] = (float) Math.random();
        }
        for (int i = 0;i<vectorSize;i++){
            vecCData[i] = vecAData[i] + vecBData[i];
        }
        long timeEnd = System.currentTimeMillis();
        System.out.println(timeEnd-timeStart);

        FloatMatrix matrix1 = new FloatMatrix(2,3);
        matrix1.set(10,1,2);
        matrix1.set(-10,1,1);
        matrix1.set(5,0,1);
        matrix1.set(-5,0,0);
        System.out.println(matrix1);

        FloatMatrix matrix2 = new FloatMatrix(3,1);
        matrix2.set(3,0,0);
        matrix2.set(6,1,0);
        matrix2.set(-2,2,0);
        System.out.println(matrix2);
        vectorMatrixMul(matrix1,matrix2);
    }

    public static class FloatMatrix{
        public int height;
        public int width;
        public float[] data;
        public FloatMatrix(int v, int h){
            this(new float[v*h],v,h);
        }
        public FloatMatrix(float[] data, int v, int h) {
            this.height = v;
            this.width = h;
            this.data = data;
        }

        @Override
        public String toString() {
            StringBuilder out = new StringBuilder("[");
            for (int i = 0; i < this.data.length; i++) {
                int x = (i+1)% width;
                out.append(this.data[i]);
                if(i+1 >= this.data.length) out.append("]");
                else{
                    out.append(", ");
                    if(x == 0){
                        out.append("\n ");
                    }
                }
            }
            return out.toString();
        }

        public void set(float value,int v,int h){
            this.data[h+v*this.width] = value;
        }
    }

    private static void vectorMatrixMul(FloatMatrix leftMatrix, FloatMatrix rightMatrix){
        if(leftMatrix.width !=rightMatrix.height)throw new RuntimeException("matrix dimension mismatch");

        long[] computeDevices = getPlatformDevices(CL_DEVICE_TYPE_GPU);
        long gpuComputeDevice = computeDevices[0];
        long context = getContext(gpuComputeDevice);
        long commandQueue = getCommandQueue(gpuComputeDevice, context);
        long program = compileProgram(gpuComputeDevice, context,matrixMultiplySrc);

        IntBuffer result = BufferUtils.createIntBuffer(1);
        long kernel = clCreateKernel(program,"matrix_multiply",result);
        checkIfSuccess(result,"create kernel");

        long timeStart = System.currentTimeMillis();

        FloatMatrix resultMatrix = new FloatMatrix(leftMatrix.height,rightMatrix.width);

        IntBuffer dimensions = BufferUtils.createIntBuffer(3);
        dimensions.put(leftMatrix.height);
        dimensions.put(rightMatrix.height);
        dimensions.put(rightMatrix.width);
        dimensions.rewind();

        long rightMatrixCLBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rightMatrix.data,null);
        long leftMatrixCLBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, leftMatrix.data,null);
        long dimensionsCLBuffer = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dimensions,null);
        long resultMatrixCLBuffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, resultMatrix.data,null);

        clEnqueueWriteBuffer(commandQueue,rightMatrixCLBuffer,true,0,rightMatrix.data,null,null);
        clEnqueueWriteBuffer(commandQueue,leftMatrixCLBuffer,true,0,leftMatrix.data,null,null);
        clEnqueueWriteBuffer(commandQueue,dimensionsCLBuffer,true,0,dimensions,null,null);

        clSetKernelArg(kernel,0,new long[]{rightMatrixCLBuffer});
        clSetKernelArg(kernel,1,new long[]{leftMatrixCLBuffer});
        clSetKernelArg(kernel,2,new long[]{resultMatrixCLBuffer});
        clSetKernelArg(kernel,3,new long[]{dimensionsCLBuffer});

        PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(2);
        PointerBuffer localWorkSize = BufferUtils.createPointerBuffer(2);

        globalWorkSize.put(resultMatrix.height);
        globalWorkSize.put(resultMatrix.width);
        localWorkSize.put(Math.min(256,resultMatrix.height));
        localWorkSize.put(Math.min(256,resultMatrix.width));

        globalWorkSize.rewind();
        localWorkSize.rewind();

        clEnqueueNDRangeKernel(
                commandQueue,kernel,2,
                null,globalWorkSize,localWorkSize,
                null, null
        );


        clEnqueueReadBuffer(commandQueue,resultMatrixCLBuffer,true,0,resultMatrix.data,null,null);

        clFinish(commandQueue);

        long timeComplete = System.currentTimeMillis();

        System.out.println(resultMatrix);

        System.out.println(timeComplete-timeStart);
    }

    private static void benchmarkCL(int vectorSize) {
        long[] computeDevices = getPlatformDevices(CL_DEVICE_TYPE_GPU);
        long gpuComputeDevice = computeDevices[0];
        long context = getContext(gpuComputeDevice);
        long commandQueue = getCommandQueue(gpuComputeDevice, context);
        long program = compileProgram(gpuComputeDevice, context,src);

        IntBuffer result = BufferUtils.createIntBuffer(1);
        long kernel = clCreateKernel(program,"vector_sum",result);
        checkIfSuccess(result,"create kernel");

        long timeStart = System.currentTimeMillis();

        float[] vecAData = new float[vectorSize];
        float[] vecBData = new float[vectorSize];
        float[] vecCData = new float[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            vecAData[i] = (float) Math.random();
            vecBData[i] = (float) Math.random();
        }

        long vecA = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vecAData,null);
        long vecB = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vecBData,null);
        long vecC = clCreateBuffer(context,CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, vecCData,null);

        clEnqueueWriteBuffer(commandQueue,vecA,true,0,vecAData,null,null);
        clEnqueueWriteBuffer(commandQueue,vecB,true,0,vecBData,null,null);

        clSetKernelArg(kernel,0,new long[]{vecA});
        clSetKernelArg(kernel,1,new long[]{vecB});
        clSetKernelArg(kernel,2,new long[]{vecC});

        PointerBuffer globalWorkOffset = BufferUtils.createPointerBuffer(1);
        PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(1);
        PointerBuffer localWorkSize = BufferUtils.createPointerBuffer(1);

        globalWorkOffset.put(0);
        globalWorkSize.put(vectorSize);
        localWorkSize.put(256);

        globalWorkSize.rewind();
        globalWorkOffset.rewind();
        localWorkSize.rewind();

        clEnqueueNDRangeKernel(
                commandQueue,kernel,1,
                globalWorkOffset.rewind(),globalWorkSize.rewind(),localWorkSize.rewind(),
                null, null
            );


        clEnqueueReadBuffer(commandQueue,vecC,true,0,vecCData,null,null);

        clFinish(commandQueue);

        long timeComplete = System.currentTimeMillis();

        System.out.println(Arrays.toString(vecCData));

        System.out.println(timeComplete-timeStart);
    }

    private static long compileProgram(long device, long context,String src) {
        ProgramStatus programStatus = getProgram(context,src);
        long program = programStatus.program;
        int programBuildResult = clBuildProgram(program, device,"",null,0);
        int result = programStatus.resultBuffer.get();
        if(programBuildResult!=CL_SUCCESS){
            ByteBuffer byteBuffer = BufferUtils.createByteBuffer(256);
            PointerBuffer logSize = BufferUtils.createPointerBuffer(1);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, byteBuffer, logSize);
            byte[] log = new byte[256];
            byteBuffer.get(log);
            System.out.println(new String(log));
        }
        return program;
    }

    private static ProgramStatus getProgram(long context,String src) {
        IntBuffer resultBuffer = BufferUtils.createIntBuffer(1);
        long program = clCreateProgramWithSource(context, src, resultBuffer);
        return new ProgramStatus(program,resultBuffer);
    }

    private record ProgramStatus(long program, IntBuffer resultBuffer){}

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
                            ConsoleStylizer.formatByteSize(getDeviceInfoLong(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE))+
                            " global mem cache size\n"+
                    ConsoleStylizer.lineDivider(60, "platform:"+String.valueOf(platform))
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
