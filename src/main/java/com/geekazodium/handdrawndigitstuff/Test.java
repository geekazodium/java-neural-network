package com.geekazodium.handdrawndigitstuff;

import com.geekazodium.handdrawndigitstuff.utils.ConsoleStylizer;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL30;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Test {
    private static final String src = "__kernel void vector_sum(__constant float *a,__constant float *b,__global float *c){\n" +
            "    int i = get_global_id(0);\n" +
            "    float sum = a[i] + b[i];\n" +
            "    c[i] = sum;\n" +
            "}";

    public static void main(String[] args){
        long[] computeDevices = getPlatformDevices(CL30.CL_DEVICE_TYPE_GPU);
        long gpuComputeDevice = computeDevices[0];
        long context = getContext(gpuComputeDevice);
        long commandQueue = getCommandQueue(gpuComputeDevice, context);
        long program = compileProgram(gpuComputeDevice, context,src);

        IntBuffer result = BufferUtils.createIntBuffer(1);
        long kernel = CL30.clCreateKernel(program,"vector_sum",result);
        checkIfSuccess(result,"create kernel");

        float[] vecAData = new float[]{0.8f,0.7f};
        float[] vecBData = new float[]{0.8f,0.7f};
        float[] vecCData = new float[2];

        FloatBuffer vecABuffer = BufferUtils.createFloatBuffer(2);
        FloatBuffer vecBBuffer = BufferUtils.createFloatBuffer(2);
        FloatBuffer vecCBuffer = BufferUtils.createFloatBuffer(2);

        long vecA = CL30.clCreateBuffer(context,CL30.CL_MEM_READ_ONLY | CL30.CL_MEM_COPY_HOST_PTR, vecABuffer,null);
        long vecB = CL30.clCreateBuffer(context,CL30.CL_MEM_READ_ONLY | CL30.CL_MEM_COPY_HOST_PTR, vecBBuffer,null);
        long vecC = CL30.clCreateBuffer(context,CL30.CL_MEM_WRITE_ONLY | CL30.CL_MEM_COPY_HOST_PTR, vecCBuffer,null);

        vecABuffer.put(vecAData);
        vecBBuffer.put(vecBData);
        vecABuffer.rewind();
        vecBBuffer.rewind();

        CL30.clEnqueueWriteBuffer(commandQueue,vecA,true,0,vecABuffer,null,null);
        CL30.clEnqueueWriteBuffer(commandQueue,vecB,true,0,vecBBuffer,null,null);

        CL30.clSetKernelArg(kernel,0,new long[]{vecA});
        CL30.clSetKernelArg(kernel,1,new long[]{vecB});
        CL30.clSetKernelArg(kernel,2,new long[]{vecC});

        PointerBuffer globalWorkOffset = BufferUtils.createPointerBuffer(1);
        PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(1);
        PointerBuffer localWorkSize = BufferUtils.createPointerBuffer(1);

        globalWorkOffset.put(0);
        globalWorkSize.put(2);
        localWorkSize.put(2);

        CL30.clEnqueueNDRangeKernel(
                commandQueue,kernel,1,
                globalWorkOffset.rewind(),globalWorkSize.rewind(),localWorkSize.rewind(),
                null, null
            );


        CL30.clEnqueueReadBuffer(commandQueue,vecC,true,0,vecCBuffer,null,null);

        CL30.clFinish(commandQueue);

        vecCBuffer.get(vecCData);

        System.out.println(Arrays.toString(vecCData));
    }

    private static long compileProgram(long device, long context,String src) {
        ProgramStatus programStatus = getProgram(context,src);
        long program = programStatus.program;
        int programBuildResult = CL30.clBuildProgram(program, device,"",null,0);
        int result = programStatus.resultBuffer.get();
        if(programBuildResult!=CL30.CL_SUCCESS){
            ByteBuffer byteBuffer = BufferUtils.createByteBuffer(256);
            PointerBuffer logSize = BufferUtils.createPointerBuffer(1);
            CL30.clGetProgramBuildInfo(program, device, CL30.CL_PROGRAM_BUILD_LOG, byteBuffer, logSize);
            byte[] log = new byte[256];
            byteBuffer.get(log);
            System.out.println(new String(log));
        }
        return program;
    }

    private static ProgramStatus getProgram(long context,String src) {
        IntBuffer resultBuffer = BufferUtils.createIntBuffer(1);
        long program = CL30.clCreateProgramWithSource(context, src, resultBuffer);
        return new ProgramStatus(program,resultBuffer);
    }

    private record ProgramStatus(long program, IntBuffer resultBuffer){}

    private static long getCommandQueue(long computeDevice, long context) {
        IntBuffer resultBuffer = BufferUtils.createIntBuffer(1);
        long commandQueue = CL30.clCreateCommandQueue(context, computeDevice,CL30.CL_NONE,resultBuffer);
        checkIfSuccess(resultBuffer,"create command queue");
        return commandQueue;
    }

    private static long getContext(long computeDevice) {
        IntBuffer resultBuffer = BufferUtils.createIntBuffer(1);
        long context = CL30.clCreateContext(null, computeDevice, null, 0, resultBuffer);
        checkIfSuccess(resultBuffer,"get context");
        return context;
    }

    private static void checkIfSuccess(IntBuffer resultBuffer,String action){
        int result = resultBuffer.get();
        if(result != CL30.CL_SUCCESS) throw new RuntimeException("failed to "+action+", error code:"+result);
    }

    private static long[] getPlatformDevices(int deviceType) {
        PointerBuffer platformsBuffer = BufferUtils.createPointerBuffer(64);
        IntBuffer platformCountBuffer = BufferUtils.createIntBuffer(1);
        int platformResult = CL30.clGetPlatformIDs(platformsBuffer,platformCountBuffer);
        if (platformResult != CL30.CL_SUCCESS) throw new RuntimeException("Failed to get Platform ID error code:"+platformResult);
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
        int deviceResult = CL30.clGetDeviceIDs(platform, deviceType, devicesBuffer, deviceCountBuffer);
        if(deviceResult != CL30.CL_SUCCESS){
            if(deviceResult == CL30.CL_INVALID_PLATFORM)throw new RuntimeException("Invalid platform");
            System.out.println("device incompatible #");
            return new ArrayList<>();
        }
        long[] devices = new long[64];
        int deviceCount = deviceCountBuffer.get();
        devicesBuffer.get(devices);

        List<Long> devicesList = new ArrayList<>();

        for (int j = 0; j < deviceCount; j++) {
            System.out.println(
                    ConsoleStylizer.lineDivider(60, getDeviceInfoString(devices[j],CL30.CL_DEVICE_VENDOR)) +
                    "\n"+
                    getDeviceInfoString(devices[j],CL30.CL_DEVICE_NAME)+
                    "\n"+
                    ConsoleStylizer.formatByteSize(getDeviceInfoLong(devices[j], CL30.CL_DEVICE_GLOBAL_MEM_SIZE))+
                    " global mem size\n"+
                    ConsoleStylizer.formatByteSize(getDeviceInfoLong(devices[j], CL30.CL_DEVICE_GLOBAL_MEM_CACHE_SIZE))+
                    " global mem cache size\n"+
                    ConsoleStylizer.lineDivider(60, "platform:"+String.valueOf(platform))
            );
            devicesList.add(devices[j]);
        }
        return devicesList;
    }

    private static String getDeviceInfoString(long deviceID, int deviceInfo) {
        ByteBuffer attribBuffer = BufferUtils.createByteBuffer(256);
        PointerBuffer lengthBuffer = BufferUtils.createPointerBuffer(1);
        CL30.clGetDeviceInfo(deviceID,deviceInfo,attribBuffer,lengthBuffer);
        int length = (int) lengthBuffer.get();

        byte[] bytes = new byte[length-1];
        attribBuffer.get(bytes,0,length-1);

        return new String(bytes);
    }

    private static long getDeviceInfoLong(long deviceID, int deviceInfo){
        LongBuffer attribBuffer = BufferUtils.createLongBuffer(1);
        PointerBuffer lengthBuffer = BufferUtils.createPointerBuffer(1);
        CL30.clGetDeviceInfo(deviceID,deviceInfo,attribBuffer,lengthBuffer);
        return attribBuffer.get();
    }
}
