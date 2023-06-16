package com.geekazodium.handdrawndigitstuff;

import com.geekazodium.handdrawndigitstuff.utils.ConsoleStylizer;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.opencl.CL30.*;

public class GPUComputeContext {

    private final long commandQueue;
    private final long gpuContext;
    private final long[] deviceLocalMaxWorkSize = new long[3];

    public GPUComputeContext() {
        long[] computeDevices = getPlatformDevices(CL_DEVICE_TYPE_GPU);
        long gpuComputeDevice = computeDevices[0];
        gpuContext = getContext(gpuComputeDevice);
        commandQueue = getCommandQueue(gpuComputeDevice, gpuContext);

        long[] workDimLongs = getDeviceInfoLongs(gpuComputeDevice, CL_DEVICE_MAX_WORK_ITEM_SIZES);
        System.arraycopy(workDimLongs,0,deviceLocalMaxWorkSize,0,3);
    }

    private long getKernel(long gpuComputeDevice, String src, String kernelName) {
        long computeMatrixMultiplyProgram = compileProgram(gpuComputeDevice, gpuContext, src);
        IntBuffer result = BufferUtils.createIntBuffer(1);
        final long kernel = clCreateKernel(computeMatrixMultiplyProgram, kernelName, result);
        checkIfSuccess(result, "create kernel");
        return kernel;
    }

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

    public void delete() {
        clReleaseContext(this.gpuContext);
        clReleaseCommandQueue(this.commandQueue);
    }
}
