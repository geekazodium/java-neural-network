package com.geekazodium.handdrawndigitstuff;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;

public class Test {
    public static void main(String[] args){
//        FloatBuffer a = BufferUtils.createFloatBuffer(10);
//        FloatBuffer b = BufferUtils.createFloatBuffer(10);
//        FloatBuffer out = BufferUtils.createFloatBuffer(a.capacity());
//        a.put(new float[]{1f,2f,3f,4f,5f,6f,7f,8f,9f,10f});
//        b.put(new float[]{1f,2f,3f,4f,5f,6f,7f,8f,9f,10f});


        PointerBuffer platformsBuffer = BufferUtils.createPointerBuffer(64);
        IntBuffer platformCountBuffer = BufferUtils.createIntBuffer(1);
        int platformResult = CL30.clGetPlatformIDs(platformsBuffer,platformCountBuffer);
        if (platformResult != CL30.CL_SUCCESS) throw new RuntimeException("Failed to get Platform ID error code:"+platformResult);
        long[] platforms = new long[64];
        int platformCount = platformCountBuffer.get();
        platformsBuffer.get(platforms);

        for(int i = 0; i<platformCount; i++){
            getDevices(platforms[i]);
        }
    }

    private static void getDevices(long platform) {
        PointerBuffer devicesBuffer = BufferUtils.createPointerBuffer(64);
        IntBuffer deviceCountBuffer = BufferUtils.createIntBuffer(1);
        int deviceResult = CL30.clGetDeviceIDs(platform, CL30.CL_DEVICE_TYPE_ALL, devicesBuffer, deviceCountBuffer);
        if(deviceResult != CL30.CL_SUCCESS){
            if(deviceResult == CL30.CL_INVALID_PLATFORM)throw new RuntimeException("Invalid platform");
            System.out.println("device incompatible #");
            return;
        }
        long[] devices = new long[64];
        int deviceCount = deviceCountBuffer.get();
        devicesBuffer.get(devices);

        for (int j = 0; j < deviceCount; j++) {
            getDeviceInfo(devices[j],CL30.CL_DEVICE_NAME);
            getDeviceInfo(devices[j],CL30.CL_DEVICE_VENDOR);
        }
    }

    private static void getDeviceInfo(long deviceID,int DeviceInfo) {
        ByteBuffer vendorNameBuffer = BufferUtils.createByteBuffer(256);
        PointerBuffer nameLengthBuffer = BufferUtils.createPointerBuffer(1);
        CL30.clGetDeviceInfo(deviceID,DeviceInfo,vendorNameBuffer,nameLengthBuffer);
        int length = (int) nameLengthBuffer.get();

        byte[] bytes = new byte[length-1];
        vendorNameBuffer.get(bytes,0,length-1);
        String s = new String(bytes);

        System.out.println(s);
    }

}
