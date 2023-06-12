package com.geekazodium.handdrawndigitstuff;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;
import org.lwjgl.system.FunctionProviderLocal;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.system.Platform;

import java.awt.*;
import java.nio.FloatBuffer;

public class Test {
    public static void main(String[] args){
//        FloatBuffer a = BufferUtils.createFloatBuffer(10);
//        FloatBuffer b = BufferUtils.createFloatBuffer(10);
//        FloatBuffer out = BufferUtils.createFloatBuffer(a.capacity());
//        a.put(new float[]{1f,2f,3f,4f,5f,6f,7f,8f,9f,10f});
//        b.put(new float[]{1f,2f,3f,4f,5f,6f,7f,8f,9f,10f});


        int[] platforms = new int[1];

        PointerBuffer platformsPointer = MemoryUtil.memAllocPointer(32);
        int success = CL30.clGetPlatformIDs(platformsPointer,platforms);

        if(success == CL30.CL_SUCCESS){
            System.out.println("yay");
            System.out.println(platformsPointer.get());
        }else {
            System.out.println("failed:" + success);
        }
    }
}
