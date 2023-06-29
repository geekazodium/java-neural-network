package com.geekazodium.javaneuralnetwork.utils;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

public class NetworkFileFormatHelper {
    public static byte[] getLongBytes(long l){
        return ByteBuffer.allocate(Long.BYTES).putLong(l).array();
    }

    public static byte[] getIntBytes(int i) {
        return ByteBuffer.allocate(Integer.BYTES).putInt(i).array();
    }
    public static byte[] getFloatBytes(float f) {
        return ByteBuffer.allocate(Float.BYTES).putFloat(f).array();
    }
    public static int getFloatBytesAsInt(float f) {
        return ByteBuffer.allocate(Math.max(Float.BYTES,Integer.BYTES)).putFloat(f).rewind().getInt();
    }

    public static long writeFloatArray(int id, float[] array, FileOutputStream outputStream) throws IOException {
        long segmentLength = 0;
        int arrayLength = array.length;
        ByteBuffer byteBuffer = ByteBuffer.allocate(Integer.BYTES);

        byteBuffer.clear();
        outputStream.write(byteBuffer.putInt(id).array());
        byteBuffer.clear();
        outputStream.write(byteBuffer.putInt(arrayLength).array());

        segmentLength += Integer.BYTES * 2;

        ByteBuffer floatsBuffer = ByteBuffer.allocate(Float.BYTES * array.length);
        for (int i = 0, length = array.length; i < length; i++) {
            float f = array[i];
            floatsBuffer.putFloat(f);
        }
        outputStream.write(floatsBuffer.array());
        segmentLength += (long) Float.BYTES * arrayLength;

        return segmentLength;
    }

    private static ByteBuffer getFlagBytes(FileInputStream inputStream) throws IOException {
        byte[] idBytes = new byte[Integer.BYTES];
        inputStream.read(idBytes);
        ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES);
        buffer.put(idBytes);
        buffer.rewind();
        return buffer;
    }

    public static float[] readFloatArray(IntBuffer idBuffer, FileInputStream inputStream) throws IOException {
        int id = getFlagBytes(inputStream).getInt();
        idBuffer.rewind();
        idBuffer.put(id);
        int arrayLength = getFlagBytes(inputStream).getInt();
        float[] array = new float[arrayLength];

        byte[] bytes = inputStream.readNBytes(Float.BYTES * arrayLength);
        ByteBuffer floatsBuffer = ByteBuffer.allocate(bytes.length).put(bytes);
        floatsBuffer.rewind();
        for (int i = 0;i < arrayLength;i++) {
            float floatValue = floatsBuffer.getFloat();
            array[i] = floatValue;
        }

        idBuffer.rewind();
        return array;
    }

    public static int readNextInt(FileInputStream inputStream) throws IOException {
        byte[] bytes = inputStream.readNBytes(Integer.BYTES);
        return ByteBuffer.allocate(Integer.BYTES).put(bytes).rewind().getInt();
    }
}
