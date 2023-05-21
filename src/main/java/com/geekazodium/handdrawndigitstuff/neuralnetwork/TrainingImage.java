package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class TrainingImage {
    private final float[] data;
    public final byte label;
    public static final int height = 28;
    public static final int width = 28;

    public TrainingImage(byte[] data, byte label) {
        this.data = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            this.data[i] = normalizeByte(data[i])/256f;
        }
        this.label = label;
    }
    public void log(){
        String asciiColor = " -=*%#";
        float asciiColorLength = asciiColor.length();
        for (int i = 0; i < width*height; i++) {
            if(i%width==0) {
                System.out.println();
            }
            System.out.print(" "+asciiColor.charAt((int)(this.data[i]*asciiColorLength)));
        }
        System.out.println();
        System.out.println(label);
    }

    public static int normalizeByte(byte b){
        return (b<0)?(int)b + 256:b;
    }

    public float[] getData() {
        return this.data;
    }
}
