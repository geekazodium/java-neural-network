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
    public void log(float rotate, float x, float y,float scale){
        String asciiColor = " -=*%#";
        float asciiColorLength = asciiColor.length();
        for (int i = 0; i < width*height; i++) {
            if(i%width==0) {
                System.out.println();
            }
            System.out.print(" "+asciiColor.charAt((int)(this.getDataTransformed(rotate,x,y,scale)[i]*asciiColorLength)));
        }
        System.out.println();
        System.out.println(label+","+"rotation:"+rotate+" x:"+x+" y:"+y+" scale:"+scale);
    }

    public static int normalizeByte(byte b){
        return (b<0)?(int)b + 256:b;
    }

    public float[] getData() {
        return this.data;
    }

    public float[] getDataTransformed(float rotation, float shiftX, float shiftY,float scale){
        float[] dataPoints = new float[width*height];
        for (int i = 0; i < width * height; i++) {
            int x = i%width;
            int y = Math.floorDiv(i,width);
            x -= 14;
            y -= 14;
            float rotatedX = (float) (y*Math.sin(rotation)+x*Math.cos(rotation));
            float rotatedY = (float) (y*Math.cos(rotation)-x*Math.sin(rotation));
            rotatedX *= scale;
            rotatedY *= scale;
            rotatedX += 14;
            rotatedY += 14;
            rotatedX += shiftX;
            rotatedY += shiftY;
            x = (int) Math.floor(rotatedX);
            y = (int) Math.floor(rotatedY);
            int xEnd = x+1;
            int yEnd = y+1;
            float interpolateTop = lerp(getPixelAt(x,y),getPixelAt(xEnd,y),rotatedX%1f);
            float interpolateBottom = lerp(getPixelAt(x,yEnd),getPixelAt(xEnd,yEnd),rotatedX%1f);
            dataPoints[i] = lerp(interpolateTop,interpolateBottom,rotatedY%1f);
        }
        return dataPoints;
    }

    public float getPixelAt(int x, int y){
        if(x>=width||x<0)return 0;
        if(y>=height||y<0)return 0;
        return this.data[x+y*width];
    }
    public float lerp(float a,float b,float i){
        return a + (b - a) * i;
    }
}
