package com.geekazodium.handdrawndigitstuff;

public class FloatMatrix {
    public int height;
    public int width;
    public float[] data;

    public FloatMatrix(int v, int h) {
        this(new float[v * h], v, h);
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
            int x = (i + 1) % width;
            out.append(this.data[i]);
            if (i + 1 >= this.data.length) out.append("]");
            else {
                out.append(", ");
                if (x == 0) {
                    out.append("\n ");
                }
            }
        }
        return out.toString();
    }

    public void set(float value, int v, int h) {
        this.data[h + v * this.width] = value;
    }

    public int size() {
        return this.data.length;
    }
}
