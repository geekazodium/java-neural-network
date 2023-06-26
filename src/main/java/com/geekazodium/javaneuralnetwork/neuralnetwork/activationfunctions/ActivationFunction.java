package com.geekazodium.javaneuralnetwork.neuralnetwork.activationfunctions;

public interface ActivationFunction {
    float activation(float in);

    float derivative(float in);

    String getKernelString(String result);
    String getGradientKernelString(String gradient);

    default float[] derivative(float[] in){
        float[] d = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            d[i] = this.derivative(in[i]);
        }
        return d;
    }

}
