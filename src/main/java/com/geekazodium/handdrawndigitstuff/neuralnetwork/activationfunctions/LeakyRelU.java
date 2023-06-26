package com.geekazodium.handdrawndigitstuff.neuralnetwork.activationfunctions;

public class LeakyRelU implements ActivationFunction {
    @Override
    public float activation(float in) {
        return  (in>0)?in:0.01f*in;
    }

    @Override
    public float derivative(float in) {
        return (in>0)?1:0.01f;
    }

    @Override
    public String getKernelString(String result) {
        return " ("+result+">0)?"+result+":"+result+"* 0.01f;\n";
    }

    @Override
    public String getGradientKernelString(String gradient) {
        return " (("+gradient+">0)? 1.0f : 0.01f);\n";
    }
}
