package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public class LeakyRelU implements ActivationFunction{
    @Override
    public float activation(float in) {
        return  (in>0)?in:0.01f*in;
    }

    @Override
    public float derivative(float in) {
        return (in>0)?1:0.01f;
    }
}
