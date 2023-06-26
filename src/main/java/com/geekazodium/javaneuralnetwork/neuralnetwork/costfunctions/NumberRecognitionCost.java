package com.geekazodium.javaneuralnetwork.neuralnetwork.costfunctions;

import com.geekazodium.javaneuralnetwork.GPUComputeContext;
import com.geekazodium.javaneuralnetwork.neuralnetwork.CostFunction;
import com.geekazodium.javaneuralnetwork.neuralnetwork.RunnableKernel;
import com.geekazodium.javaneuralnetwork.neuralnetwork.TrainingImage;

public class NumberRecognitionCost implements CostFunction {
    @Override
    public float[] cost(float[] outs, Object trainingDataObj) {
        TrainingImage image = ((TrainingImage) trainingDataObj);
        float[] costs = new float[10];
        for (int i = 0; i < outs.length; i++) {
            if (i == image.label) {
                costs[i] = (outs[i] - 1) * (outs[i] - 1);
            } else {
                costs[i] = (outs[i] - 0) * (outs[i] - 0);
            }
        }
        return costs;
    }

    @Override
    public float[] derivative(float[] outs, Object trainingDataObj) {
        TrainingImage image = ((TrainingImage) trainingDataObj);
        float[] derivatives = new float[10];
        for (int i = 0; i < outs.length; i++) {
            if (i == image.label) {
                derivatives[i] = 2 * (outs[i] - 1);
            } else {
                derivatives[i] = 2 * (outs[i] - 0);
            }
        }
        return derivatives;
    }

    @Override
    public RunnableKernel createKernel(GPUComputeContext context) {
        return null;
    }
}
