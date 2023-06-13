package com.geekazodium.handdrawndigitstuff.neuralnetwork.costfunctions;

import com.geekazodium.handdrawndigitstuff.neuralnetwork.CostFunction;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.trainingdatatypes.TextSection;

public class TokenPredictionCost implements CostFunction {
    private float next;

    public TokenPredictionCost() {
    }

    @Override
    public float[] cost(float[] outs, Object trainingDataObj) {
        TextSection textSection = ((TextSection) trainingDataObj);
        float[] costs = new float[textSection.inverseCharset.size()];
        for (int i = 0; i < outs.length; i++) {
            if (i == this.next) {
                costs[i] = (outs[i] - 1) * (outs[i] - 1);
            } else {
                costs[i] = (outs[i] - 0) * (outs[i] - 0);
            }
        }
        return costs;
    }

    public void setNext(float next) {
        this.next = next;
    }
//    @Override
//    public float[] cost(float[] outs, Object trainingDataObj) {
//        TrainingTextSection image = ((TrainingTextSection) trainingDataObj);
//        float[] costs = new float[10];
//        for (int i = 0; i < outs.length; i++) {
//            if (i == image.label) {
//                costs[i] = (outs[i] - 1) * (outs[i] - 1);
//            } else {
//                costs[i] = (outs[i] - 0) * (outs[i] - 0);
//            }
//        }
//        return costs;
//    }
//
    @Override
    public float[] derivative(float[] outs, Object trainingDataObj) {
        TextSection textSection = ((TextSection) trainingDataObj);
        float[] derivatives = new float[textSection.inverseCharset.size()];
        float cost = 0;
        for (int i = 0; i < outs.length; i++) {
            if (i == this.next) {
                derivatives[i] = 2 * (outs[i] - 1);
                cost += (outs[i] - 1) * (outs[i] - 1);
            } else {
                derivatives[i] = 2 * (outs[i] - 0);
                cost += (outs[i] - 0) * (outs[i] - 0);
            }
        }
        System.out.println("cost:" + cost);
        return derivatives;
    }
}