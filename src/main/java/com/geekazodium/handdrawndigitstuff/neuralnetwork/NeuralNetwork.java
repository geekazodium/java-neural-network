package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import java.util.Arrays;

public class NeuralNetwork {
    private final OutputLayer outputLayer;
    private final InputLayer inputLayer;
    private final AbstractLayer[] layers;

    public NeuralNetwork(InputLayer inputLayer, HiddenLayer[] hiddenLayers, OutputLayer outputLayer){
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.layers = new AbstractLayer[hiddenLayers.length+2];
        this.layers[0] = this.inputLayer;
        this.layers[this.layers.length-1] = this.outputLayer;
        System.arraycopy(hiddenLayers, 0, this.layers, 1, hiddenLayers.length);
        initLayers();
    }

    private void initLayers() {
        for (int i = 0; i < this.layers.length-1; i++) {
            NonFinalLayer layer = (NonFinalLayer) this.layers[i];
            AbstractEvaluateLayer next = (AbstractEvaluateLayer) this.layers[i + 1];
            next.setPreviousLayer((AbstractLayer) layer);
            layer.setNextLayer(next);
            next.initBiases();
            next.initWeights();
        }
    }

    public float[] evaluate(float[] inputs){
        this.inputLayer.setInputs(inputs);
        for (int i = 1; i < this.layers.length; i++) {
            this.layers[i].evaluate((in -> (in>0)?in:0.01f*in));
        }
        return this.outputLayer.getOutputs();
    }

    public static void main(String[] args){
        NeuralNetwork neuralNetwork = new NeuralNetwork(
                new InputLayer(28*28),
                new HiddenLayer[]{
                        new HiddenLayer(20),
                        new HiddenLayer(20)
                },
                new OutputLayer(10)
        );
        neuralNetwork.evaluate(new float[]{
                0,0,0,0,0,0,1,1,1,1
        });
        System.out.println(Arrays.toString(neuralNetwork.outputLayer.getOutputs()));
    }
}
