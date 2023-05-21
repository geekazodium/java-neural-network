package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    public float[] evaluate(float[] inputs,ActivationFunction activationFunction){
        this.inputLayer.setInputs(inputs);
        for (int i = 1; i < this.layers.length; i++) {
            this.layers[i].evaluate(activationFunction);
        }
        return this.outputLayer.getOutputs();
    }

    private void backpropagate(Object trainingDataObject,InputFunction inputFunction,CostFunction costFunction,ActivationFunction activationFunction){
        float[] in = inputFunction.createInputs(trainingDataObject);
        float[] out = evaluate(in,activationFunction);
        float[] cost = costFunction.cost(out,trainingDataObject);

        AbstractLayer layer = this.outputLayer;
        float[] activationChanges = costFunction.derivative(out,trainingDataObject);
        while (layer instanceof AbstractEvaluateLayer){
            AbstractEvaluateLayer evaluateLayer = (AbstractEvaluateLayer) layer;
            // compute derivatives specific to out layer
            float[] activationDerivatives = activationFunction.derivative(evaluateLayer.combinedInputs);
            float[] nodeDerivatives = IndividualMultiply(activationChanges,activationDerivatives);

            // changes biases for out layer
            evaluateLayer.accumulateBiasChanges(nodeDerivatives);
            // changes for out layer
            float[] weightChanges = evaluateLayer.getWeightDerivatives(nodeDerivatives);
            evaluateLayer.accumulateWeightChanges(weightChanges);

            // derivative for 2nd layer
            activationChanges = evaluateLayer.getInputActivationDerivatives(nodeDerivatives);
            layer = evaluateLayer.previousLayer;
        }

        System.out.println(Arrays.toString(cost));
    }

    public void batch(List<?> trainingDataObjects,InputFunction inputFunction,CostFunction costFunction,ActivationFunction activationFunction){
        trainingDataObjects.forEach(o -> {
            this.backpropagate(o,inputFunction,costFunction,activationFunction);
        });
        for (AbstractLayer layer : this.layers) {
            if(!(layer instanceof AbstractEvaluateLayer evaluateLayer))continue;
            evaluateLayer.pushWeightAccumulator();
            evaluateLayer.pushBiasesAccumulator();
        }
    }

    private float[] IndividualMultiply(float[] a, float[] b) {
        float[] der = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            der[i] = a[i]*b[i];
        }
        return der;
    }

    public void enableTraining(){
        for (int i = 1; i < this.layers.length; i++) {
            if(!(this.layers[i] instanceof AbstractEvaluateLayer evaluateLayer))continue;
            evaluateLayer.enableTraining();
        }
    }

    public static void main(String[] args) throws Exception {
        String imagePath = "C:\\Users\\Geeka\\Documents\\GitHub\\handdrawn-digit-test\\src\\main\\resources\\train-images.idx3-ubyte";
        String labelPath = "C:\\Users\\Geeka\\Documents\\GitHub\\handdrawn-digit-test\\src\\main\\resources\\train-labels.idx1-ubyte";
        File imageFile = new File(imagePath);
        File labelFile = new File(labelPath);
        FileInputStream imageStream = new FileInputStream(imageFile);
        FileInputStream labelStream = new FileInputStream(labelFile);
        byte[] imageFileBytes = imageStream.readAllBytes();
        byte[] labelStreamBytes = labelStream.readAllBytes();
        imageStream.close();
        labelStream.close();
        List<TrainingImage> trainingData = loadTrainingData(imageFileBytes,labelStreamBytes);
        System.out.println(trainingData.size());

        NeuralNetwork neuralNetwork = new NeuralNetwork(
                new InputLayer(TrainingImage.width*TrainingImage.height),
                new HiddenLayer[]{
                        new HiddenLayer(100),
                        new HiddenLayer(100)
                },
                new OutputLayer(10)
        );

        neuralNetwork.enableTraining();


        neuralNetwork.batch(
                trainingData.subList(1000,1500),
                trainingDataObject -> ((TrainingImage) trainingDataObject).getData(),
                new NumberRecognitionCost(),
                new LeakyRelU()
        );
        neuralNetwork.batch(
                trainingData.subList(500,1000),
                trainingDataObject -> ((TrainingImage) trainingDataObject).getData(),
                new NumberRecognitionCost(),
                new LeakyRelU()
        );
        neuralNetwork.batch(
                trainingData.subList(0,500),
                trainingDataObject -> ((TrainingImage) trainingDataObject).getData(),
                new NumberRecognitionCost(),
                new LeakyRelU()
        );

        for (TrainingImage trainingImage : trainingData) {
            trainingImage.log();
            neuralNetwork.backpropagate(
                    trainingImage,
                    trainingDataObject -> ((TrainingImage) trainingDataObject).getData(),
                    new NumberRecognitionCost(),
                    new LeakyRelU()
            );
        }
    }

    public static class NumberRecognitionCost implements CostFunction {
        @Override
        public float[] cost(float[] outs, Object trainingDataObj){
            TrainingImage image = ((TrainingImage) trainingDataObj);
            float[] costs = new float[10];
            for (int i = 0; i < outs.length; i++) {
                if(i == image.label){
                    costs[i]=(outs[i]-1)*(outs[i]-1);
                }else {
                    costs[i]=(outs[i]-0)*(outs[i]-0);
                }
            }
            return costs;
        }

        @Override
        public float[] derivative(float[] outs, Object trainingDataObj) {
            TrainingImage image = ((TrainingImage) trainingDataObj);
            float[] derivatives = new float[10];
            for (int i = 0; i < outs.length; i++) {
                if(i == image.label){
                    derivatives[i]=2*(outs[i]-1);
                }else {
                    derivatives[i]=2*(outs[i]-0);
                }
            }
            return derivatives;
        }
    }

    protected static List<TrainingImage> loadTrainingData(byte[] imageFileBytes, byte[] labelStreamBytes){
        List<TrainingImage> images = new ArrayList<>();
        byte[] image = new byte[TrainingImage.width*TrainingImage.height];
        int c = 0;
        int labelReaderIndex = 8;
        for (int i = 16;i<imageFileBytes.length;i++) {
            image[c] = imageFileBytes[i];
            c++;
            if (c >= TrainingImage.width*TrainingImage.height) {
                c = 0;
                images.add(new TrainingImage(image.clone(),labelStreamBytes[labelReaderIndex]));
                labelReaderIndex++;
            }
        }
        return images;
    }

}
