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

    public float[] evaluate(float[] inputs){
        this.inputLayer.setInputs(inputs);
        for (int i = 1; i < this.layers.length; i++) {
            this.layers[i].evaluate((in -> (in>0)?in:0.01f*in));
        }
        return this.outputLayer.getOutputs();
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


        NeuralNetwork neuralNetwork = new NeuralNetwork(
                new InputLayer(TrainingImage.width*TrainingImage.height),
                new HiddenLayer[]{
                        new HiddenLayer(100),
                        new HiddenLayer(100)
                },
                new OutputLayer(10)
        );

        for (int i = 0; i < trainingData.size(); i++) {
            trainingData.get(i).log();
            System.out.println(Arrays.toString(neuralNetwork.evaluate(trainingData.get(i).getData())));
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
