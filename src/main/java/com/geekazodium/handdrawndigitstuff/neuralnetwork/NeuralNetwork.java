package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.geekazodium.handdrawndigitstuff.Main;
import com.google.gson.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;

public class NeuralNetwork {
    private final OutputLayer outputLayer;
    private final InputLayer inputLayer;
    private final AbstractLayer[] layers;

    public NeuralNetwork(InputLayer inputLayer, HiddenLayer[] hiddenLayers, OutputLayer outputLayer){
        this(inputLayer,hiddenLayers,outputLayer,false);
    }

    public NeuralNetwork(InputLayer inputLayer, HiddenLayer[] hiddenLayers, OutputLayer outputLayer,boolean init){
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.layers = new AbstractLayer[hiddenLayers.length+2];
        this.layers[0] = this.inputLayer;
        this.layers[this.layers.length-1] = this.outputLayer;
        System.arraycopy(hiddenLayers, 0, this.layers, 1, hiddenLayers.length);
        initLayers(init);
    }

    private void initLayers(boolean init) {
        for (int i = 0; i < this.layers.length-1; i++) {
            NonFinalLayer layer = (NonFinalLayer) this.layers[i];
            AbstractEvaluateLayer next = (AbstractEvaluateLayer) this.layers[i + 1];
            next.setPreviousLayer((AbstractLayer) layer);
            layer.setNextLayer(next);
            if(init) continue;
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
        //float[] cost = costFunction.cost(out,trainingDataObject);

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

    //Training attempt #1 - failed
    //all weights and biases passed the float limit

    public static void main(String[] args) throws Exception {
        Path basePath = Path.of(Main.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        Path resourcesPath = Path.of(basePath.getParent().getParent().getParent().toString() + File.separator + "resources" + File.separator + "main");

        String imagePath = resourcesPath+File.separator+"train-images.idx3-ubyte";
        String labelPath = resourcesPath+File.separator+"train-labels.idx1-ubyte";
        File imageFile = new File(imagePath);
        File labelFile = new File(labelPath);
        FileInputStream imageStream = new FileInputStream(imageFile);
        FileInputStream labelStream = new FileInputStream(labelFile);
        byte[] imageFileBytes = imageStream.readAllBytes();
        byte[] labelStreamBytes = labelStream.readAllBytes();
        imageStream.close();
        labelStream.close();
        List<TrainingImage> trainingData = loadTrainingData(imageFileBytes,labelStreamBytes);
        NeuralNetwork neuralNetwork;
        File networkFile = new File("Network-784-100-100-10.json");
        if (networkFile.exists()){
            neuralNetwork = deserialize(networkFile);
        }else {
            neuralNetwork = new NeuralNetwork(
                            new InputLayer(TrainingImage.width * TrainingImage.height),
                            new HiddenLayer[]{
                                    new HiddenLayer(100),
                                    new HiddenLayer(100)
                            },
                            new OutputLayer(10)
            );
        }

        neuralNetwork.enableTraining();

        int trainingSetSize = 6000;
        int batchSize = 1000;
        Random random = new Random();

        for (int i = 0; i < 1000; i++) {
            int start = random.nextInt(trainingSetSize-batchSize);
            neuralNetwork.batch(
                    trainingData.subList(start,start+batchSize),
                    trainingDataObject -> ((TrainingImage) trainingDataObject).getDataTransformed(
                            random.nextFloat(-0.4f,0.4f),
                            random.nextFloat(-6,6),
                            random.nextFloat(-6,6),
                            random.nextFloat(1f,1.5f)
                    ),
                    new NumberRecognitionCost(),
                    new LeakyRelU()
            );
            if(i%50 == 0){
                neuralNetwork.serialize(new File("Network-784-100-100-10.json"));
                int randint = random.nextInt(trainingSetSize-10);
                for (TrainingImage trainingImage : trainingData.subList(randint,randint+10)) {
                    float rotate = random.nextFloat(-0.4f,0.4f);
                    float x = random.nextFloat(-6,6);
                    float y = random.nextFloat(-6,6);
                    float scale = random.nextFloat(1f,1.5f);

                    trainingImage.log(rotate,x,y,scale);

                    float[] out = neuralNetwork.evaluate(
                            trainingImage.getDataTransformed(rotate,x,y,scale),
                            new LeakyRelU()
                    );
                    float highestVal = -100;
                    int highestIndex = -1;
                    for (int number = 0; number < out.length; number++) {
                        if(out[number] > highestVal){
                            highestIndex = number;
                            highestVal = out[number];
                        }
                    }
                    System.out.println("the neural network said:"+highestIndex);
                    System.out.println(Arrays.toString(out));
                }
            }
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

    public void serialize(File file){
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        JsonObject object = new JsonObject();

        JsonObject inputLayer = new JsonObject();
        object.add("inputLayer",inputLayer);

        inputLayer.addProperty("nodes",this.inputLayer.nodeCount);

        JsonArray evaluateLayers = new JsonArray();

        for (int i = 1; i < this.layers.length; i++) {
            evaluateLayers.add(((AbstractEvaluateLayer)this.layers[i]).serializeToJson());
        }
        object.add("evaluateLayers",evaluateLayers);


        String out = gson.toJson(object);
        Runnable saveToFile = () -> {
            try {
                if (!file.exists()) {
                    file.createNewFile();
                }
                FileOutputStream outputStream = new FileOutputStream(file);
                outputStream.write(out.getBytes(StandardCharsets.UTF_8));
                outputStream.close();
            } catch(IOException e){
                System.out.println(e.getMessage());
            }
        };

        Thread saveThread = new Thread(saveToFile);
        saveThread.start();
    }

    public static NeuralNetwork deserialize(File file) throws IOException {
        FileInputStream inputStream = new FileInputStream(file);
        byte[] b = inputStream.readAllBytes();
        inputStream.close();
        String json = new String(b,StandardCharsets.UTF_8);
        JsonObject object = (JsonObject) JsonParser.parseString(json);
        int nodes = object.get("inputLayer").getAsJsonObject().get("nodes").getAsInt();
        InputLayer inLayer = new InputLayer(nodes);

        JsonArray evaluateLayers = object.get("evaluateLayers").getAsJsonArray();

        List<HiddenLayer> hiddenLayers = new ArrayList<>();

        OutputLayer outLayer = null;

        for (JsonElement evaluateLayer : evaluateLayers) {
            JsonObject layerJson = evaluateLayer.getAsJsonObject();
            String type = layerJson.get("type").getAsString();

            if(Objects.equals(type, "HiddenLayer")){
                HiddenLayer hiddenLayer = new HiddenLayer(layerJson.get("nodes").getAsInt());
                JsonArray weights = layerJson.get("weights").getAsJsonArray();
                copyWeights(hiddenLayer, weights);
                JsonArray biases = layerJson.get("biases").getAsJsonArray();
                copyBiases(hiddenLayer,biases);
                hiddenLayers.add(hiddenLayer);
            }else if(Objects.equals(type, "OutputLayer")){
                outLayer = new OutputLayer(layerJson.get("nodes").getAsInt());
                JsonArray weights = layerJson.get("weights").getAsJsonArray();
                copyWeights(outLayer, weights);
                JsonArray biases = layerJson.get("biases").getAsJsonArray();
                copyBiases(outLayer,biases);
            }
        }
        HiddenLayer[] hiddenLayersArray = new HiddenLayer[hiddenLayers.size()];
        hiddenLayers.toArray(hiddenLayersArray);
        return new NeuralNetwork(inLayer,hiddenLayersArray,outLayer,true);
    }

    private static void copyWeights(AbstractEvaluateLayer hiddenLayer, JsonArray array) {
        hiddenLayer.weights = new float[array.size()];
        for (int i = 0; i < array.size(); i++) {
            hiddenLayer.weights[i] = array.get(i).getAsFloat();
        }
    }
    private static void copyBiases(AbstractEvaluateLayer hiddenLayer, JsonArray array) {
        hiddenLayer.biases = new float[array.size()];
        for (int i = 0; i < array.size(); i++) {
            hiddenLayer.biases[i] = array.get(i).getAsFloat();
        }
    }
}
