package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.geekazodium.handdrawndigitstuff.Main;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork.ResidualBlockFrame;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork.ResidualConcatBlock;
import com.google.gson.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class NeuralNetwork {
    public static final String SAVE_PATH = "Deep_network.json";
    private final OutputLayer outputLayer;
    private final InputLayer inputLayer;
    private final AbstractLayer[] layers;

    public NeuralNetwork(InputLayer inputLayer, EvaluateLayer[] internalLayers, OutputLayer outputLayer){
        this(inputLayer,internalLayers,outputLayer,false);
    }

    public NeuralNetwork(InputLayer inputLayer, EvaluateLayer[] internalLayers, OutputLayer outputLayer, boolean init){
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.layers = new AbstractLayer[internalLayers.length+2];
        this.layers[0] = this.inputLayer;
        this.layers[this.layers.length-1] = this.outputLayer;
        System.arraycopy(internalLayers, 0, this.layers, 1, internalLayers.length);
        initLayers(init);
        this.inputLayer.printStructure();
    }

    private void initLayers(boolean init) {
        for (int i = 0; i < this.layers.length-1; i++) {
            NonFinalLayer layer = (NonFinalLayer) this.layers[i];
            EvaluateLayer next = (EvaluateLayer) this.layers[i + 1];
            next.setPreviousLayer(((AbstractLayer)layer).getEnd());
            layer.setNextLayer(next);
            if(init) continue;
            next.init();
        }
    }

    public float[] evaluate(float[] inputs){
        return this.inputLayer.evaluate(inputs, null);
    }

    public void setActivationFunction(ActivationFunction activationFunction){
        for (AbstractLayer layer : this.layers) {
            if(layer instanceof EvaluateLayer evaluateLayer){
                evaluateLayer.setActivationFunction(activationFunction);
            }
        }
    }

    private void backpropagateMultithreaded(Object trainingDataObject,InputFunction inputFunction,CostFunction costFunction){
        float[] in = inputFunction.createInputs(trainingDataObject);
        this.inputLayer.backpropagate(in,costFunction,trainingDataObject);
    }

    public void batchMultithreaded(List<?> trainingDataObjects, InputFunction inputFunction, CostFunction costFunction, int trainingThreadLimit){
        final int toComplete = trainingDataObjects.size();
        final AtomicInteger completed = new AtomicInteger(0);
        final AtomicInteger active = new AtomicInteger(0);
        trainingDataObjects.forEach(o -> {
            while (trainingThreadLimit<=active.get()){
                Thread.onSpinWait();
            }
            active.addAndGet(1);
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    backpropagateMultithreaded(o,inputFunction,costFunction);
                    completed.addAndGet(1);
                    active.addAndGet(-1);
                }
            });
            thread.start();
        });
        while (toComplete>completed.get()){
            Thread.onSpinWait();
        }
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
            evaluateLayers.add(((SerializableToJsonLayer)this.layers[i]).serializeToJson());
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

        List<EvaluateLayer> hiddenLayers = new ArrayList<>();

        OutputLayer outLayer = null;

        for (JsonElement evaluateLayer : evaluateLayers) {
            AbstractLayer layer = (AbstractLayer) deserializeLayer(evaluateLayer);
            if(layer instanceof HiddenLayer hiddenLayer){
                hiddenLayers.add(hiddenLayer);
            }else if(layer instanceof OutputLayer outputLayer){
                outLayer = outputLayer;
            }else if(layer instanceof ResidualBlockFrame blockFrame){
                hiddenLayers.add(blockFrame);
            }
        }
        EvaluateLayer[] hiddenLayersArray = new EvaluateLayer[hiddenLayers.size()];
        hiddenLayers.toArray(hiddenLayersArray);
        return new NeuralNetwork(inLayer,hiddenLayersArray,outLayer,true);
    }

    public static SerializableToJsonLayer deserializeLayer(JsonElement evaluateLayer){
        JsonObject layerJson = evaluateLayer.getAsJsonObject();
        String type = layerJson.get("type").getAsString();

        SerializableToJsonLayer abstractLayer = null;
        if(Objects.equals(type, "HiddenLayer")){
            abstractLayer = new HiddenLayer(layerJson.get("nodes").getAsInt());
            abstractLayer.deserializeFromJson(layerJson);
        }else if(Objects.equals(type, "OutputLayer")){
            abstractLayer = new OutputLayer(layerJson.get("nodes").getAsInt());
            abstractLayer.deserializeFromJson(layerJson);
        }else if(Objects.equals(type,"ResidualBlock")){
            abstractLayer = new ResidualBlockFrame(layerJson.get("nodes").getAsInt());
            abstractLayer.deserializeFromJson(layerJson);
        }
        return abstractLayer;
    }

    public static void main(String[] args) throws Exception {
        int trainingThreadLimit = 5;

        String imagePath = "dataset"+File.separator+"train-images.idx3-ubyte";
        String labelPath = "dataset"+File.separator+"train-labels.idx1-ubyte";
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
        File networkFile = new File(SAVE_PATH);
        if (networkFile.exists()){
            neuralNetwork = deserialize(networkFile);
        }else {
            neuralNetwork = new NeuralNetwork(
                    new InputLayer(TrainingImage.width * TrainingImage.height),
                    new EvaluateLayer[]{
                            new ResidualBlockFrame(784, new AbstractLayer[]{
                                    new HiddenLayer(200),
                                    new HiddenLayer(100),
                                    new HiddenLayer(50)
                            }, new ResidualConcatBlock(784,50)),
                            new ResidualBlockFrame(784+50, new AbstractLayer[]{
                                    new HiddenLayer(200),
                                    new HiddenLayer(100),
                                    new HiddenLayer(50)
                            }, new ResidualConcatBlock(784+50,50)),//TODO gradient is not passed through layer properly
                            new HiddenLayer(200),
                            new HiddenLayer(100),
                            new HiddenLayer(50)
                    },
                    new OutputLayer(10)
            );
        }

        neuralNetwork.setActivationFunction(new LeakyRelU());

        int trainingSetSize = 6000;
        int batchSize = 1000;
        Random random = new Random();

        NumberRecognitionCost costFunction = new NumberRecognitionCost();
        for (int i = 0; i < 1000; i++) {
            int start = random.nextInt(trainingSetSize);

            long startTime = System.currentTimeMillis();
            neuralNetwork.batchMultithreaded(
                    trainingData.subList(start,start+batchSize),
                    trainingDataObject -> ((TrainingImage) trainingDataObject).getDataTransformed(
                            random.nextFloat(-0.4f,0.4f),
                            random.nextFloat(-6,6),
                            random.nextFloat(-6,6),
                            random.nextFloat(0.75f,2f)
                    ),
                    costFunction,
                    trainingThreadLimit
            );
            long now = System.currentTimeMillis();
            System.out.println("batch #"+(i%50+1)+" completed in:"+(now-startTime)+"ms");

            if(i%50 == 49){
                neuralNetwork.serialize(new File(SAVE_PATH));
                int total = 0;
                int correct = 0;
                int randint = random.nextInt(trainingSetSize-10);
                for (TrainingImage trainingImage : trainingData.subList(randint,randint+10)) {
                    float rotate = random.nextFloat(-0.4f,0.4f);
                    float x = random.nextFloat(-6,6);
                    float y = random.nextFloat(-6,6);
                    float scale = random.nextFloat(0.75f,2f);

                    trainingImage.log(rotate,x,y,scale);

                    float[] out = neuralNetwork.evaluate(
                            trainingImage.getDataTransformed(rotate,x,y,scale)
                    );
                    float highestVal = -100;
                    int highestIndex = -1;
                    for (int number = 0; number < out.length; number++) {
                        if(out[number] > highestVal){
                            highestIndex = number;
                            highestVal = out[number];
                        }
                    }

                    total++;
                    if(highestIndex == trainingImage.label){
                        correct++;
                    }
                    System.out.println("the neural network said:"+highestIndex);
                    System.out.println(Arrays.toString(out));
                }
                System.out.println("test accuracy:"+(float)correct/(float)total);
            }
        }
    }

}
