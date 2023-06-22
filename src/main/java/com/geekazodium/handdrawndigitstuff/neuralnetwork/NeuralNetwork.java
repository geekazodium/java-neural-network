package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.geekazodium.handdrawndigitstuff.GPUComputeContext;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.costfunctions.SimpleExpectedOutputCostFunction;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.costfunctions.TokenPredictionCost;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork.ResidualAddBlock;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork.ResidualBlockFrame;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork.ResidualConcatBlock;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.trainingdatatypes.TrainingText;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.trainingdatatypes.TextSection;
import com.google.gson.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class NeuralNetwork {
    public static final String SAVE_PATH = "Deep_Testing_network.json";
    private final OutputLayer outputLayer;
    private final InputLayer inputLayer;
    public final AbstractLayer[] layers;
    private float learnRate;

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


    public void batchMultithreaded(List<?> trainingDataObjects,TrainingFunction function, int trainingThreadLimit){
        this.batchCount++;
        final int toComplete = trainingDataObjects.size();
        final AtomicInteger completed = new AtomicInteger(0);
        final AtomicInteger active = new AtomicInteger(0);
        final NeuralNetwork thisNetwork = this;
        trainingDataObjects.forEach(o -> {
            while (trainingThreadLimit<=active.get()){
                Thread.onSpinWait();
            }
            active.addAndGet(1);
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    function.trainOnData(o,thisNetwork);
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
            if(!(layer instanceof EvaluateModifiableLayer evaluateLayer)) continue;
            evaluateLayer.pushChanges(this.learnRate);
        }
    }

    private float[] IndividualMultiply(float[] a, float[] b) {
        float[] der = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            der[i] = a[i]*b[i];
        }
        return der;
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

    private int batchCount = 0;

    public void serialize(File file){ //TODO find more RAM efficient way to store neural network in a file
        JsonObject object = new JsonObject();

        object.addProperty("batchCount",batchCount);

        JsonObject inputLayer = new JsonObject();
        object.add("inputLayer",inputLayer);

        inputLayer.addProperty("nodes",this.inputLayer.nodeCount);
        JsonArray evaluateLayers = new JsonArray();

        for (int i = 1; i < this.layers.length; i++) {
            evaluateLayers.add(((SerializableToJsonLayer)this.layers[i]).serializeToJson());
        }
        object.add("evaluateLayers",evaluateLayers);


        String out = object.toString();
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
        NeuralNetwork network = new NeuralNetwork(inLayer, hiddenLayersArray, outLayer, true);

        JsonElement count = object.get("batchCount");
        network.batchCount = (count==null)?0:count.getAsInt();
        return network;
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
        //int trainingThreadLimit = 4;

        int inputSize = 2;
        TextSection.setInputLength(inputSize);

        TrainingText trainingData = loadTrainingText(inputSize+1);

        int inputNeurons = trainingData.characterSet.size()*inputSize;
        int outputNeurons = trainingData.characterSet.size();

        NeuralNetwork neuralNetwork;
        File networkFile = new File(SAVE_PATH);
        if (networkFile.exists()){
            neuralNetwork = deserialize(networkFile);
        }else {
            neuralNetwork = new NeuralNetwork(
                    new InputLayer(inputNeurons),
                    new EvaluateLayer[]{
                            new ResidualBlockFrame(inputNeurons, new AbstractLayer[]{
                                    new HiddenLayer(10),
                                    new HiddenLayer(10),
                                    new HiddenLayer(10)
                            }, ResidualConcatBlock.instantiate(inputNeurons,10)),
                            new ResidualBlockFrame(inputNeurons+10, new AbstractLayer[]{
                                    new HiddenLayer(10),
                                    new HiddenLayer(10),
                                    new HiddenLayer(10)
                            }, new ResidualAddBlock(inputNeurons+10,10,0)),
                            new HiddenLayer(20),
                            new HiddenLayer(20),
                            new HiddenLayer(20)
                    },
                    new OutputLayer(outputNeurons)
            );
            neuralNetwork.serialize(new File(SAVE_PATH));
        }
        int stackSize = 4;

        //int batchSize = 12;

        GPUComputeContext gpuComputeContext = neuralNetwork.useGPUTrainingContext();

        neuralNetwork.setActivationFunction(new LeakyRelU());
        neuralNetwork.setLearnRate(1f/20f);

        gpuComputeContext.setStackSize(stackSize);
        gpuComputeContext.createNetworkBuffers();
        gpuComputeContext.createStackedLayerBuffers();
        gpuComputeContext.setKernelArgs();
        gpuComputeContext.createBackpropagationKernels();
        gpuComputeContext.updateStackSizeBuffer();
        gpuComputeContext.updateLearnRateBuffer(neuralNetwork.learnRate);

        SimpleExpectedOutputCostFunction expectedOutputCostFunction = new SimpleExpectedOutputCostFunction(gpuComputeContext, neuralNetwork.outputLayer.nodeCount,stackSize);
        gpuComputeContext.setCostFunctionKernel(expectedOutputCostFunction);

        TextSection section = trainingData.getExample();
        section.log();

        testExample(trainingData, neuralNetwork);

        for (int batchCounter = 0; batchCounter < 10000; batchCounter++) {
//            long startTime = System.currentTimeMillis();
//            neuralNetwork.batchMultithreaded(
//                    trainingData.getExamples(batchSize),
//                    new TokenPredictionTrainingFunction(),
//                    trainingThreadLimit
//            );
//            long now = System.currentTimeMillis();
//            System.out.println("batch #"+(batchCounter+1)+" completed in:"+(now-startTime)+"ms");
//
//            neuralNetwork.serialize(new File(SAVE_PATH));
//            testExample(trainingData,neuralNetwork);

            float[][] inputs = new float[stackSize][];
            float[][] expectedOuts = new float[stackSize][];
            for (int i = 0; i < stackSize; i++) {
                int characterIndex = i%inputSize;
                TextSection example = trainingData.getExample();
                float[] data = example.getData(characterIndex);
                inputs[i] = data;

                float[] output = new float[neuralNetwork.outputLayer.nodeCount];
                output[example.section.get(characterIndex)] = 1f;
                expectedOuts[i] = output;
            }
            expectedOutputCostFunction.setKernelExpectedResults(gpuComputeContext,expectedOuts);
            float[] stackedInputs = gpuComputeContext.stackInput(inputs);
            gpuComputeContext.setInputs(stackedInputs);
            gpuComputeContext.train();
            System.out.println("");
            testExample(trainingData, neuralNetwork);
            Thread.sleep(1000);
        }

        neuralNetwork.closeGPUTrainingContext();
    }

    public static final int DEFAULT_USE = 0;
    public static final int TRAINING_GPU = 1;
    private int networkContext = DEFAULT_USE;
    private GPUComputeContext gpuComputeContext;

    //special case layers - enqueue different kernel

    public GPUComputeContext useGPUTrainingContext() {
        if(networkContext != DEFAULT_USE)return this.gpuComputeContext;
        networkContext = TRAINING_GPU;
        this.gpuComputeContext = new GPUComputeContext();

        this.getDepth();
        this.uploadToGPU();

        for (AbstractLayer layer : this.layers) {
            System.out.println(layer.getIndex());
        }

        return this.gpuComputeContext;
    }

    public int getDepth() {
        int depth = 0;
        for (AbstractLayer layer : this.layers) {
            layer.assignIndex(depth);
            depth += layer.layerDepth();
        }
        return depth;
    }
    public void uploadToGPU(){
        gpuComputeContext.setNeuralNetwork(this);
        gpuComputeContext.createNetworkBuffers();
        gpuComputeContext.uploadNetworkToGPU();
        gpuComputeContext.compileNetworkLayerKernels();
    }

    public void closeGPUTrainingContext(){
        if(networkContext != TRAINING_GPU)return;
        networkContext = DEFAULT_USE;
        gpuComputeContext.delete();
    }

    private void setLearnRate(float learnRate) {
        this.learnRate = learnRate;
    }

    private static void testExample(TrainingText trainingData, NeuralNetwork neuralNetwork) {
        StringBuilder s = new StringBuilder("aaaaaaaaaa");
        for (int c = 0; c < 128; c++) {
            float[] inputs = TextSection.chunkString(s.toString(), trainingData.characterSet, trainingData.inverseCharset);
            float[] outs = neuralNetwork.evaluate(inputs);
            int index = 0;
            float highest = -10;
            for (int i = 0; i < outs.length; i++) {
                if (outs[i] > highest) {
                    highest = outs[i];
                    index = i;
                }
            }
            System.out.println(highest+","+Arrays.toString(outs));
            s.append(trainingData.inverseCharset.get(index));
            System.out.println(s.toString());
        }
    }

    private static TrainingText loadTrainingText(int chunkSize) throws IOException {
        String textPath = "datasetText"+File.separator+"dataset.txt";
        String charsetPath = "datasetText"+File.separator+"charset.txt";
        byte[] input = getBytes(textPath);
        byte[] charset = getBytes(charsetPath);
        TrainingText trainingText = new TrainingText(new String(input, StandardCharsets.UTF_8),new String(charset,StandardCharsets.UTF_8), chunkSize);
        return trainingText;
    }

    private static byte[] getBytes(String textPath) throws IOException {
        FileInputStream charSetStream = new FileInputStream(textPath);
        byte[] input = charSetStream.readAllBytes();
        charSetStream.close();
        return input;
    }

    private static List<TrainingImage> loadTrainingImages() throws IOException {
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
        trainingData.addAll(loadInputFails());
        return trainingData;
    }

    private static List<TrainingImage> subListOf(List<TrainingImage> trainingData, int start, int end) {
        List<TrainingImage> section = new ArrayList<>(end-start);
        int trainingDataSize = trainingData.size();
        for (int i = start; i < end; i++) {
            section.add(trainingData.get(i%trainingDataSize));
        }
        return section;
    }

    private static List<TrainingImage> loadInputFails() throws IOException {
        List<TrainingImage> inputFails = new ArrayList<>();
        File trainingDataFolder = new File("dataset");
        for (File file : trainingDataFolder.listFiles()) {
            if(!(file.getName().contains("inputFail")))continue;
            FileInputStream inputStream = new FileInputStream(file);
            byte[] bytes = inputStream.readAllBytes();
            inputStream.close();

            byte[] image = new byte[TrainingImage.width*TrainingImage.height];
            for (int i = 1; i < bytes.length; i++) {
                image[i-1] = bytes[i];
            }
            TrainingImage trainingImage = new TrainingImage(image,bytes[0]);
            inputFails.add(trainingImage);
        }
        return inputFails;
    }

    private static class TokenPredictionTrainingFunction implements TrainingFunction {

        private final TokenInputFunction inputFunction;

        public TokenPredictionTrainingFunction(){
            this.inputFunction = new TokenInputFunction();
        }

        @Override
        public void trainOnData(Object trainingDataObject, NeuralNetwork neuralNetwork) {
            TextSection textSection = (TextSection) trainingDataObject;
            for (int i = 0; i < TextSection.inputLength; i++) {
                System.out.println(i);
                float[] in = this.inputFunction.createInputs(trainingDataObject,i);
                TokenPredictionCost cost = new TokenPredictionCost();
                cost.setNext(textSection.section.get(i));
                neuralNetwork.inputLayer.backpropagate(in, cost, trainingDataObject);
            }
        }
    }

    private static class TokenInputFunction{
        public float[] createInputs(Object trainingDataObject,int endIndex) {
            return ((TextSection) trainingDataObject).getData(endIndex);
        }
    }
}
