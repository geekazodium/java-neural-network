package com.geekazodium.javaneuralnetwork.neuralnetwork;

import com.geekazodium.javaneuralnetwork.GPUComputeContext;
import com.geekazodium.javaneuralnetwork.neuralnetwork.activationfunctions.ActivationFunction;
import com.geekazodium.javaneuralnetwork.neuralnetwork.residualneuralnetwork.ResidualBlockFrame;
import com.geekazodium.javaneuralnetwork.neuralnetwork.trainingdatatypes.TrainingImage;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class NeuralNetwork {
    private final OutputLayer outputLayer;
    private final InputLayer inputLayer;
    public final AbstractLayer[] layers;
    public float learnRate;

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

    public void incrementBatchCount(){
        batchCount++;
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

    public void serializeToJson(File file){ //TODO find more RAM efficient way to store neural network in a file
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

    public static void main(String[] args) throws IOException {
        FileOutputStream outputStream = new FileOutputStream(new File("NeuralNetwork/out"));
        writeFloatArray(10,new float[]{0f,2f,1.5f,9.69f},outputStream);
        outputStream.flush();
        float[] array = new float[259526];
        for (int i = 0; i < array.length; i++) {
            array[i] = (float) Math.random();
        }
        writeFloatArray(13, array,outputStream);
        outputStream.flush();
        writeFloatArray(10,new float[]{1000f,10000000f,659f,989f},outputStream);
        outputStream.flush();
        writeFloatArray(10,new float[]{192f,3.1415926f,0.69f,9.69f},outputStream);
        outputStream.flush();
        writeFloatArray(10,new float[]{0f,2f,1.5f,9.69f},outputStream);
        outputStream.flush();
        outputStream.close();

        FileInputStream inputStream = new FileInputStream(new File("NeuralNetwork/out"));
        readAndLogArray(inputStream);
        readAndLogArray(inputStream);
        readAndLogArray(inputStream);
        readAndLogArray(inputStream);
        readAndLogArray(inputStream);
        inputStream.close();
    }

    private static float[] readAndLogArray(FileInputStream inputStream) throws IOException {
        IntBuffer buffer = IntBuffer.allocate(1);
        float[] floats = readFloatArray(buffer, inputStream);
        System.out.println(Arrays.toString(floats));
        return floats;
    }

    public void serialize(File file) throws IOException {
        FileOutputStream outputStream = new FileOutputStream(file);
        long writeIndex = 0;
        List<AbstractLayer> allLayers = getAllLayers();
        for (AbstractLayer layer : allLayers) {
            outputStream.write(layer.getId());
        }
        outputStream.flush();
        outputStream.close();
    }

    private List<AbstractLayer> getAllLayers() {
        List<AbstractLayer> unstackedLayers = new ArrayList<>();
        for (AbstractLayer layer : this.layers) {
            unstackedLayers.addAll(List.of(layer.getAsLayerArray()));
        }
        return unstackedLayers;
    }

    public static long writeFloatArray(int id, float[] array, FileOutputStream outputStream) throws IOException {
        long segmentLength = 0;
        int arrayLength = array.length;
        ByteBuffer byteBuffer = ByteBuffer.allocate(Integer.BYTES);

        byteBuffer.clear();
        outputStream.write(byteBuffer.putInt(id).array());
        byteBuffer.clear();
        outputStream.write(byteBuffer.putInt(arrayLength).array());

        segmentLength += Integer.BYTES * 2;

        ByteBuffer floatsBuffer = ByteBuffer.allocate(Float.BYTES * array.length);
        for (int i = 0, length = array.length; i < length; i++) {
            float f = array[i];
            floatsBuffer.putFloat(f);
        }
        outputStream.write(floatsBuffer.array());
        segmentLength += (long) Float.BYTES * arrayLength;

        return segmentLength;
    }

    public static float[] readFloatArray(IntBuffer idBuffer, FileInputStream inputStream) throws IOException {
        int id = getFlagBytes(inputStream).getInt();
        idBuffer.rewind();
        idBuffer.put(id);
        int arrayLength = getFlagBytes(inputStream).getInt();
        float[] array = new float[arrayLength];

        byte[] bytes = inputStream.readNBytes(Float.BYTES * arrayLength);
        ByteBuffer floatsBuffer = ByteBuffer.allocate(bytes.length).put(bytes);
        floatsBuffer.rewind();
        for (int i = 0;i < arrayLength;i++) {
            float floatValue = floatsBuffer.getFloat();
            array[i] = floatValue;
        }

        return array;
    }

    private static ByteBuffer getFlagBytes(FileInputStream inputStream) throws IOException {
        byte[] idBytes = new byte[Integer.BYTES];
        inputStream.read(idBytes);
        ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES);
        buffer.put(idBytes);
        buffer.rewind();
        return buffer;
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
    }

    public void closeGPUTrainingContext(){
        if(networkContext != TRAINING_GPU)return;
        networkContext = DEFAULT_USE;
        gpuComputeContext.delete();
    }

    public void setLearnRate(float learnRate) {
        this.learnRate = learnRate;
    }

    public InputLayer getInputLayer() {
        return this.inputLayer;
    }

    public OutputLayer getOutputLayer() {
        return this.outputLayer;
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
}
