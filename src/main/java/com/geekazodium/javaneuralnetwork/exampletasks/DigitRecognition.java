package com.geekazodium.javaneuralnetwork.exampletasks;

import com.geekazodium.javaneuralnetwork.GPUComputeContext;
import com.geekazodium.javaneuralnetwork.neuralnetwork.*;
import com.geekazodium.javaneuralnetwork.neuralnetwork.activationfunctions.LeakyRelU;
import com.geekazodium.javaneuralnetwork.neuralnetwork.costfunctions.NumberRecognitionCost;
import com.geekazodium.javaneuralnetwork.neuralnetwork.costfunctions.SimpleExpectedOutputCostFunction;
import com.geekazodium.javaneuralnetwork.neuralnetwork.costfunctions.TokenPredictionCost;
import com.geekazodium.javaneuralnetwork.neuralnetwork.residualneuralnetwork.ResidualAddBlock;
import com.geekazodium.javaneuralnetwork.neuralnetwork.residualneuralnetwork.ResidualBlockFrame;
import com.geekazodium.javaneuralnetwork.neuralnetwork.residualneuralnetwork.ResidualConcatBlock;
import com.geekazodium.javaneuralnetwork.neuralnetwork.trainingdatatypes.TextSection;
import com.geekazodium.javaneuralnetwork.neuralnetwork.trainingdatatypes.TrainingImage;
import com.geekazodium.javaneuralnetwork.neuralnetwork.trainingdatatypes.TrainingText;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class DigitRecognition {
    public static final String SAVE_PATH = "better digit recognition.json";

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

    public static void main(String[] args) throws Exception {

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

        int trainingSetSize = 6000;
        int stackSize = 1000;
        int inputSize = 784;

        if (networkFile.exists()){
            neuralNetwork = NeuralNetwork.deserialize(networkFile);
        }else {
            neuralNetwork = new NeuralNetwork(
                    new InputLayer(TrainingImage.width * TrainingImage.height),
                    new EvaluateLayer[]{
                            new ResidualBlockFrame(inputSize, new AbstractLayer[]{
                                    new HiddenLayer(200),
                                    new HiddenLayer(100),
                                    new HiddenLayer(50)
                            }, ResidualConcatBlock.instantiate(inputSize,50)),
                            new ResidualBlockFrame(inputSize+50, new AbstractLayer[]{
                                    new HiddenLayer(200),
                                    new HiddenLayer(100),
                                    new HiddenLayer(50)
                            }, ResidualConcatBlock.instantiate(inputSize+50,50)),
                            new HiddenLayer(200),
                            new HiddenLayer(100),
                            new HiddenLayer(50)
                    },
                    new OutputLayer(10)
            );
        }

        neuralNetwork.setActivationFunction(new LeakyRelU());
        Random random = new Random();

        GPUComputeContext gpuComputeContext = neuralNetwork.useGPUTrainingContext();

        neuralNetwork.setActivationFunction(new LeakyRelU());
        neuralNetwork.setLearnRate(1f);

        gpuComputeContext.setStackSize(stackSize);
        gpuComputeContext.createNetworkBuffers();
        gpuComputeContext.createStackedLayerBuffers();
        gpuComputeContext.compileNetworkLayerKernels();
        gpuComputeContext.createBackpropagationKernels();
        gpuComputeContext.updateStackSizeBuffer();
        gpuComputeContext.updateLearnRateBuffer(neuralNetwork.learnRate);

        SimpleExpectedOutputCostFunction expectedOutputCostFunction = new SimpleExpectedOutputCostFunction(gpuComputeContext, neuralNetwork.getOutputLayer().nodeCount, stackSize);
        gpuComputeContext.setCostFunctionKernel(expectedOutputCostFunction);

        for (int batchCount = 0; batchCount < 1000; batchCount++) {
            long startTime = System.currentTimeMillis();
            int start = random.nextInt(trainingSetSize);
            List<TrainingImage> trainingImages = subListOf(trainingData,start, start + stackSize);

            float[][] inputs = new float[stackSize][];
            float[][] expectedOuts = new float[stackSize][];
            for (int i = 0; i < stackSize; i++) {
                TrainingImage image = trainingImages.get(i);
                float[] data = image.getDataTransformed(
                        random.nextFloat(-0.4f,0.4f),
                        random.nextFloat(-6,6),
                        random.nextFloat(-6,6),
                        random.nextFloat(0.75f,2f)
                );
                inputs[i] = data;

                float[] output = new float[neuralNetwork.getOutputLayer().nodeCount];
                output[image.label] = 1f;
                expectedOuts[i] = output;
            }
            expectedOutputCostFunction.setKernelExpectedResults(gpuComputeContext,expectedOuts);
            float[] stackedInputs = gpuComputeContext.stackInput(inputs);
            gpuComputeContext.setInputs(stackedInputs);
            boolean saveBatch = batchCount% 50 == 50 - 1;

            gpuComputeContext.train(saveBatch);

            long now = System.currentTimeMillis();
            System.out.println("batch #"+(batchCount+1)+" completed in:"+(now-startTime)+"ms");;

            if(saveBatch){
                gpuComputeContext.downloadNetworkFromGPU();
                neuralNetwork.serialize(new File(SAVE_PATH));
                int total = 0;
                int correct = 0;
                int randomInt = random.nextInt(trainingSetSize-10);
                List<TrainingImage> testImages = trainingData.subList(randomInt, randomInt + 10);
                for (TrainingImage trainingImage : testImages) {
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

    private static class TokenInputFunction{
        public float[] createInputs(Object trainingDataObject,int endIndex) {
            return ((TextSection) trainingDataObject).getData(endIndex);
        }
    }

    private static void runTestExample(NeuralNetwork neuralNetwork) {
        Thread thread;
        thread = new Thread(new Runnable() {
            @Override
            public void run() {
                testExample(neuralNetwork);
            }
        });
        thread.start();
    }

    private static void testExample(NeuralNetwork neuralNetwork) {
    }

    private static byte[] getBytes(String textPath) throws IOException {
        FileInputStream charSetStream = new FileInputStream(textPath);
        byte[] input = charSetStream.readAllBytes();
        charSetStream.close();
        return input;
    }

    private static TrainingText loadTrainingText(int chunkSize) throws IOException {
        String textPath = "datasetText"+File.separator+"dataset.txt";
        String charsetPath = "datasetText"+File.separator+"charset.txt";
        byte[] input = getBytes(textPath);
        byte[] charset = getBytes(charsetPath);
        return new TrainingText(new String(input, StandardCharsets.UTF_8),new String(charset,StandardCharsets.UTF_8), chunkSize);
    }

    private static <T> List<T> subListOf(List<T> trainingData, int start, int end) {
        List<T> section = new ArrayList<>(end-start);
        int trainingDataSize = trainingData.size();
        for (int i = start; i < end; i++) {
            section.add(trainingData.get(i%trainingDataSize));
        }
        return section;
    }
}
