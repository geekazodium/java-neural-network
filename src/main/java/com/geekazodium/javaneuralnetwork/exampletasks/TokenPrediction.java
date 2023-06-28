package com.geekazodium.javaneuralnetwork.exampletasks;

import com.geekazodium.javaneuralnetwork.GPUComputeContext;
import com.geekazodium.javaneuralnetwork.neuralnetwork.*;
import com.geekazodium.javaneuralnetwork.neuralnetwork.activationfunctions.LeakyRelU;
import com.geekazodium.javaneuralnetwork.neuralnetwork.costfunctions.SimpleExpectedOutputCostFunction;
import com.geekazodium.javaneuralnetwork.neuralnetwork.residualneuralnetwork.ResidualAddBlock;
import com.geekazodium.javaneuralnetwork.neuralnetwork.residualneuralnetwork.ResidualBlockFrame;
import com.geekazodium.javaneuralnetwork.neuralnetwork.trainingdatatypes.TextSection;
import com.geekazodium.javaneuralnetwork.neuralnetwork.trainingdatatypes.TrainingText;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.TreeSet;

public class TokenPrediction {
    public static final String SAVE_PATH = "capable of sending help.json";

    public static void main(String[] args) throws Exception {

        int inputSize = 128;
        TextSection.setInputLength(inputSize);

        TrainingText trainingData = loadTrainingText(inputSize+1);

        int inputNeurons = trainingData.characterSet.size()*inputSize;
        int outputNeurons = trainingData.characterSet.size();

        NeuralNetwork neuralNetwork;
        File networkFile = new File(SAVE_PATH);
        if (networkFile.exists()){
            neuralNetwork = NeuralNetwork.deserialize(networkFile);
        }else {
            int mergeChars = 16;
            neuralNetwork = new NeuralNetwork(
                    new InputLayer(inputNeurons),
                    new EvaluateLayer[]{
                            new ResidualBlockFrame(inputNeurons, new AbstractLayer[]{
                                    new HiddenLayer(512*3).initValMultiplier(0.007),
                                    new HiddenLayer(256*3).initValMultiplier(0.007),
                                    new HiddenLayer(trainingData.characterSet.size()* mergeChars).initValMultiplier(0.005)
                            }, new ResidualAddBlock(inputNeurons,trainingData.characterSet.size()* mergeChars,0)),
                            new ResidualBlockFrame(inputNeurons, new AbstractLayer[]{
                                    new HiddenLayer(512*3).initValMultiplier(0.007),
                                    new HiddenLayer(256*3).initValMultiplier(0.007),
                                    new HiddenLayer(trainingData.characterSet.size()* mergeChars).initValMultiplier(0.005)
                            }, new ResidualAddBlock(inputNeurons,trainingData.characterSet.size()* mergeChars,trainingData.characterSet.size()* mergeChars)),
                            new ResidualBlockFrame(inputNeurons, new AbstractLayer[]{
                                    new HiddenLayer(512*3).initValMultiplier(0.007),
                                    new HiddenLayer(256*3).initValMultiplier(0.007),
                                    new HiddenLayer(trainingData.characterSet.size()* mergeChars).initValMultiplier(0.005)
                            }, new ResidualAddBlock(inputNeurons,trainingData.characterSet.size()* mergeChars,trainingData.characterSet.size()* mergeChars *2)),
                            new HiddenLayer(512*4).initValMultiplier(0.01),
                            new HiddenLayer(256*4).initValMultiplier(0.02),
                            new HiddenLayer(128*5).initValMultiplier(0.025)
                    },
                    (OutputLayer) new OutputLayer(outputNeurons).initValMultiplier(0.03)
            );
            neuralNetwork.serializeToJson(new File(SAVE_PATH));
        }

        int batchSize = 24;
        int stackSize = inputSize*batchSize;

        GPUComputeContext gpuComputeContext = neuralNetwork.useGPUTrainingContext();

        neuralNetwork.setActivationFunction(new LeakyRelU());
        neuralNetwork.setLearnRate(0.25f);

        gpuComputeContext.setStackSize(stackSize);
        gpuComputeContext.createNetworkBuffers();
        gpuComputeContext.createStackedLayerBuffers();
        gpuComputeContext.compileNetworkLayerKernels();
        gpuComputeContext.createBackpropagationKernels();
        gpuComputeContext.updateStackSizeBuffer();
        gpuComputeContext.updateLearnRateBuffer(neuralNetwork.learnRate);

        SimpleExpectedOutputCostFunction expectedOutputCostFunction = new SimpleExpectedOutputCostFunction(gpuComputeContext, neuralNetwork.getOutputLayer().nodeCount,stackSize);
        gpuComputeContext.setCostFunctionKernel(expectedOutputCostFunction);

        TextSection section = trainingData.getExample();
        section.log();

        //runTestExample(trainingData,neuralNetwork);

        for (int batchCounter = 0; batchCounter < 30000; batchCounter++) {
            float[][] inputs = new float[stackSize][];
            float[][] expectedOuts = new float[stackSize][];
            TextSection example = null;
            for (int i = 0; i < stackSize; i++) {
                if(i%inputSize == 0){
                    example = trainingData.getNextExample();
                }
                int characterIndex = i%inputSize;
                float[] data = example.getData(characterIndex);
                inputs[i] = data;

                float[] output = new float[neuralNetwork.getOutputLayer().nodeCount];
                output[example.section.get(characterIndex+1)] = 1f;
                expectedOuts[i] = output;
            }
            expectedOutputCostFunction.setKernelExpectedResults(gpuComputeContext,expectedOuts);
            float[] stackedInputs = gpuComputeContext.stackInput(inputs);
            gpuComputeContext.setInputs(stackedInputs);
            boolean saveBatch = batchCounter % 5 == 5 - 1;

            gpuComputeContext.train(saveBatch);

            if(saveBatch){
                gpuComputeContext.downloadNetworkFromGPU();
                if(batchCounter%50 == 50 - 1)runTestExample(trainingData, neuralNetwork);
                neuralNetwork.serializeToJson(new File(SAVE_PATH));
                System.out.println("saving network");
            }
            System.out.println("batch #"+batchCounter);
            Thread.sleep(1000*30);
        }

        neuralNetwork.closeGPUTrainingContext();
    }

    private static void runTestExample(TrainingText trainingData, NeuralNetwork neuralNetwork) {
        Thread thread;
        thread = new Thread(new Runnable() {
            @Override
            public void run() {
                testExample(trainingData, neuralNetwork);
            }
        });
        thread.start();
    }

    private static void testExample(TrainingText trainingData, NeuralNetwork neuralNetwork) {
        char[] chars = new char[trainingData.inverseCharset.size()];
        trainingData.inverseCharset.forEach((integer, character) -> {
            chars[integer] = character;
        });
        StringBuilder s = new StringBuilder("How does the concentration of baking soda solution affect the rate of photosy");
        for (int c = 0; c < 32; c++) {
            float[] inputs = TextSection.chunkString(s.toString(), trainingData.characterSet, trainingData.inverseCharset);
            float[] outs = neuralNetwork.evaluate(inputs);

            //double[] normalized = new double[outs.length];

//            double expSum = 0;
//            for (float out : outs) {
//                if(out<=0)continue;
//                expSum += Math.pow(Math.E,out);
//            }
//
//            for (int i = 0; i < outs.length; i++) {
//                if(outs[i]<=0)continue;
//                normalized[i] = Math.pow(Math.E,outs[i])/expSum;
//            }

            int index = 0;
            float highest = -10;
            for (int i = 0; i < outs.length; i++) {
                if (outs[i] > highest) {
                    highest = outs[i];
                    index = i;
                }
            }
//
//            int chosenChar = 0;
//            Random random = new Random();
//            double randomCounter = random.nextDouble();
//            for (int i = 0; i < normalized.length; i++) {
//                randomCounter -= normalized[i];
//                if(randomCounter<0){
//                    chosenChar = i;
//                    break;
//                }
//            }

            StringBuilder sorted = new StringBuilder();

            TreeSet<IndexedOutputs> indexedOutputs = new TreeSet<>();
            for (int i = 0; i < outs.length; i++) {
                indexedOutputs.add(new IndexedOutputs(outs[i],chars[i]));
            }
            for (IndexedOutputs indexedOutput : indexedOutputs) {
                if(indexedOutput.value<=0)continue;
                sorted.append(indexedOutput.character + "," + indexedOutput.value + "\n");
            }

            System.out.println(sorted);

            s.append(trainingData.inverseCharset.get(index));
            System.out.println(s);
        }
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

    private record IndexedOutputs(float value, char character) implements Comparable<IndexedOutputs>{

        @Override
        public int compareTo(IndexedOutputs o) {
            return Float.compare(this.value, o.value);
        }
    }
}
