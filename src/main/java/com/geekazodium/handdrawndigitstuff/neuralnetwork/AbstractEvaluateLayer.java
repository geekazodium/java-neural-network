package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class AbstractEvaluateLayer extends AbstractLayer implements EvaluateLayer {
    protected AbstractLayer previousLayer;

    public boolean training = false;
    //public float[] combinedInputs;
    public float[] weights;
    public float[] biases;

    public AbstractEvaluateLayer(int nodes) {
        super(nodes);
    }

    public void setPreviousLayer(AbstractLayer previousLayer){
        this.previousLayer = previousLayer;
    }

    public void initWeights(){
        int nodeCount = this.previousLayer.nodeCount;
        this.weights = new float[nodeCount*this.nodeCount];
        fillArrayWithRandomValues(this.weights);
    }
    public void initBiases(){
        this.biases = new float[this.nodeCount];
        fillArrayWithRandomValues(this.biases);
    }

    public AbstractLayer getPreviousLayer(){
        return previousLayer;
    }

    private void fillArrayWithRandomValues(float[] array){
        for (int i = 0; i <array.length; i++) {
            array[i] = (float) ((Math.random()*2d-1d)/10d);
        }
    }


    @Override
    public float[][] trainingEvaluate(ActivationFunction activationFunction, float[] previousLayerNodes) {
        float[] nodes = new float[this.nodeCount];
        System.arraycopy(biases, 0, nodes, 0, biases.length);
        int prevLayerCount = this.previousLayer.nodeCount;
        for (int p = 0;p < prevLayerCount; p++){
            for (int n = 0;n < this.nodeCount; n++){
                float effect = previousLayerNodes[p];
                effect*=this.weights[p+n*prevLayerCount];
                nodes[n] += effect;
            }
        }
        float[] preActivation = nodes.clone();
        applyActivationFunction(nodes, activationFunction);
        return new float[][]{nodes, preActivation};
    }

    public void enableTraining() {
        this.training = true;
    }

    public float[] getWeightDerivatives(float[] nodeDerivatives,float[] previousLayerNodes) {

        float[] weightDerivatives = new float[this.weights.length];

        int prevLayerCount = this.previousLayer.nodeCount;

        for (int p = 0;p < prevLayerCount; p++){
            float prevNodeActivation = previousLayerNodes[p];
            for (int n = 0;n < this.nodeCount; n++){// the previous node activation is the derivative of the weight to the value before activation f()
                weightDerivatives[p+n*prevLayerCount] = nodeDerivatives[n]*prevNodeActivation; // chain rule
            }
        }

        return weightDerivatives;
    }


    public float[] asyncGetWeightDerivatives(float[] nodeDerivatives,float[] prevLayerNodes) {

        float[] weightDerivatives = new float[this.weights.length];

        int prevLayerCount = this.previousLayer.nodeCount;

        for (int p = 0;p < prevLayerCount; p++){
            float prevNodeActivation = prevLayerNodes[p];
            for (int n = 0;n < this.nodeCount; n++){// the previous node activation is the derivative of the weight to the value before activation f()
                weightDerivatives[p+n*prevLayerCount] = nodeDerivatives[n]*prevNodeActivation; // chain rule
            }
        }

        return weightDerivatives;
    }
    public float[] getInputActivationDerivatives(float[] nodeDerivatives) {

        int prevLayerCount = this.previousLayer.nodeCount;

        float[] activationDerivatives = new float[prevLayerCount];

        for (int p = 0;p < prevLayerCount; p++){
            for (int n = 0;n < this.nodeCount; n++){// the previous node activation is the derivative of the weight to the value before activation f()
                activationDerivatives[p] += nodeDerivatives[n]*this.weights[p+n*prevLayerCount]; // chain rule
            }
        }

        return activationDerivatives;
    }


    private final AtomicInteger weightAccumulations = new AtomicInteger(0);
    private final AtomicBoolean writingToWeightAccumulator = new AtomicBoolean(false);
    private volatile float[] weightAccumulator = null;
    public void accumulateWeightChanges(float[] weightChanges){
        this.weightAccumulations.addAndGet(1);
        while (writingToWeightAccumulator.get()){
            Thread.onSpinWait();
        }
        writingToWeightAccumulator.set(true);
        if(this.weightAccumulator == null){
            this.weightAccumulator = new float[this.weights.length];
        }
        for (int i = 0; i < this.weightAccumulator.length; i++) {
            this.weightAccumulator[i] += weightChanges[i];
        }
        writingToWeightAccumulator.set(false);
    }

    private final AtomicInteger biasAccumulations = new AtomicInteger(0);
    private final AtomicBoolean writingToBiasAccumulator = new AtomicBoolean(false);
    private volatile float[] biasAccumulator;
    public void accumulateBiasChanges(float[] biasChanges){
        this.biasAccumulations.addAndGet(1);
        while (writingToBiasAccumulator.get()){
            Thread.onSpinWait();
        }
        writingToBiasAccumulator.set(true);
        if(this.biasAccumulator== null){
            this.biasAccumulator = new float[this.biases.length];
        }
        for (int i = 0; i < this.biasAccumulator.length; i++) {
            this.biasAccumulator[i] += biasChanges[i];
        }
        writingToBiasAccumulator.set(false);
    }


    /**
     * @deprecated just use the input values, the derivative of the bias with respect to the pre-activation is literally just 1, this is USELESS
     * @param nodeDerivatives
     * @return
     */
    @Deprecated
    public float[] getBiasesDerivatives(float[] nodeDerivatives){
        float[] biasDerivatives = new float[this.weights.length];

        // the previous node activation is the derivative of the weight to the value before activation f()
        // chain rule but the derivative of the bias with respect to the pre-activation is literally just 1
        if (this.nodeCount >= 0) System.arraycopy(nodeDerivatives, 0, biasDerivatives, 0, this.nodeCount);

        return biasDerivatives;
    }

    public void pushWeightAccumulator() {
        while (writingToWeightAccumulator.get()){
            Thread.onSpinWait();
        }
        for (int i = 0; i < this.weightAccumulator.length; i++) {
            float changes = this.weightAccumulator[i]/((float) this.weightAccumulations.get());
            this.weights[i] -= changes;
        }
        this.weightAccumulations.set(0);
        this.weightAccumulator = null;
    }

    public void pushBiasesAccumulator(){
        while (writingToBiasAccumulator.get()){
            Thread.onSpinWait();
        }
        for (int i = 0; i < this.biasAccumulator.length; i++) {
            float changes = this.biasAccumulator[i]/((float) this.biasAccumulations.get());
            this.biases[i] -= changes;
        }
        this.biasAccumulations.set(0);
        this.biasAccumulator = null;
    }

    public JsonObject serializeToJson(){
        JsonObject object = new JsonObject();
        object.add("weights", serializeFloatArray(this.weights));
        object.add("biases", serializeFloatArray(this.biases));
        object.addProperty("type", this.name());
        object.addProperty("nodes", this.nodeCount);
        return object;
    }

    @Override
    public float[] evaluate(float[] prevLayer) {
        float[] out = evaluateSelf(prevLayer);
        if(this instanceof NonFinalLayer thisLayer){
            AbstractLayer nextLayer = thisLayer.getNextLayer();
            return nextLayer.evaluate(out);
        }else {
            return out;
        }
    }
    @Override
    public float[] evaluateSelf(float[] prevLayer) {
        float[] out = new float[this.nodeCount];
        System.arraycopy(this.biases,0,out,0,this.nodeCount);
        int prevLayerCount = this.previousLayer.nodeCount;
        for (int p = 0;p < prevLayerCount; p++){
            for (int n = 0;n < this.nodeCount; n++){
                float effect = prevLayer[p];
                effect*=this.weights[p+n*prevLayerCount];
                out[n] += effect;
            }
        }
        applyActivationFunction(out, activationFunction);
        return out;
    }

    private void applyActivationFunction(float[] out, ActivationFunction activationFunction) {
        for (int i = 0; i < this.nodeCount; i++) {
            out[i] = activationFunction.activation(out[i]);
        }
    }

    private JsonArray serializeFloatArray(float[] array){
        JsonArray jsonArray = new JsonArray();
        for (float v : array) {
            jsonArray.add(v);
        }
        return jsonArray;
    }

    @Override
    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public abstract String name();
}
