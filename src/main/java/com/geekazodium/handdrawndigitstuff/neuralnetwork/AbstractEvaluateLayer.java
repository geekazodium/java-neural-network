package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class AbstractEvaluateLayer extends AbstractLayer implements EvaluateLayer,SerializableToJsonLayer {
    protected AbstractLayer previousLayer;
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
            array[i] = (float) ((Math.random()*2d-1d)/20d);
        }
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

    public void pushWeightAccumulator(float learnRate) {
        while (writingToWeightAccumulator.get()){
            Thread.onSpinWait();
        }
        for (int i = 0; i < this.weightAccumulator.length; i++) {
            float changes = this.weightAccumulator[i]/((float) this.weightAccumulations.get());
            this.weights[i] -= changes*learnRate;
        }
        this.weightAccumulations.set(0);
        this.weightAccumulator = null;
    }

    public void pushBiasesAccumulator(float learnRate){
        while (writingToBiasAccumulator.get()){
            Thread.onSpinWait();
        }
        for (int i = 0; i < this.biasAccumulator.length; i++) {
            float changes = this.biasAccumulator[i]/((float) this.biasAccumulations.get());
            this.biases[i] -= changes*learnRate;
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
    public void deserializeFromJson(JsonObject object) {
        JsonArray weights = object.get("weights").getAsJsonArray();
        copyWeights(this,weights);
        JsonArray biases = object.get("biases").getAsJsonArray();
        copyBiases(this,biases);
    }

    public static void copyWeights(AbstractEvaluateLayer hiddenLayer, JsonArray array) {
        hiddenLayer.weights = new float[array.size()];
        for (int i = 0; i < array.size(); i++) {
            hiddenLayer.weights[i] = array.get(i).getAsFloat();
        }
    }
    public static void copyBiases(AbstractEvaluateLayer hiddenLayer, JsonArray array) {
        hiddenLayer.biases = new float[array.size()];
        for (int i = 0; i < array.size(); i++) {
            hiddenLayer.biases[i] = array.get(i).getAsFloat();
        }
    }

    @Override
    public float[] evaluate(float[] prevLayer, Object[] args) {
        float[] out = evaluateSelf(prevLayer, args);
        if(this instanceof NonFinalLayer thisLayer){
            EvaluateLayer nextLayer = thisLayer.getNextLayer();
            return nextLayer.evaluate(out, args);
        }else {
            return out;
        }
    }
    @Override
    public float[] evaluateSelf(float[] prevLayer, Object[] args) {
        float[] out = getPreActivation(this.biases, this.nodeCount, prevLayer);
        applyActivationFunction(out, activationFunction);
        return out;
    }

    private float[] getPreActivation(float[] biases, int nodeCount, float[] prevLayer) {
        float[] out = new float[this.nodeCount];
        System.arraycopy(biases, 0, out, 0, nodeCount);
        int prevLayerCount = this.previousLayer.nodeCount;
        for (int p = 0; p < prevLayerCount; p++) {
            for (int n = 0; n < this.nodeCount; n++) {
                float effect = prevLayer[p];
                effect *= this.weights[p + n * prevLayerCount];
                out[n] += effect;
            }
        }
        return out;
    }

    private void applyActivationFunction(float[] out, ActivationFunction activationFunction) {
        for (int i = 0; i < this.nodeCount; i++) {
            out[i] = activationFunction.activation(out[i]);
        }
    }

    private static JsonArray serializeFloatArray(float[] array){
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

    @Override
    public float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject, Object[] args) {
        float[] preActivation = this.getPreActivation(this.biases,this.nodeCount,in);
        float[] activation = preActivation.clone();
        applyActivationFunction(activation, this.activationFunction);
        float[] activationChanges;
        if(this instanceof NonFinalLayer self){
            EvaluateLayer nextEvaluateLayer = self.getNextLayer();
            activationChanges = nextEvaluateLayer.backpropagate(activation,costFunction,trainingDataObject,args);
        }else{
            activationChanges = costFunction.derivative(activation,trainingDataObject);
        }

        float[] activationDerivatives = activationFunction.derivative(preActivation);
        float[] nodeDerivatives = IndividualMultiply(activationChanges,activationDerivatives);

        float[] weightChanges = this.asyncGetWeightDerivatives(nodeDerivatives,in);

        this.accumulateWeightChanges(weightChanges);

        this.accumulateBiasChanges(nodeDerivatives);

        return this.getInputActivationDerivatives(nodeDerivatives);
    }

    private float[] IndividualMultiply(float[] a, float[] b) {
        float[] der = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            der[i] = a[i]*b[i];
        }
        return der;
    }

    @Override
    public void init() {
        this.initBiases();
        this.initWeights();
    }

    public abstract String name();

    @Override
    public AbstractLayer getEnd() {
        return this;
    }
}
