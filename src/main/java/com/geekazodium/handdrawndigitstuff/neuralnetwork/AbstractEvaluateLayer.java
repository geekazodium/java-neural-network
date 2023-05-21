package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public abstract class AbstractEvaluateLayer extends AbstractLayer{
    protected AbstractLayer previousLayer;

    public boolean training = false;
    public float[] combinedInputs;
    protected float[] weights;
    protected float[] biases;

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

    private void fillArrayWithRandomValues(float[] array){
        for (int i = 0; i <array.length; i++) {
            array[i] = (float) ((Math.random()*2d-1d)/10d);
        }
    }

    @Override
    public void evaluate(ActivationFunction activationFunction) {
        System.arraycopy(biases, 0, this.nodes, 0, biases.length);
        int prevLayerCount = this.previousLayer.nodeCount;
        float[] prevLayerNodes = this.previousLayer.nodes;
        for (int p = 0;p < prevLayerCount; p++){
            for (int n = 0;n < this.nodeCount; n++){
                float effect = prevLayerNodes[p];
                effect*=this.weights[p+n*prevLayerCount];
                this.nodes[n] += effect;
            }
        }
        if(training){
            this.combinedInputs = this.nodes.clone();
        }
        for (int i = 0; i < this.nodeCount; i++) {
            this.nodes[i] = activationFunction.activation(this.nodes[i]);
        }
    }

    public void enableTraining() {
        this.training = true;
    }

    public float[] getWeightDerivatives(float[] nodeDerivatives) {

        float[] weightDerivatives = new float[this.weights.length];

        int prevLayerCount = this.previousLayer.nodeCount;
        float[] prevLayerNodes = this.previousLayer.nodes;

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
            //activationDerivatives[p] /= this.nodeCount;
        }

        return activationDerivatives;
    }

    private int weightAccumulations = 0;
    private float[] weightAccumulator;
    public void accumulateWeightChanges(float[] weightChanges){
        this.weightAccumulations++;
        if(this.weightAccumulator == null){
            this.weightAccumulator = new float[this.weights.length];
        }
        for (int i = 0; i < this.weightAccumulator.length; i++) {
            this.weightAccumulator[i] += weightChanges[i];
        }
    }
    private int biasAccumulations = 0;
    private float[] biasAccumulator;
    public void accumulateBiasChanges(float[] biasChanges){
        this.biasAccumulations++;
        if(this.biasAccumulator== null){
            this.biasAccumulator = new float[this.biases.length];
        }
        for (int i = 0; i < this.biasAccumulator.length; i++) {
            this.biasAccumulator[i] += biasChanges[i];
        }
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
        for (int i = 0; i < this.weightAccumulator.length; i++) {
            float changes = this.weightAccumulator[i]/((float) this.weightAccumulations);
            this.weights[i] -= changes;
        }
        this.weightAccumulations = 0;
        this.weightAccumulator = null;
    }

    public void pushBiasesAccumulator(){
        for (int i = 0; i < this.biasAccumulator.length; i++) {
            float changes = this.biasAccumulator[i]/((float) this.biasAccumulations);
            this.biases[i] -= changes;
        }
        this.biasAccumulations = 0;
        this.biasAccumulator = null;
    }
}
