package com.geekazodium.handdrawndigitstuff.neuralnetwork;

public abstract class AbstractEvaluateLayer extends AbstractLayer{
    protected AbstractLayer previousLayer;

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
            array[i] = (float) (Math.random()*2d-1d);
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
        for (int i = 0; i < this.nodeCount; i++) {
            this.nodes[i] = activationFunction.activation(this.nodes[i]);
        }
    }
}
