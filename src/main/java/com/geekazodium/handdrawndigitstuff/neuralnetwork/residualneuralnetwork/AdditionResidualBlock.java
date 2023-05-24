package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.geekazodium.handdrawndigitstuff.neuralnetwork.*;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class AdditionResidualBlock extends AbstractLayer implements NonFinalLayer, EvaluateLayer{
    private final AbstractLayer[] internalLayers;
    private AbstractLayer previousLayer;
    private AbstractLayer nextLayer;

    public AdditionResidualBlock(int outNodes,AbstractLayer[] internalLayers) {
        super(outNodes);
        this.internalLayers = internalLayers;
        for (int i = 0; i < this.internalLayers.length; i++) {
            if(i+1 > this.internalLayers.length){
                ((NonFinalLayer) this.internalLayers[i]).setNextLayer(this);
            }else {
                ((NonFinalLayer) this.internalLayers[i]).setNextLayer(this.internalLayers[i+1]);
            }
            if(i-1 < 0){
                ((NonInputLayer) this.internalLayers[i]).setPreviousLayer(this);
            }else {
                ((NonInputLayer) this.internalLayers[i]).setPreviousLayer(this.internalLayers[i-1]);
            }
        }
    }

    @Override
    public void evaluate(ActivationFunction activationFunction) {
        System.arraycopy(this.previousLayer.nodes,0,this.nodes,0,this.nodes.length);
        for (AbstractLayer abstractLayer : this.internalLayers) {
            abstractLayer.evaluate(activationFunction);
        }
        AbstractLayer lastLayer = this.internalLayers[this.internalLayers.length-1];
        for (int i = 0; i < lastLayer.nodes.length; i++) {
            this.nodes[i]+=lastLayer.nodes[i];
        }
    }

    @Override
    public float[][] trainingEvaluate(ActivationFunction activationFunction, float[] previousLayerNodes) {
//        float[] nodes = new float[this.nodeCount];
//        System.arraycopy(biases, 0, nodes, 0, biases.length);
//        int prevLayerCount = this.previousLayer.nodeCount;
//        for (int p = 0;p < prevLayerCount; p++){
//            for (int n = 0;n < this.nodeCount; n++){
//                float effect = previousLayerNodes[p];
//                effect*=this.weights[p+n*prevLayerCount];
//                nodes[n] += effect;
//            }
//        }
//        float[] preActivation = nodes.clone();
//        for (int i = 0; i < this.nodeCount; i++) {
//            nodes[i] = activationFunction.activation(nodes[i]);
//        }
//        return new float[][]{nodes, preActivation};
        throw new RuntimeException("training for residual blocks is incomplete");
    }

    //TODO MAKE BACKPROPAGATION SELF-CONTAINED TO ALLOW FOR DIFFERENT BACKPROPAGATION FUNCTIONS

    @Override
    public String name() {
        return "AddResidualBlock";
    }

    @Override
    public void setNextLayer(AbstractLayer layer) {
        this.nextLayer = layer;
    }

    @Override
    public AbstractLayer getNextLayer(){
        return this.nextLayer;
    }

    @Override
    public void setPreviousLayer(AbstractLayer layer) {
        if(previousLayer.nodeCount!=this.nodeCount)throw new RuntimeException("layer before residual block must have the same amount of neurons as residual block");
        this.previousLayer = layer;
    }

    public void accumulateWeightChanges(float[] weightChanges,int layer){
        ((AbstractEvaluateLayer) this.internalLayers[layer]).accumulateWeightChanges(weightChanges);
    }

    public void accumulateBiasChanges(float[] biasChanges,int layer){
        ((AbstractEvaluateLayer) this.internalLayers[layer]).accumulateBiasChanges(biasChanges);
    }
}
