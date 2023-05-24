package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.geekazodium.handdrawndigitstuff.neuralnetwork.*;

public class ResidualBlockFrame extends AbstractLayer implements NonFinalLayer, EvaluateLayer{
    private final AbstractLayer[] internalLayers;
    private ActivationFunction activationFunction;
    private final ResidualMergeOperation residualMergeOperation;
    private AbstractLayer previousLayer;
    private AbstractLayer nextLayer;

    public ResidualBlockFrame(int outNodes, AbstractLayer[] internalLayers, ResidualMergeOperation residualMergeOperation) {
        super(outNodes);
        this.residualMergeOperation = residualMergeOperation;
        this.internalLayers = internalLayers;
        for (int i = 0; i < this.internalLayers.length; i++) {
            if(i+1 > this.internalLayers.length){
                ((NonFinalLayer) this.internalLayers[i]).setNextLayer(residualMergeOperation);
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
    public void setActivationFunction(ActivationFunction activationFunction) {
     for (AbstractLayer internalLayer : this.internalLayers) {
         if(internalLayer instanceof EvaluateLayer evaluateLayer){
             evaluateLayer.setActivationFunction(activationFunction);
         }
     }
     this.activationFunction = activationFunction;
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

    @Override
    public AbstractLayer getPreviousLayer() {
        return this.previousLayer;
    }

    @Override
    public float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject) {
        return this.internalLayers[0].backpropagate(in,costFunction,trainingDataObject);
    }

    public static abstract class ResidualMergeOperation extends AbstractLayer implements EvaluateLayer,NonFinalLayer{
        protected AbstractLayer previousLayer;
        protected AbstractLayer nextLayer;
        public ResidualMergeOperation(int nodes) {
            super(nodes);
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
            this.previousLayer = layer;
        }

        @Override
        public AbstractLayer getPreviousLayer() {
            return this.previousLayer;
        }
    }
}
