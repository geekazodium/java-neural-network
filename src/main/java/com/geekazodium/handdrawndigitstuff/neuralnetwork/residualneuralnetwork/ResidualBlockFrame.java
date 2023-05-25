package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.geekazodium.handdrawndigitstuff.neuralnetwork.*;

public class ResidualBlockFrame extends AbstractLayer implements NonFinalLayer, EvaluateLayer{
    private final AbstractLayer[] internalLayers;
    private ActivationFunction activationFunction;
    private final ResidualMergeOperation residualMergeOperation;
    private AbstractLayer previousLayer;
    private AbstractLayer internalNextLayer;

    public ResidualBlockFrame(int inNodes, AbstractLayer[] internalLayers, ResidualMergeOperation residualMergeOperation) {
        super(inNodes);
        this.residualMergeOperation = residualMergeOperation;
        this.internalLayers = internalLayers;
        for (int i = 0; i < this.internalLayers.length; i++) {
            AbstractLayer internalLayer = this.internalLayers[i];
            if(i+1 > this.internalLayers.length-1){
                ((NonFinalLayer) internalLayer).setNextLayer(residualMergeOperation);
                this.residualMergeOperation.internalPreviousLayer = internalLayer;
            }else {
                ((NonFinalLayer) internalLayer).setNextLayer((EvaluateLayer) this.internalLayers[i+1]);
            }
            if(i-1 < 0){
                ((NonInputLayer) internalLayer).setPreviousLayer(this);
                this.setNextLayer((EvaluateLayer) internalLayer);
            }else {
                ((NonInputLayer) internalLayer).setPreviousLayer(this.internalLayers[i-1]);
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
    public void setNextLayer(EvaluateLayer layer) {
        this.residualMergeOperation.nextLayer = layer;
    }

    @Override
    public EvaluateLayer getNextLayer(){
        return this.residualMergeOperation.nextLayer;
    }

    @Override
    public void setPreviousLayer(AbstractLayer layer) {
        if(layer.nodeCount!=this.nodeCount)throw new RuntimeException("layer before residual block must have the same amount of neurons as residual block");
        this.previousLayer = layer;
    }

    @Override
    public AbstractLayer getPreviousLayer() {
        return this.previousLayer;
    }

    @Override
    public float[] evaluate(float[] in, Object[] args) {
        return this.internalLayers[0].evaluate(in, new Object[]{in,args});
    }

    /**
     * @param in
     * @param args
     * @deprecated can not evaluate self on connector layer
     */
    @Override
    @Deprecated
    public float[] evaluateSelf(float[] in, Object[] args) {
        throw new RuntimeException("can not evaluate self on a connector");
    }

    @Override
    public float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject, Object[] args) {
        return this.internalLayers[0].backpropagate(in,costFunction,trainingDataObject,new Object[]{in,args});
    }

    @Override
    public void init() {
        for (AbstractLayer internalLayer : this.internalLayers) {
            ((EvaluateLayer) internalLayer).init();
        }
    }

    public static abstract class ResidualMergeOperation extends AbstractLayer implements EvaluateLayer,NonFinalLayer{
        protected AbstractLayer internalPreviousLayer;
        protected EvaluateLayer nextLayer;
        public ResidualMergeOperation(int nodes) {
            super(nodes);
        }

        @Override
        public void setNextLayer(EvaluateLayer layer) {
            this.nextLayer = layer;
        }

        @Override
        public EvaluateLayer getNextLayer(){
            return this.nextLayer;
        }

        @Override
        public void setPreviousLayer(AbstractLayer layer) {
            throw new RuntimeException("can not modify internal layer links");
        }

        @Override
        public AbstractLayer getPreviousLayer() {
            throw new RuntimeException("can not get internal layer links");
        }

        @Override
        public float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject, Object[] args) {
            float[] blockOuts = evaluateSelf(in, args);
            return this.trim(nextLayer.backpropagate(blockOuts,costFunction,trainingDataObject,(Object[]) args[1]));
        }

        @Override
        public float[] evaluate(float[] in, Object[] args) {
            float[] blockOuts = evaluateSelf(in, args);
            return nextLayer.evaluate(blockOuts, ((Object[]) args[1]));
        }

        @Override
        public void init() {

        }

        public float[] evaluateSelf(float[] in, Object[] args){
            return this.merge(in, (float[]) args[0]);
        }

        public abstract float[] merge(float[] lastLayer,float[] in);

        public abstract float[] trim(float[] activationChanges);
    }
}
