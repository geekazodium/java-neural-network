package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.geekazodium.handdrawndigitstuff.neuralnetwork.*;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class AdditionResidualBlock {//extends AbstractLayer implements NonFinalLayer, EvaluateLayer{
//    private final AbstractLayer[] internalLayers;
//    private ActivationFunction activationFunction;
//    private AbstractLayer previousLayer;
//    private AbstractLayer nextLayer;
//
//    public AdditionResidualBlock(int outNodes,AbstractLayer[] internalLayers) {
//        super(outNodes);
//        this.internalLayers = internalLayers;
//        for (int i = 0; i < this.internalLayers.length; i++) {
//            if(i+1 > this.internalLayers.length){
//                ((NonFinalLayer) this.internalLayers[i]).setNextLayer(this);
//            }else {
//                ((NonFinalLayer) this.internalLayers[i]).setNextLayer(this.internalLayers[i+1]);
//            }
//            if(i-1 < 0){
//                ((NonInputLayer) this.internalLayers[i]).setPreviousLayer(this);
//            }else {
//                ((NonInputLayer) this.internalLayers[i]).setPreviousLayer(this.internalLayers[i-1]);
//            }
//        }
//    }
//
////    /**
////     * @deprecated
////     * @param activationFunction
////     */
////    @Override
////    @Deprecated
////    public void evaluate(ActivationFunction activationFunction) {
////        System.arraycopy(this.previousLayer.nodes,0,this.nodes,0,this.nodes.length);
////        for (AbstractLayer abstractLayer : this.internalLayers) {
////            abstractLayer.evaluate(activationFunction);
////        }
////        AbstractLayer lastLayer = this.internalLayers[this.internalLayers.length-1];
////        for (int i = 0; i < lastLayer.nodes.length; i++) {
////            this.nodes[i]+=lastLayer.nodes[i];
////        }
////    }
//
//    @Override
//    public void setActivationFunction(ActivationFunction activationFunction) {
//        for (AbstractLayer internalLayer : this.internalLayers) {
//            if(internalLayer instanceof EvaluateLayer evaluateLayer){
//                evaluateLayer.setActivationFunction(activationFunction);
//            }
//        }
//    }
//
//    @Override
//    public float[][] trainingEvaluate(ActivationFunction activationFunction, float[] previousLayerNodes) {
////        float[] nodes = new float[this.nodeCount];
////        System.arraycopy(biases, 0, nodes, 0, biases.length);
////        int prevLayerCount = this.previousLayer.nodeCount;
////        for (int p = 0;p < prevLayerCount; p++){
////            for (int n = 0;n < this.nodeCount; n++){
////                float effect = previousLayerNodes[p];
////                effect*=this.weights[p+n*prevLayerCount];
////                nodes[n] += effect;
////            }
////        }
////        float[] preActivation = nodes.clone();
////        for (int i = 0; i < this.nodeCount; i++) {
////            nodes[i] = activationFunction.activation(nodes[i]);
////        }
////        return new float[][]{nodes, preActivation};
//        throw new RuntimeException("training for residual blocks is incomplete");
//    }
//
//    @Override
//    public float[] evaluate(float[] in) {
//        float[] input = in.clone();
//        for (AbstractLayer abstractLayer : this.internalLayers) {
//            abstractLayer.evaluate(input);
//        }
//        AbstractLayer lastLayer = this.internalLayers[this.internalLayers.length-1];
//        for (int i = 0; i < lastLayer.nodes.length; i++) {
//            input[i]+=lastLayer.nodes[i];
//        }
//        return input;
//    }
//
//    //TODO MAKE BACKPROPAGATION SELF-CONTAINED TO ALLOW FOR DIFFERENT BACKPROPAGATION FUNCTIONS
//
//    @Override
//    public String name() {
//        return "AddResidualBlock";
//    }
//
//    @Override
//    public void setNextLayer(AbstractLayer layer) {
//        this.nextLayer = layer;
//    }
//
//    @Override
//    public AbstractLayer getNextLayer(){
//        return this.nextLayer;
//    }
//
//    @Override
//    public void setPreviousLayer(AbstractLayer layer) {
//        if(previousLayer.nodeCount!=this.nodeCount)throw new RuntimeException("layer before residual block must have the same amount of neurons as residual block");
//        this.previousLayer = layer;
//    }

}
