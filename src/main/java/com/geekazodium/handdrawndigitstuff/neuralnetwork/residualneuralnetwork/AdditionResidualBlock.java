package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.geekazodium.handdrawndigitstuff.neuralnetwork.*;

public class AdditionResidualBlock extends AbstractLayer implements NonFinalLayer, NonInputLayer {
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
    public void setNextLayer(AbstractLayer layer) {
        this.nextLayer = layer;
    }

    @Override
    public void setPreviousLayer(AbstractLayer layer) {
        if(previousLayer.nodeCount!=this.nodeCount)throw new RuntimeException("layer before residual block must have the same amount of neurons as residual block");
        this.previousLayer = layer;
    }
}
