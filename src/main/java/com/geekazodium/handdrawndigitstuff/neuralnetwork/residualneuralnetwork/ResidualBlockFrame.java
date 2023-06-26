package com.geekazodium.handdrawndigitstuff.neuralnetwork.residualneuralnetwork;

import com.geekazodium.handdrawndigitstuff.GPUComputeContext;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.*;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL10.CL_MEM_COPY_HOST_PTR;

public class ResidualBlockFrame extends AbstractLayer implements NonFinalLayer, EvaluateLayer,SerializableToJsonLayer,EvaluateModifiableLayer{

    private static final int RESIDUAL_BLOCK_ID = 997;
    public static final int RESIDUAL_ADD_ID = 286;
    public static final int RESIDUAL_CONCAT_ID = 133;
    private AbstractLayer[] internalLayers;
    private ActivationFunction activationFunction;
    private ResidualMergeOperation residualMergeOperation;
    private AbstractLayer previousLayer;
    private AbstractLayer internalNextLayer;
    private long residualMergeGradientBuffer;

    public ResidualBlockFrame(int inNodes, AbstractLayer[] internalLayers, ResidualMergeOperation residualMergeOperation) {
        super(inNodes);
        this.residualMergeOperation = residualMergeOperation;
        this.internalLayers = internalLayers;
        this.residualMergeOperation.setResidualBlockFrame(this);
    }

    public ResidualBlockFrame(int inNodes) {
        super(inNodes);
    }

    public void setInternalLayers(AbstractLayer[] internalLayers){
        this.internalLayers = internalLayers;
    }

    public void setResidualMergeOperation(ResidualMergeOperation mergeOperation){
        this.residualMergeOperation = mergeOperation;
        this.residualMergeOperation.setResidualBlockFrame(this);
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
        return "ResidualBlock";
    }

    @Override
    public void setNextLayer(EvaluateLayer layer) {
        this.residualMergeOperation.nextLayer = layer;
    }

    @Override
    public EvaluateLayer getNextLayer(){
        return (EvaluateLayer) this.internalNextLayer;
    }

    @Override
    public void setPreviousLayer(AbstractLayer layer) {
        //if(layer.nodeCount!=this.nodeCount)throw new RuntimeException("layer before residual block must have the same amount of neurons as residual block");
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
    public AbstractLayer getEnd() {
        return this.residualMergeOperation.getEnd();
    }

    @Override
    public float[] backpropagate(float[] in, CostFunction costFunction, Object trainingDataObject, Object[] args) {
        GradientRemaining gradientRemaining = new GradientRemaining();
        float[] gradient = this.internalLayers[0].backpropagate(in, costFunction, trainingDataObject, new Object[]{in, gradientRemaining, args});
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] += gradientRemaining.remaining[i];
        }
        return gradient;
    }

    @Override
    public void pushChanges(float learnRate) {
        for (AbstractLayer internalLayer : this.internalLayers) {
            if(!(internalLayer instanceof EvaluateModifiableLayer modifiableLayer))continue;
            modifiableLayer.pushChanges(learnRate);
        }
    }

    public long getMergeGradientBuffer() {
        return this.residualMergeGradientBuffer;
    }

    private static class GradientRemaining {
        float[] remaining;
    }

    @Override
    public void init() {
        initInternalLayers();
        for (AbstractLayer internalLayer : this.internalLayers) {
            ((EvaluateLayer) internalLayer).init();
        }
    }

    public void initInternalLayers() {
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
                this.internalNextLayer = internalLayer;
            }else {
                ((NonInputLayer) internalLayer).setPreviousLayer(this.internalLayers[i-1]);
            }
        }
    }

    @Override
    public void createLayerBuffer(long[] layerDataBuffers, float[][] layerStackedData, GPUComputeContext gpuContext, int stackSize, int index) {
        layerStackedData[index] = layerStackedData[index-1];
        layerDataBuffers[index] = layerDataBuffers[index-1];
        this.residualMergeGradientBuffer = clCreateBuffer(gpuContext.getGPUContext(),CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,new float[stackSize*this.nodeCount],null);
        gpuContext.preActivationBuffers[index] = clCreateBuffer(gpuContext.getGPUContext(),CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,new float[stackSize*this.nodeCount],null);
        gpuContext.layerGradientBuffers[index] = clCreateBuffer(gpuContext.getGPUContext(),CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,new float[stackSize*this.nodeCount],null);
    }

    @Override
    public JsonObject serializeToJson() {
        JsonObject serialized = new JsonObject();
        serialized.addProperty("type",this.name());
        serialized.addProperty("nodes",this.nodeCount);
        serialized.addProperty("mergeType",this.residualMergeOperation.getClass().getName());
        this.residualMergeOperation.serialize(serialized);
        JsonArray internalLayers = new JsonArray();
        for (int i = 0; i < this.internalLayers.length; i++) {
            SerializableToJsonLayer internalLayer = (SerializableToJsonLayer) this.internalLayers[i];
            internalLayers.add(internalLayer.serializeToJson());
        }
        serialized.add("internalLayers",internalLayers);
        return serialized;
    }

    @Override
    public void deserializeFromJson(JsonObject object) {
        JsonArray array = object.get("internalLayers").getAsJsonArray();
        ArrayList<AbstractLayer> internalLayers = new ArrayList<>();
        array.forEach(jsonElement -> {
            internalLayers.add((AbstractLayer) NeuralNetwork.deserializeLayer(jsonElement));
        });
        AbstractLayer[] internalLayersArray = new AbstractLayer[internalLayers.size()];
        internalLayers.toArray(internalLayersArray);
        this.setInternalLayers(internalLayersArray);
        String mergeType = object.get("mergeType").getAsString();
        ResidualMergeOperation merge;
        try {
            merge = getMergeBlock(mergeType)
                    .getDeclaredConstructor(JsonObject.class)
                    .newInstance(object);
        } catch (NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
        this.setResidualMergeOperation(merge);
        this.initInternalLayers();
    }

    private static final Map<String,Class<? extends ResidualMergeOperation>> mergeOperations = new HashMap<>();
    static {
        putMergeBlock(ResidualConcatBlock.class);
        putMergeBlock(ResidualAddBlock.class);
    }

    private static void putMergeBlock(Class<? extends ResidualMergeOperation> mergeType) {
        mergeOperations.put(mergeType.getName(),mergeType);
    }

    public static Class<? extends ResidualMergeOperation> getMergeBlock(String mergeType){
        return mergeOperations.get(mergeType);
    }

    @Override
    public int layerDepth() {
        int depth = 2;
        for (AbstractLayer internalLayer : this.internalLayers) {
            depth += internalLayer.layerDepth();
        }
        return depth;
    }

    @Override
    public AbstractLayer[] getAsLayerArray() {
        int depth = this.layerDepth();
        AbstractLayer[] array = new AbstractLayer[depth];

        array[0] = this;
        array[depth-1] = this.residualMergeOperation;

        for (AbstractLayer internalLayer : this.internalLayers) {
            AbstractLayer[] internalArray = internalLayer.getAsLayerArray();

            for (int i = 0; i < internalArray.length; i++) {
                array[i+1+internalLayer.getIndex()] = internalArray[i];
            }
        }

        return array;
    }

    @Override
    public LayerBuffers createBuffer(long gpuContext) {
        return new LayerBuffers(0,0,RESIDUAL_BLOCK_ID);
    }

    @Override
    public void setIndex(int index) {
        super.setIndex(index);
        AbstractLayer[] layers = this.internalLayers;
        for (int i = 0; i < layers.length; i++) {
            AbstractLayer internalLayer = layers[i];
            internalLayer.setIndex(i);
        }
    }

    public static abstract class ResidualMergeOperation extends AbstractLayer implements EvaluateLayer,NonFinalLayer{
        protected AbstractLayer internalPreviousLayer;
        protected final int inputLength;
        protected ResidualBlockFrame residualBlockFrame;
        protected EvaluateLayer nextLayer;
        public ResidualMergeOperation(int nodes,int inputLength) {
            super(nodes);
            this.inputLength = inputLength;
        }

        public void setResidualBlockFrame(ResidualBlockFrame residualBlockFrame) {
            this.residualBlockFrame = residualBlockFrame;
        }

        public ResidualMergeOperation(JsonObject object){
            super(object.get("mergeNodes").getAsInt());
            this.inputLength = object.get("mergeInputs").getAsInt();
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
            float[][] split = this.trim(nextLayer.backpropagate(blockOuts, costFunction, trainingDataObject, (Object[]) args[2]));
            ((GradientRemaining) args[1]).remaining = split[1];
            return split[0];
        }

        @Override
        public float[] evaluate(float[] in, Object[] args) {
            float[] blockOuts = evaluateSelf(in, args);
            return nextLayer.evaluate(blockOuts, ((Object[]) args[1]));
        }

        @Override
        public void init() {

        }

        @Override
        public void setActivationFunction(ActivationFunction activationFunction) {

        }

        public float[] evaluateSelf(float[] in, Object[] args){
            return this.merge(in, (float[]) args[0]);
        }

        public abstract float[] merge(float[] lastLayer,float[] in);

        public abstract float[][] trim(float[] activationChanges);

        @Override
        public AbstractLayer getEnd() {
            return this;
        }

        public void serialize(JsonObject serialized) {
            serialized.addProperty("mergeNodes",this.nodeCount);
            serialized.addProperty("mergeInputs",this.inputLength);
        }

        public abstract int getType();

        @Override
        public LayerBuffers createBuffer(long gpuContext) {
            return new LayerBuffers(0,0,getType());
        }
    }

    @Override
    public RunnableKernel createBackpropagationKernels(GPUComputeContext context, int index) {
        String prevLayerGradientsSrc = """
                __kernel void getResidualGradients(
                        __global float *previousLayerActivationGradients,
                        __constant float *thisLayerGradients,
                        __constant float *mergeGradients,
                        __constant int *layerSizePointer
                        ){
                    int neuron = get_global_id(0);
                    int stackLayer = get_global_id(1);

                    int layerSize = layerSizePointer[0]; //for residual blocks it is always true that this layer size is equal to the previous layer size so we don't need to pass in the previous layer size

                    int stackOffset = layerSize * stackLayer;

                    int index = neuron + stackOffset;

                    float gradient = thisLayerGradients[index] + mergeGradients[index];

                    previousLayerActivationGradients[index] = gradient;
                }
                """;

        if(layerNodeCountBuffer == 0){
            int[] _layerNodeCount = new int[]{this.nodeCount};
            layerNodeCountBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, _layerNodeCount, null);
        }

        long residualGradientsKernel = context.getKernel(prevLayerGradientsSrc, "getResidualGradients");
        clSetKernelArg(residualGradientsKernel,0,pointerOf(context.layerGradientBuffers[index-1]));
        clSetKernelArg(residualGradientsKernel,1,pointerOf(context.layerGradientBuffers[index]));
        clSetKernelArg(residualGradientsKernel,2,pointerOf(this.getMergeGradientBuffer()));
        clSetKernelArg(residualGradientsKernel,3,pointerOf(this.layerNodeCountBuffer));
        return new RunnableKernel() {

            @Override
            public void run(GPUComputeContext context) {
                PointerBuffer workSize = BufferUtils.createPointerBuffer(2);
                workSize.put(nodeCount);
                workSize.put(context.getStackSize());
                workSize.rewind();

                clEnqueueNDRangeKernel(context.getCommandQueue(), residualGradientsKernel,2,null,workSize,null,null,null);
            }
        };
    }


    long layerNodeCountBuffer = 0;

}
