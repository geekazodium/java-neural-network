package com.geekazodium.handdrawndigitstuff.neuralnetwork;

import com.geekazodium.handdrawndigitstuff.GPUComputeContext;
import com.geekazodium.handdrawndigitstuff.neuralnetwork.activationfunctions.ActivationFunction;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.lwjgl.opencl.CL30.*;

public abstract class AbstractEvaluateLayer extends AbstractLayer implements EvaluateLayer,SerializableToJsonLayer,EvaluateModifiableLayer {
    protected AbstractLayer previousLayer;
    public float[] weights;
    public float[] biases;
    private long weightGradientBuffer;
    private long nodeGradientBuffer;
    public float[] biasGradients;
    public float[] weightGradients;
    private long weightBuffer;
    private long biasBuffer;

    public AbstractEvaluateLayer(int nodes) {
        super(nodes);
    }

    public void setPreviousLayer(AbstractLayer previousLayer){
        this.previousLayer = previousLayer;
    }

    public void initWeights(){
        int nodeCount = this.previousLayer.nodeCount;
        this.weights = new float[nodeCount*this.nodeCount];
        fillArrayWithRandomValuesNormalDist(this.weights,0.05);
    }
    public void initBiases(){
        this.biases = new float[this.nodeCount];
        fillArrayWithRandomValues(this.biases,0);
    }

    public AbstractLayer getPreviousLayer(){
        return previousLayer;
    }

    private void fillArrayWithRandomValues(float[] array,double multiplier){
        for (int i = 0; i <array.length; i++) {
            array[i] = (float) ((Math.random()*2d-1d)*multiplier);
        }
    }

    private void fillArrayWithRandomValuesNormalDist(float[] array, double multiplier) {
        for (int i = 0; i <array.length; i++) {
            array[i] = (float) ((Math.sqrt(-2*Math.log(Math.random())) * Math.cos(2d*Math.PI*Math.random()))*multiplier);
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

    @Override
    public void pushChanges(float learnRate) {
        this.pushWeightAccumulator(learnRate);
        this.pushBiasesAccumulator(learnRate);
    }

    public float[] asyncGetWeightDerivatives(float[] nodeDerivatives, float[] prevLayerNodes) {

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

    @Override
    public LayerBuffers createBuffer(long gpuContext) {
        this.weightBuffer = clCreateBuffer(gpuContext,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this.weights,null);
        biasBuffer = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, this.biases, null);
        return new LayerBuffers(
                this.weightBuffer,
                biasBuffer,
                EVALUATE_LAYER_ID
        );
    }

    @Override
    public void createLayerBuffer(long[] layerDataBuffers, float[][] layerStackedData, GPUComputeContext gpuContext, int stackSize, int index) {
        super.createLayerBuffer(layerDataBuffers, layerStackedData, gpuContext, stackSize, index);
    }

    @Override
    public float[] getWeights() {
        return weights;
    }

    @Override
    public float[] getBiases() {
        return biases;
    }

    @Override
    public RunnableKernel getEvaluateKernel(GPUComputeContext context, int index) {
        String kernelSrc = """ 
                __kernel void evaluate(
                        __constant float *weights,
                        __constant float *biases,
                        __global float *previousLayer,
                        __global float *preActivation,
                        __global float *output,
                        __constant int *previousLayerSizePointer,
                        __constant int *layerSizePointer
                        ){
                    int neuron = get_global_id(0);
                    int stackLayer = get_global_id(1);
                    
                    int layerSize = layerSizePointer[0];
                    int previousLayerSize = previousLayerSizePointer[0];
                    
                    int stackOffset = layerSize * stackLayer;
                    int previousLayerStackOffset = previousLayerSize * stackLayer;
                    
                    int resultLocation = neuron + stackOffset;
                    
                    float result = biases[neuron];
                    
                    for(int p = 0;p<previousLayerSize;p++){
                        result += weights[p + neuron * previousLayerSize] * previousLayer[p + previousLayerStackOffset];
                    }
                    
                    preActivation[resultLocation] = result;
                    output[resultLocation] = """ + activationFunctionString("result") + """
                }
                """;
        long layerEvaluateKernel = context.getKernel(kernelSrc,"evaluate");
        if(prevLayerNodeCountBuffer == 0) {
            int[] _prevLayerNodeCount = new int[]{this.previousLayer.nodeCount};
            int[] _layerNodeCount = new int[]{this.nodeCount};

            prevLayerNodeCountBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, _prevLayerNodeCount, null);
            layerNodeCountBuffer = clCreateBuffer(context.getGPUContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, _layerNodeCount, null);
        }

        clSetKernelArg(layerEvaluateKernel,0,pointerOf(weightBuffer));
        clSetKernelArg(layerEvaluateKernel,1,pointerOf(biasBuffer));
        clSetKernelArg(layerEvaluateKernel,2,pointerOf(context.layerDataBuffers[index-1]));
        clSetKernelArg(layerEvaluateKernel,3,pointerOf(context.preActivationBuffers[index]));
        clSetKernelArg(layerEvaluateKernel,4,pointerOf(context.layerDataBuffers[index]));
        clSetKernelArg(layerEvaluateKernel,5,pointerOf(prevLayerNodeCountBuffer));
        clSetKernelArg(layerEvaluateKernel,6,pointerOf(layerNodeCountBuffer));
        return new RunnableKernel() {
            @Override
            public void run(GPUComputeContext context) {
                PointerBuffer globalWorkSize = BufferUtils.createPointerBuffer(2);
                globalWorkSize.put(nodeCount);
                globalWorkSize.put(context.getStackSize());
                globalWorkSize.rewind();

                clEnqueueNDRangeKernel(
                        context.getCommandQueue(), layerEvaluateKernel, 2,
                        null, globalWorkSize,null,null,null
                );
            }
        };
    }

    @Override
    public RunnableKernel createBackpropagationKernels(GPUComputeContext context, int index){
        String nodeGradientsSrc = """ 
                __kernel void getLayerNodeGradient(
                        __constant float *preActivations,
                        __constant float *activationGradients,
                        __global float *biasGradient,
                        __constant int *layerSizePointer
                        ){
                    int neuron = get_global_id(0);
                    int stackLayer = get_global_id(1);
                    
                    int layerSize = layerSizePointer[0];
                    
                    int stackOffset = layerSize * stackLayer;
                    
                    int listIndex = stackOffset + neuron;
                    float preActivation = preActivations[listIndex];
                    biasGradient[listIndex] = activationGradients[listIndex] * """+ activationGradientString("preActivation")+"""
                }
                """;

        String prevLayerGradientsSrc = """ 
                __kernel void getGradientsFromNodeGradient(
                        __constant float *nodeGradients, //equal to biasGradients
                        __constant float *previousLayerActivations,
                        __constant float *weightBuffers,
                        __global float *previousLayerActivationGradient,
                        __constant int *previousLayerSizePointer,
                        __constant int *layerSizePointer,
                        __constant int *stackSizePointer
                        ){
                    int previousLayerNeuron = get_global_id(0);
                    int stackLayer = get_global_id(1);
                    
                    int previousLayerSize = previousLayerSizePointer[0];
                    int previousLayerStackOffset = previousLayerSize * stackLayer;
                    
                    int layerSize = layerSizePointer[0];
                    int stackSize = stackSizePointer[0];
                    int stackOffset = layerSize * stackLayer;
                    
                    int previousLayerIndex = previousLayerNeuron + previousLayerStackOffset;
                    
                    previousLayerActivationGradient[previousLayerIndex] = 0;
                    
                    
                    int localGroup = get_local_id(1);
                    
                    for (int n = 0;n < layerSize; n++){// the previous node activation is the derivative of the weight to the value before activation f()
                        float nodeGradient = nodeGradients[n + stackOffset];
                        int weightIndex = previousLayerNeuron + n * previousLayerSize;
                        
                        previousLayerActivationGradient[previousLayerIndex] +=
                                nodeGradient *
                                weightBuffers[weightIndex];
                        
                    }
                }
                """;//todo add stack size stuff here

        String weightGradientSrc = """
                __kernel void getWeightGradients(
                        __constant float *nodeGradients, //equal to biasGradients
                        __constant float *previousLayerActivations,
                        __global float *weightGradient,
                        __constant int *previousLayerSizePointer,
                        __constant int *layerSizePointer,
                        __constant int *stackSizePointer
                        ){
                    int previousLayerNeuron = get_global_id(0);
                    int neuron = get_global_id(1);
                    
                    int previousLayerSize = previousLayerSizePointer[0];
                    int layerSize = layerSizePointer[0];
                    int stackSize = stackSizePointer[0];
                    
                    int weightIndex = previousLayerNeuron + neuron * previousLayerSize;
                    
                    float adjustment = 0.0f;
                    for(int stackLayer = 0; stackLayer < stackSize; stackLayer ++){
                        int stackOffset = layerSize * stackLayer;
                        int previousLayerStackOffset = previousLayerSize * stackLayer;
                        
                        float prevNodeActivation = previousLayerActivations[previousLayerNeuron + previousLayerStackOffset];
                        float nodeGradient = nodeGradients[neuron + stackOffset];
                        
                        adjustment += (nodeGradient * prevNodeActivation);
                    }
                    weightGradient[weightIndex] = adjustment;
                }
                """;

        String parameterAdjustSrc = """
                __kernel void parameterAdjust(
                        __constant float *nodeGradients, //equal to biasGradients
                        __constant float *weightGradients,
                        __global float *biasBuffers,
                        __global float *weightBuffers,
                        __constant int *previousLayerSizePointer,
                        __constant int *layerSizePointer,
                        __constant float *learnRatePointer,
                        __constant int *stackSizePointer
                        ){
                    int neuron = get_global_id(0);
                    
                    int previousLayerSize = previousLayerSizePointer[0];
                    int layerSize = layerSizePointer[0];
                    
                    float learnRate = learnRatePointer[0];
                    int stackSize = stackSizePointer[0];
                    
                    float stackSizeFloat = (float) stackSize;
                    
                    if(neuron < layerSize){
                        float biasAdjustment = 0;
                        for(int stackLayer = 0; stackLayer < stackSize; stackLayer ++){
                            int biasStackOffset = stackLayer * layerSize;
                            biasAdjustment += nodeGradients[biasStackOffset + neuron] / stackSizeFloat;
                        }
                        biasBuffers[neuron] += biasAdjustment * -learnRate;
                    }
                    
                    float weightAdjustment = weightGradients[neuron];
                    weightBuffers[neuron] += weightAdjustment * -learnRate / stackSizeFloat;
                }
                """;


        long layerNodeGradientKernel = context.getKernel(nodeGradientsSrc, "getLayerNodeGradient");
        long prevLayerActivationGradientKernel = context.getKernel(prevLayerGradientsSrc, "getGradientsFromNodeGradient");

        long gpuContext = context.getGPUContext();
        biasGradients = new float[this.nodeCount * context.getStackSize()];
        weightGradients = new float[this.nodeCount * this.previousLayer.nodeCount];
        nodeGradientBuffer = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, biasGradients, null);
        weightGradientBuffer = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightGradients, null);

        List<Integer> resultList = new ArrayList<>();
        resultList.add(clSetKernelArg(layerNodeGradientKernel,0,pointerOf(context.preActivationBuffers[index])));
        resultList.add(clSetKernelArg(layerNodeGradientKernel,1,pointerOf(context.layerGradientBuffers[index])));
        resultList.add(clSetKernelArg(layerNodeGradientKernel,2,pointerOf(nodeGradientBuffer)));
        resultList.add(clSetKernelArg(layerNodeGradientKernel,3,pointerOf(layerNodeCountBuffer)));

        resultList.add(clSetKernelArg(prevLayerActivationGradientKernel,0,pointerOf(nodeGradientBuffer)));
        resultList.add(clSetKernelArg(prevLayerActivationGradientKernel,1,pointerOf(context.layerDataBuffers[index-1])));
        resultList.add(clSetKernelArg(prevLayerActivationGradientKernel,2,pointerOf(context.weightBuffers[index])));
        resultList.add(clSetKernelArg(prevLayerActivationGradientKernel,3,pointerOf(context.layerGradientBuffers[index-1])));
        resultList.add(clSetKernelArg(prevLayerActivationGradientKernel,4,pointerOf(this.prevLayerNodeCountBuffer)));
        resultList.add(clSetKernelArg(prevLayerActivationGradientKernel,5,pointerOf(this.layerNodeCountBuffer)));
        resultList.add(clSetKernelArg(prevLayerActivationGradientKernel,6,pointerOf(context.getStackSizePointer())));

        long weightGradientKernel = context.getKernel(weightGradientSrc, "getWeightGradients");
        resultList.add(clSetKernelArg(weightGradientKernel,0,pointerOf(nodeGradientBuffer)));
        resultList.add(clSetKernelArg(weightGradientKernel,1,pointerOf(context.layerDataBuffers[index-1])));
        resultList.add(clSetKernelArg(weightGradientKernel,2,pointerOf(weightGradientBuffer)));
        resultList.add(clSetKernelArg(weightGradientKernel,3,pointerOf(this.prevLayerNodeCountBuffer)));
        resultList.add(clSetKernelArg(weightGradientKernel,4,pointerOf(this.layerNodeCountBuffer)));
        resultList.add(clSetKernelArg(weightGradientKernel,5,pointerOf(context.getStackSizePointer())));

        long parameterAdjustKernel = context.getKernel(parameterAdjustSrc, "parameterAdjust");
        resultList.add(clSetKernelArg(parameterAdjustKernel,0,pointerOf(nodeGradientBuffer)));
        resultList.add(clSetKernelArg(parameterAdjustKernel,1,pointerOf(weightGradientBuffer)));
        resultList.add(clSetKernelArg(parameterAdjustKernel,2,pointerOf(context.biasBuffers[index])));
        resultList.add(clSetKernelArg(parameterAdjustKernel,3,pointerOf(context.weightBuffers[index])));
        resultList.add(clSetKernelArg(parameterAdjustKernel,4,pointerOf(this.prevLayerNodeCountBuffer)));
        resultList.add(clSetKernelArg(parameterAdjustKernel,5,pointerOf(this.layerNodeCountBuffer)));
        resultList.add(clSetKernelArg(parameterAdjustKernel,6,pointerOf(context.getLearnRatePointer())));
        resultList.add(clSetKernelArg(parameterAdjustKernel,7,pointerOf(context.getStackSizePointer())));

        return new EvaluateBackpropagateKernel(layerNodeGradientKernel,prevLayerActivationGradientKernel,weightGradientKernel,this.nodeCount,this.previousLayer.nodeCount,parameterAdjustKernel,this);
    }

    private static class EvaluateBackpropagateKernel implements RunnableKernel{

        private final long nodeGradientKernel;
        private final long previousLayerActivationGradientKernel;

        private final long weightGradientKernel;
        private final int prevLayerSize;
        private final int layerSize;
        private final long parameterAdjustKernel;
        private final AbstractEvaluateLayer evaluateLayer;

        public EvaluateBackpropagateKernel(long nodeGradientKernel, long previousLayerActivationGradientKernel, long weightGradientKernel, int layerSize, int prevLayerSize, long parameterAdjustKernel, AbstractEvaluateLayer evaluateLayer){
            this.nodeGradientKernel = nodeGradientKernel;
            this.previousLayerActivationGradientKernel = previousLayerActivationGradientKernel;
            this.weightGradientKernel = weightGradientKernel;
            this.parameterAdjustKernel = parameterAdjustKernel;
            this.layerSize = layerSize;
            this.prevLayerSize = prevLayerSize;
            this.evaluateLayer = evaluateLayer;
        }

        @Override
        public void run(GPUComputeContext context) {

            long commandQueue = context.getCommandQueue();

            PointerBuffer biasKernelWorkSize = BufferUtils.createPointerBuffer(2);
            biasKernelWorkSize.clear();
            biasKernelWorkSize.put(this.layerSize);
            biasKernelWorkSize.put(context.getStackSize());
            biasKernelWorkSize.rewind();
            clEnqueueNDRangeKernel(commandQueue, this.nodeGradientKernel, 2, null, biasKernelWorkSize, null, null, null);

            PointerBuffer prevLayerGradientKernelSize = BufferUtils.createPointerBuffer(2);
            prevLayerGradientKernelSize.put(this.prevLayerSize);
            prevLayerGradientKernelSize.put(context.getStackSize());
            prevLayerGradientKernelSize.rewind();
            clEnqueueNDRangeKernel(commandQueue, this.previousLayerActivationGradientKernel, 2, null, prevLayerGradientKernelSize, null, null, null);

            PointerBuffer weightGradientKernelSize = BufferUtils.createPointerBuffer(2);
            weightGradientKernelSize.put(this.prevLayerSize);
            weightGradientKernelSize.put(this.layerSize);
            weightGradientKernelSize.rewind();
            clEnqueueNDRangeKernel(commandQueue, this.weightGradientKernel, 2, null,weightGradientKernelSize,null,null,null);

            PointerBuffer adjustParameterWorkSize = BufferUtils.createPointerBuffer(1);
            adjustParameterWorkSize.put((long) this.prevLayerSize * (long) this.layerSize);
            adjustParameterWorkSize.rewind();
            clEnqueueNDRangeKernel(commandQueue, this.parameterAdjustKernel, 1, null, adjustParameterWorkSize, null, null, null);

//            clEnqueueReadBuffer(commandQueue,evaluateLayer.weightGradientBuffer,true,0,evaluateLayer.weightGradients,null,null);
 //           clEnqueueReadBuffer(commandQueue,evaluateLayer.nodeGradientBuffer,true,0,evaluateLayer.biasGradients,null,null);

//            for (int j = 0; j < evaluateLayer.weightGradients.length; j++) {
//                float weightGradient = evaluateLayer.weightGradients[j];
//                if (weightGradient > 1) {
//                    System.out.println(j);
//                }
//            }
        }
    }

    long prevLayerNodeCountBuffer = 0;
    long layerNodeCountBuffer = 0;

    private String activationFunctionString(String result){
        return this.activationFunction.getKernelString(result);
    }

    private String activationGradientString(String gradient){
        return this.activationFunction.getGradientKernelString(gradient);
    }
}
