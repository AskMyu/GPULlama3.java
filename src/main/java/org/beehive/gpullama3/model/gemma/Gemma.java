package org.beehive.gpullama3.model.gemma;

import org.beehive.gpullama3.inference.GemmaInferenceCore;
import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.GemmaState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Gemma 3 model implementation.
 * 
 * Supports Google's Gemma 3 models including the compact 270M variant
 * with its distinctive large vocabulary (256K tokens).
 */
public class Gemma extends AbstractModel {

    private final GemmaConfiguration configuration;

    public Gemma(GemmaConfiguration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    @Override
    public GemmaConfiguration configuration() {
        return configuration;
    }

    @Override
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GEMMA_3;
    }

    @Override
    public State createNewState() {
        // Create a new state for Gemma model
        State state = new GemmaState(configuration(), -1);
        
        // Set the beginning of text token if available
        // Gemma uses <bos> token similar to other models
        if (tokenizer.getSpecialTokens().containsKey("<bos>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
        } else if (tokenizer.getSpecialTokens().containsKey("<|begin_of_text|>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        } else {
            // Default to token 1 if no special token found
            state.latestToken = 1;
        }
        
        return state;
    }
    
    @Override
    public State createNewState(int batchsize) {
        State state = new GemmaState(configuration(), batchsize);
        
        // Set the beginning of text token if available
        if (tokenizer.getSpecialTokens().containsKey("<bos>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
        } else if (tokenizer.getSpecialTokens().containsKey("<|begin_of_text|>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        } else {
            state.latestToken = 1;
        }
        
        return state;
    }
    
    @Override
    public void forward(State state, int token, int position) {
        // Use Gemma 3-specific inference with 5:1 local-global attention pattern
        // Implements QK-norm, 1,024-token sliding window, and enhanced RoPE scaling
        GemmaInferenceCore.forwardGemma3(this, state, token, position);
    }
    
    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, 
                                       Set<Integer> stopTokens, int maxTokens, Sampler sampler, 
                                       boolean echo, IntConsumer onTokenGenerated) {
        // Use Llama's generation engine for now
        // Gemma uses similar transformer architecture
        return InferenceEngine.generateTokensLlama(this, state, startPosition, promptTokens, 
                                                  stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }
    
    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens,
                                          Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                                          boolean echo, IntConsumer onTokenGenerated,
                                          TornadoVMMasterPlan tornadoVMPlan) {
        // Use optimized Gemma GPU generation with memory efficiency improvements
        // TODO: Implement GemmaInferenceEngine.generateTokensGPUGemma3() in Phase 3
        // For now, use enhanced LLaMA generation with Gemma state optimizations
        return InferenceEngine.generateTokensGPULlama(this, state, startPosition, promptTokens,
                                                     stopTokens, maxTokens, sampler, echo,
                                                     onTokenGenerated, tornadoVMPlan);
    }
}