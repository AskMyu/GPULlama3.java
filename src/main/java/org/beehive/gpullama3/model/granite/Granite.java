package org.beehive.gpullama3.model.granite;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.GraniteState;
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
 * IBM Granite 3.3 model implementation.
 * 
 * Supports IBM's Granite 3.3 models with features like:
 * - Dense transformer architecture with GQA
 * - SwiGLU activation function
 * - Fill-in-the-Middle (FIM) support for code completion
 * - Structured reasoning with <think></think> tags
 * - Large context window (128K tokens)
 */
public class Granite extends AbstractModel {

    private final GraniteConfiguration configuration;

    public Granite(GraniteConfiguration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    @Override
    public GraniteConfiguration configuration() {
        return configuration;
    }

    @Override
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GRANITE_3_3;
    }

    @Override
    public boolean shouldIncludeReasoning() {
        // Granite 3.3 supports structured reasoning
        return true;
    }

    @Override
    public State createNewState() {
        State state = new GraniteState(configuration(), -1);

        // GRANITE FIX: Set the beginning of text token, avoiding token 0 which may be treated as EOS
        if (tokenizer.getSpecialTokens().containsKey("<|begin_of_text|>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
            System.err.printf("[GRANITE-STATE] Using BOS token: %d%n", state.latestToken);
        } else if (tokenizer.getSpecialTokens().containsKey("<bos>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
            System.err.printf("[GRANITE-STATE] Using fallback BOS token: %d%n", state.latestToken);
        } else {
            // CRITICAL: Default to token 1, never 0 (token 0 may be incorrectly mapped as EOS)
            state.latestToken = 1;
            System.err.printf("[GRANITE-STATE] Using default token 1 (avoiding token 0)%n");
        }

        // Additional safety check
        if (state.latestToken == 0) {
            System.err.println("[GRANITE-STATE] WARNING: Token 0 detected, forcing to token 1");
            state.latestToken = 1;
        }

        return state;
    }
    
    @Override
    public State createNewState(int batchsize) {
        State state = new GraniteState(configuration(), batchsize);

        // GRANITE FIX: Set the beginning of text token, avoiding token 0 which may be treated as EOS
        if (tokenizer.getSpecialTokens().containsKey("<|begin_of_text|>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
            System.err.printf("[GRANITE-STATE-BATCH] Using BOS token: %d%n", state.latestToken);
        } else if (tokenizer.getSpecialTokens().containsKey("<bos>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
            System.err.printf("[GRANITE-STATE-BATCH] Using fallback BOS token: %d%n", state.latestToken);
        } else {
            // CRITICAL: Default to token 1, never 0 (token 0 may be incorrectly mapped as EOS)
            state.latestToken = 1;
            System.err.printf("[GRANITE-STATE-BATCH] Using default token 1 (avoiding token 0)%n");
        }

        // Additional safety check
        if (state.latestToken == 0) {
            System.err.println("[GRANITE-STATE-BATCH] WARNING: Token 0 detected, forcing to token 1");
            state.latestToken = 1;
        }

        return state;
    }
    
    @Override
    public void forward(State state, int token, int position) {
        // Use the same inference core as Llama for now
        // Future optimization: implement Granite-specific forward pass with SwiGLU
        InferenceCore.forwardJava(this, state, token, position);
    }
    
    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, 
                                       Set<Integer> stopTokens, int maxTokens, Sampler sampler, 
                                       boolean echo, IntConsumer onTokenGenerated) {
        // Use Llama's generation engine for now
        // Granite uses similar transformer architecture
        return InferenceEngine.generateTokensLlama(this, state, startPosition, promptTokens, 
                                                  stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }
    
    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens,
                                          Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                                          boolean echo, IntConsumer onTokenGenerated,
                                          TornadoVMMasterPlan tornadoVMPlan) {
        // Use Llama's GPU generation for now
        return InferenceEngine.generateTokensGPULlama(this, state, startPosition, promptTokens,
                                                     stopTokens, maxTokens, sampler, echo,
                                                     onTokenGenerated, tornadoVMPlan);
    }
}