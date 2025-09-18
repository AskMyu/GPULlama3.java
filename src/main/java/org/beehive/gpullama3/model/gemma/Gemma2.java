package org.beehive.gpullama3.model.gemma;

import org.beehive.gpullama3.inference.Gemma2InferenceCore;
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
 * Gemma 2-specific model implementation.
 * Uses Gemma 2 inference core with logit soft-capping and interleaved attention.
 * SEPARATE from Gemma (Gemma 3) to maintain clean architecture.
 */
public class Gemma2 extends AbstractModel {

    private final Gemma2Configuration config;

    public Gemma2(Gemma2Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.config = configuration;
    }

    @Override
    public Gemma2Configuration configuration() {
        return config;
    }

    @Override
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GEMMA_2;
    }

    @Override
    public State createNewState() {
        // Create a new state for Gemma 2 model
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
        // Use Gemma 2-specific inference with logit soft-capping and interleaved attention
        Gemma2InferenceCore.forwardGemma2(this, state, token, position);
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens,
                                       Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                                       boolean echo, IntConsumer onTokenGenerated) {
        // Use Llama's generation engine for now (compatible with Gemma architecture)
        return InferenceEngine.generateTokensLlama(this, state, startPosition, promptTokens,
                                                  stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens,
                                          Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                                          boolean echo, IntConsumer onTokenGenerated,
                                          TornadoVMMasterPlan tornadoVMPlan) {
        // Use optimized LLaMA generation with Gemma 2 state optimizations
        return InferenceEngine.generateTokensGPULlama(this, state, startPosition, promptTokens,
                                                     stopTokens, maxTokens, sampler, echo,
                                                     onTokenGenerated, tornadoVMPlan);
    }

}