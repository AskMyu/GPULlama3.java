package com.example.model.llama;

import com.example.inference.InferenceCore;
import com.example.inference.InferenceEngine;
import com.example.inference.sampler.Sampler;
import com.example.inference.state.LlamaState;
import com.example.inference.state.State;
import com.example.inference.weights.Weights;
import com.example.model.AbstractModel;
import com.example.model.ModelType;
import com.example.model.format.ChatFormat;
import com.example.tokenizer.impl.LlamaTokenizer;
import com.example.tokenizer.impl.Tokenizer;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class Llama extends AbstractModel {

    LlamaConfiguration configuration;

    public Llama(LlamaConfiguration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    @Override
    public LlamaConfiguration configuration() {
        return configuration;
    }

    @Override
    public LlamaTokenizer tokenizer() {
        return (LlamaTokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.LLAMA_3;
    }

    @Override
    public State createNewState() {
        State state = new LlamaState(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new LlamaState(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    @Override
    public void forward(State state, int token, int position) {
        InferenceCore.forwardJava(this, state, token, position);
    }


    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo, IntConsumer onTokenGenerated) {
        return InferenceEngine.generateTokensLlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }
}

