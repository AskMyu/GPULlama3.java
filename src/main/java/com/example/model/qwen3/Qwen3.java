package com.example.model.qwen3;

import com.example.inference.InferenceCore;
import com.example.inference.InferenceEngine;
import com.example.inference.sampler.Sampler;
import com.example.inference.state.Qwen3State;
import com.example.inference.state.State;
import com.example.inference.weights.Weights;
import com.example.model.AbstractModel;
import com.example.model.ModelType;
import com.example.model.format.ChatFormat;
import com.example.tokenizer.impl.Qwen3Tokenizer;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class Qwen3 extends AbstractModel {

    Qwen3Configuration configuration;

    public Qwen3(Qwen3Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    public Qwen3Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.QWEN_3;
    }

    public Qwen3Tokenizer tokenizer() {
        return (Qwen3Tokenizer) tokenizer;
    }

    @Override
    public State createNewState() {
        State state = new Qwen3State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Qwen3State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    @Override
    public void forward(State state, int token, int position) {
        if (plan == null) {
            InferenceCore.forwardJavaQwen3(this, state, token, position);
        } else {
            InferenceCore.forwardTornadoVM(this, state, token, position, tornadoVMPlan());
        }
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo, IntConsumer onTokenGenerated) {
        return InferenceEngine.generateTokensQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        return InferenceEngine.generateTokensGPUQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }

}
