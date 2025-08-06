package com.example.model.qwen2;

import com.example.inference.InferenceCore;
import com.example.inference.InferenceEngine;
import com.example.inference.sampler.Sampler;
import com.example.inference.state.Qwen2State;
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

public class Qwen2 extends AbstractModel {

    Qwen2Configuration configuration;

    public Qwen2(Qwen2Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    public Qwen2Configuration configuration() {
        return configuration;
    }

    @Override
    public Tokenizer tokenizer() {
        return (Qwen3Tokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.QWEN_2;
    }

    @Override
    public State createNewState() {
        State state = new Qwen2State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Qwen2State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    /**
     * No <|beginoftext|> needed for Qwen models.
     */
    @Override
    public boolean shouldAddBeginOfText() {
        return false;
    }

    /**
     * No system prompt for Deepseek-R1-Distill-Qwen.
     * Based on <a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">Usage Recommendations</a>
     */
    @Override
    public boolean shouldAddSystemPrompt() {
        return !getModelType().isDeepSeekR1();
    }

    /**
     * Force inclusion of <think></think> for Deepseek-R1-Distill-Qwen.
     * Based on <a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">Usage Recommendations</a>
     */
    @Override
    public boolean shouldIncludeReasoning() {
        return getModelType().isDeepSeekR1();
    }

    @Override
    public void forward(State state, int token, int position) {
        if (plan == null) {
            InferenceCore.forwardJavaQwen2(this, state, token, position);
        } else {
            InferenceCore.forwardTornadoVM(this, state, token, position, tornadoVMPlan());
        }
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        return InferenceEngine.generateTokensQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        return InferenceEngine.generateTokensGPUQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
}
