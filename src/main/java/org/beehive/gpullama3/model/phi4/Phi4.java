package org.beehive.gpullama3.model.phi4;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.impl.Phi3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class Phi4 extends AbstractModel {

    Phi4Configuration configuration;

    public Phi4(Phi4Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    public Phi4Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        // Return specific model type based on configuration
        if (configuration.isReasoningModel()) {
            return ModelType.PHI_4_MINI_REASONING;
        }
        return ModelType.PHI_3; // Fallback for regular Phi-4
    }

    public Phi3Tokenizer tokenizer() {
        return (Phi3Tokenizer) tokenizer;
    }

    @Override
    public State createNewState() {
        // Reuse Phi3State as the architecture is similar
        State state = new Phi3State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        // Reuse Phi3State as the architecture is similar
        State state = new Phi3State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    /**
     * Phi-4 models need begin of text token
     */
    @Override
    public boolean shouldAddBeginOfText() {
        return true;
    }

    /**
     * Include reasoning tokens for Phi-4-Mini-Reasoning models.
     * This enables automatic `<think>\n` token insertion for reasoning traces.
     */
    @Override
    public boolean shouldIncludeReasoning() {
        return configuration.isReasoningModel();
    }

    @Override
    public void forward(State state, int token, int position) {
        if (plan == null) {
            // Use Phi3 inference core as the architectures are similar
            InferenceCore.forwardJavaPhi3(this, (Phi3State) state, token, position);
        } else {
            InferenceCore.forwardTornadoVM(this, state, token, position, tornadoVMPlan());
        }
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, 
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        // Use Phi3 generation engine as the architectures are similar
        return InferenceEngine.generateTokensPhi3(this, state, startPosition, promptTokens, 
                stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, 
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        // Use Phi3 GPU generation engine as the architectures are similar
        return InferenceEngine.generateTokensGPUPhi3(this, state, startPosition, promptTokens, 
                stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
}