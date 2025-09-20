package org.beehive.gpullama3.model.olmoe;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.state.OlmoeState;
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
 * OLMoE-1B-7B model implementation.
 * 
 * This is a Mixture-of-Experts model with 64 experts, activating 8 per token.
 * Total parameters: 7B, Active parameters: 1B per token.
 */
public class Olmoe extends AbstractModel {

    private final OlmoeConfiguration configuration;

    public Olmoe(OlmoeConfiguration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
        System.err.println("[OLMOE-CONSTRUCTOR-DEBUG] OLMoE model instantiated successfully!");
        System.err.printf("[OLMOE-CONSTRUCTOR-DEBUG] Configuration: %d experts, %d active, dim=%d%n",
                         configuration.numberOfExperts(),
                         configuration.numberOfActiveExperts(),
                         configuration.dim());
    }

    public OlmoeConfiguration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.OLMOE_1B_7B;
    }
    
    @Override
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public State createNewState() {
        State state = new OlmoeState(configuration(), -1);
        // Initialize with appropriate start token if available
        if (chatFormat != null && chatFormat.chatTokens() != null && 
            chatFormat.chatTokens().tStartHeader() != null) {
            Integer startToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
            if (startToken != null) {
                state.latestToken = startToken;
            }
        }
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new OlmoeState(configuration(), batchsize);
        // Initialize with appropriate start token if available
        if (chatFormat != null && chatFormat.chatTokens() != null && 
            chatFormat.chatTokens().tStartHeader() != null) {
            Integer startToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
            if (startToken != null) {
                state.latestToken = startToken;
            }
        }
        return state;
    }

    /**
     * OLMoE models need begin of text token
     */
    @Override
    public boolean shouldAddBeginOfText() {
        return true;
    }

    @Override
    public void forward(State state, int token, int position) {
        System.err.printf("[OLMOE-FORWARD-DEBUG] plan=%s, tornadoVMPlan()=%s%n",
                         plan, (plan != null) ? "available" : "null");

        if (plan == null) {
            // OLMoE-specific forward pass with MoE routing (CPU)
            System.err.println("[OLMOE-FORWARD-DEBUG] Using CPU forward pass - plan is null");
            InferenceCore.forwardJavaOlmoe(this, (OlmoeState) state, token, position);
        } else {
            // OLMoE-specific GPU forward pass with MoE routing
            System.err.println("[OLMOE-FORWARD-DEBUG] Using GPU forward pass - calling forwardTornadoVMOlmoe");
            InferenceCore.forwardTornadoVMOlmoe(this, (OlmoeState) state, token, position, tornadoVMPlan());
        }
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, 
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        // OLMoE-specific token generation with MoE
        return InferenceEngine.generateTokensOlmoe(this, (OlmoeState) state, startPosition, 
                promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, 
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        // OLMoE-specific GPU token generation
        return InferenceEngine.generateTokensGPUOlmoe(this, (OlmoeState) state, startPosition, 
                promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
    
    /**
     * Gets the MoE routing configuration for this model
     */
    public MoERoutingConfig getRoutingConfig() {
        return new MoERoutingConfig(
            configuration.getNumberOfExperts(),
            configuration.getNumberOfActiveExperts(),
            configuration.getRouterAuxLossCoef(),
            configuration.shouldOutputRouterLogits()
        );
    }
    
    /**
     * MoE routing configuration
     */
    public record MoERoutingConfig(
        int numExperts,
        int numActiveExperts,
        float auxLossCoef,
        boolean outputLogits
    ) {}
}