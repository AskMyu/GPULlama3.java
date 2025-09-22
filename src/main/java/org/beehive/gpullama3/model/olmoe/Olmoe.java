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

        // OLMOE FIX: Set proper BOS token to avoid Token 0 cascade failure
        // OLMoE uses GPT-NeoX tokenizer where <|endoftext|> (50279) is the BOS token
        if (tokenizer.getSpecialTokens().containsKey("<|endoftext|>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<|endoftext|>");
            System.err.printf("[OLMOE-STATE] Using BOS token: %d (<|endoftext|>)%n", state.latestToken);
        } else if (chatFormat != null && chatFormat.chatTokens() != null &&
                   chatFormat.chatTokens().tStartHeader() != null) {
            Integer startToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
            if (startToken != null) {
                state.latestToken = startToken;
                System.err.printf("[OLMOE-STATE] Using chat format token: %d%n", state.latestToken);
            }
        } else {
            // Fallback: Use token 1 instead of dangerous token 0
            state.latestToken = 1;
            System.err.printf("[OLMOE-STATE] Using fallback token 1 (avoiding token 0)%n");
        }

        // Safety check to prevent Token 0 cascade failure
        if (state.latestToken == 0) {
            System.err.println("[OLMOE-STATE] CRITICAL: Token 0 detected, forcing to proper BOS token");
            state.latestToken = tokenizer.getSpecialTokens().getOrDefault("<|endoftext|>", 1);
        }

        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new OlmoeState(configuration(), batchsize);

        // OLMOE FIX: Set proper BOS token to avoid Token 0 cascade failure
        // OLMoE uses GPT-NeoX tokenizer where <|endoftext|> (50279) is the BOS token
        if (tokenizer.getSpecialTokens().containsKey("<|endoftext|>")) {
            state.latestToken = tokenizer.getSpecialTokens().get("<|endoftext|>");
            System.err.printf("[OLMOE-STATE-BATCH] Using BOS token: %d (<|endoftext|>)%n", state.latestToken);
        } else if (chatFormat != null && chatFormat.chatTokens() != null &&
                   chatFormat.chatTokens().tStartHeader() != null) {
            Integer startToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
            if (startToken != null) {
                state.latestToken = startToken;
                System.err.printf("[OLMOE-STATE-BATCH] Using chat format token: %d%n", state.latestToken);
            }
        } else {
            // Fallback: Use token 1 instead of dangerous token 0
            state.latestToken = 1;
            System.err.printf("[OLMOE-STATE-BATCH] Using fallback token 1 (avoiding token 0)%n");
        }

        // Safety check to prevent Token 0 cascade failure
        if (state.latestToken == 0) {
            System.err.println("[OLMOE-STATE-BATCH] CRITICAL: Token 0 detected, forcing to proper BOS token");
            state.latestToken = tokenizer.getSpecialTokens().getOrDefault("<|endoftext|>", 1);
        }

        return state;
    }

    /**
     * OLMoE models need begin of text token
     */
    @Override
    public boolean shouldAddBeginOfText() {
        // CRITICAL FIX: OLMoE Tulu chat format already includes proper sequence start
        // Adding a separate BOS token creates duplicate leading tokens which breaks attention
        return false;
    }

    @Override
    public void forward(State state, int token, int position) {
        System.err.printf("[OLMOE-FORWARD-DEBUG] plan=%s, tornadoVMPlan()=%s%n",
                         plan, (plan != null) ? "available" : "null");

        // CRITICAL FIX: Always use our fixed OLMoE implementation regardless of plan status
        // This ensures our state management fix is used instead of generic TornadoVM kernels
        System.err.println("[OLMOE-FORWARD-DEBUG] Using OLMoE-specific GPU forward pass with fixed state management");
        InferenceCore.forwardTornadoVMOlmoe(this, (OlmoeState) state, token, position, tornadoVMPlan());
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