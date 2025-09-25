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
import java.util.ArrayList;
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
        // CRITICAL FIX: Clear KV cache before new sequence to prevent context contamination
        if (state instanceof OlmoeState olmoeState) {
            olmoeState.clearKVCache();
        }

        // OLMoE-specific token generation with MoE
        return InferenceEngine.generateTokensOlmoe(this, (OlmoeState) state, startPosition,
                promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        // CRITICAL FIX: Clear KV cache before new sequence to prevent context contamination
        if (state instanceof OlmoeState olmoeState) {
            olmoeState.clearKVCache();
        }

        // Check if batch processing is enabled via system property or configuration
        boolean batchEnabled = Boolean.parseBoolean(System.getProperty("olmoe.batch.enabled", "true"));
        System.err.printf("[OLMOE-BATCH] Batch processing flag: %s%n", batchEnabled);

        if (batchEnabled) {
            // 🚀 BATCH PROCESSING FOR OLMOE: Two-phase approach like llama.cpp
            System.err.println("[OLMOE-BATCH] 🎯 ACTIVATING BATCH PROCESSING TO SOLVE EXPERT ROUTING CONTEXT ISOLATION");

            try {
                // PHASE 1: Batch process the entire prompt using OLMoEBatchProcessor
                System.err.printf("[OLMOE-BATCH] Phase 1: Batch processing prompt (%d tokens)%n", promptTokens.size());

                org.beehive.gpullama3.model.olmoe.OLMoEBatchProcessorSimple batchProcessor =
                    new org.beehive.gpullama3.model.olmoe.OLMoEBatchProcessorSimple(
                        this, configuration(), weights(), true, 512);

                // Process entire prompt in batch to build proper context with expert consistency
                state = batchProcessor.forwardBatch(state, promptTokens, startPosition);

                System.err.printf("[OLMOE-BATCH] ✅ Phase 1 completed: %d tokens processed in batch%n", promptTokens.size());

                // PHASE 2: Serial generation using the original GPU method with proper context
                System.err.printf("[OLMOE-BATCH] Phase 2: Serial generation from position %d%n", startPosition + promptTokens.size());

                int newStartPosition = startPosition + promptTokens.size();
                int remainingMaxTokens = Math.max(0, maxTokens - promptTokens.size());

                if (remainingMaxTokens > 0) {
                    // Use original OLMoE GPU generation for the serial phase
                    List<Integer> serialTokens = InferenceEngine.generateTokensGPUOlmoe(
                        this, (OlmoeState) state, newStartPosition,
                        new ArrayList<>(), // Empty prompt tokens since we already processed the prompt
                        stopTokens, remainingMaxTokens, sampler, false, // echo=false for generation phase
                        onTokenGenerated, tornadoVMPlan);

                    System.err.printf("[OLMOE-BATCH] ✅ Phase 2 completed: %d tokens generated serially%n", serialTokens.size());

                    // Combine prompt and generated tokens based on echo setting
                    List<Integer> responseTokens = new ArrayList<>();
                    if (echo) {
                        responseTokens.addAll(promptTokens);
                    }
                    responseTokens.addAll(serialTokens);

                    System.err.printf("[OLMOE-BATCH] ✅ BATCH PROCESSING COMPLETED: %d total tokens%n", responseTokens.size());
                    return responseTokens;
                } else {
                    // No generation needed, just return prompt tokens if echo is enabled
                    List<Integer> responseTokens = new ArrayList<>();
                    if (echo) {
                        responseTokens.addAll(promptTokens);
                    }
                    System.err.printf("[OLMOE-BATCH] ✅ BATCH PROCESSING COMPLETED: %d prompt tokens (no generation)%n", responseTokens.size());
                    return responseTokens;
                }

            } catch (Exception e) {
                System.err.printf("[OLMOE-BATCH] ❌ Batch processing failed: %s%n", e.getMessage());
                System.err.println("[OLMOE-BATCH] 🔄 Falling back to original OLMoE generation");
                e.printStackTrace();

                // Fall through to serial processing
            }
        } else {
            System.err.println("[OLMOE-BATCH] 🔄 Batch processing DISABLED - using serial processing");
        }

        // SERIAL PROCESSING: Original OLMoE generation (used when batch disabled or as fallback)
        System.err.println("[OLMOE-BATCH] 📋 Using original serial OLMoE generation");
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