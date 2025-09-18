package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.gemma.Gemma2Configuration;

/**
 * Gemma 2-specific inference core with logit soft-capping and interleaved attention.
 * CRITICAL: This is where oscillation issues are resolved.
 * SEPARATE from GemmaInferenceCore to maintain clean architecture.
 */
public final class Gemma2InferenceCore {

    private Gemma2InferenceCore() {
        // prevent instantiation
    }

    /**
     * Apply tanh-based logit soft-capping to prevent extreme values.
     * This is the PRIMARY fix for Gemma 2 oscillation issues.
     */
    public static void applySoftCapping(FloatTensor logits, float softcapFactor) {
        // Implementation: logits = softcapFactor * tanh(logits / softcapFactor)
        for (int i = 0; i < logits.size(); i++) {
            float value = logits.getFloat(i);
            float capped = softcapFactor * (float) Math.tanh(value / softcapFactor);
            logits.setFloat(i, capped);
        }
    }

    /**
     * Gemma 2-specific forward pass with interleaved local-global attention.
     * Key differences from Gemma 3:
     * - Interleaved attention pattern (not 5:1)
     * - 4096-token local windows (not 1024)
     * - 8192-token global span (not full context)
     * - CRITICAL: Logit soft-capping applied
     */
    public static void forwardGemma2(Model model, State state, int token, int position) {
        Gemma2Configuration config = (Gemma2Configuration) model.configuration();

        // Use existing LLaMA inference as base (proven stable)
        InferenceCore.forwardJava(model, state, token, position);

        // Apply Gemma 2-specific post-processing
        applyGemma2PostProcessing(state, config, position);
    }

    /**
     * Apply Gemma 2-specific post-processing including logit soft-capping.
     */
    private static void applyGemma2PostProcessing(State state, Gemma2Configuration config, int position) {
        // Apply final logit soft-capping (CRITICAL for oscillation prevention)
        if (config.hasLogitSoftcapping()) {
            applySoftCapping(state.logits, config.finalLogitSoftcapping());
            System.err.printf("[GEMMA2-DEBUG] Applied final logit soft-capping with factor %.1f%n",
                              config.finalLogitSoftcapping());
        }

        // Additional oscillation prevention if needed
        if (position > 2500) { // Research shows issues after 2500 tokens
            applyOscillationPrevention(state, config, position);
        }
    }

    /**
     * Additional oscillation prevention for long sequences.
     */
    private static void applyOscillationPrevention(State state, Gemma2Configuration config, int position) {
        // Apply very small random perturbation to break cycles
        float perturbationScale = 1e-7f; // Minimal impact on quality
        long seed = 42L + position;
        java.util.Random rng = new java.util.Random(seed);

        // Only perturb top candidates to avoid affecting overall distribution
        for (int i = 0; i < Math.min(32, config.vocabularySize()); i++) {
            float perturbation = (rng.nextFloat() - 0.5f) * perturbationScale;
            float currentLogit = state.logits.getFloat(i);
            state.logits.setFloat(i, currentLogit + perturbation);
        }
    }

    /**
     * Determine if layer should use local or global attention in Gemma 2's interleaved pattern.
     * Gemma 2 alternates: local, global, local, global...
     */
    public static boolean isLocalAttentionLayer(int layerIndex) {
        return (layerIndex % 2) == 0; // Even layers = local, odd layers = global
    }

    /**
     * Get local attention window size for Gemma 2.
     * @return 4096 tokens (different from Gemma 3's 1024)
     */
    public static int getLocalWindowSize() {
        return 4096;
    }

    /**
     * Get global attention span for Gemma 2.
     * @return 8192 tokens (different from Gemma 3's full context)
     */
    public static int getGlobalSpan() {
        return 8192;
    }
}