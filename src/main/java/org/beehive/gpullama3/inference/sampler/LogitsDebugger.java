package org.beehive.gpullama3.inference.sampler;

import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * CRITICAL DEBUG: Helper class to analyze logits distribution and understand context independence
 */
public class LogitsDebugger {
    private static int logitsDebugCallCount = 0;

    /**
     * CRITICAL DEBUG: Analyzes logits distribution to understand why context independence occurs
     */
    public static void debugLogitsDistribution(Object logits, String tensorType, Model model) {
        logitsDebugCallCount++;

        // Only debug first few generations to avoid spam
        if (logitsDebugCallCount > 3) {
            return;
        }

        System.out.println("\n[LOGITS-DEBUG-" + logitsDebugCallCount + "] 🔍 Raw Logits Analysis (" + tensorType + "):");

        Vocabulary vocab = null;
        if (model != null && model.tokenizer() instanceof org.beehive.gpullama3.tokenizer.impl.OlmoTokenizer) {
            vocab = ((org.beehive.gpullama3.tokenizer.impl.OlmoTokenizer) model.tokenizer()).getVocabulary();
        }

        if (logits instanceof FloatTensor) {
            FloatTensor tensor = (FloatTensor) logits;
            debugFloatTensorLogits(tensor, logitsDebugCallCount, vocab);
        } else if (logits instanceof FloatArray) {
            FloatArray array = (FloatArray) logits;
            debugFloatArrayLogits(array, logitsDebugCallCount, vocab);
        }
    }

    private static void debugFloatTensorLogits(FloatTensor tensor, int generation, Vocabulary vocab) {
        int vocabSize = tensor.size();
        System.out.printf("[LOGITS-DEBUG-%d] Vocabulary size: %d%n", generation, vocabSize);

        // Find top 10 logits
        java.util.List<java.util.Map.Entry<Integer, Float>> topLogits = new java.util.ArrayList<>();
        for (int i = 0; i < vocabSize; i++) {
            topLogits.add(new java.util.AbstractMap.SimpleEntry<>(i, tensor.getFloat(i)));
        }
        topLogits.sort((a, b) -> Float.compare(b.getValue(), a.getValue())); // Descending order

        // Show top 10 with token IDs and values
        System.out.printf("[LOGITS-DEBUG-%d] Top 10 raw logits:%n", generation);
        for (int i = 0; i < Math.min(10, topLogits.size()); i++) {
            var entry = topLogits.get(i);
            String tokenText = (vocab != null) ? vocab.get(entry.getKey()) : "unknown";
            System.out.printf("  [%d] Token %d ('%s'): %.6f%n", i+1, entry.getKey(), tokenText, entry.getValue());
        }

        // Show statistics
        float max = topLogits.get(0).getValue();
        float min = tensor.getFloat(0);
        for (int i = 1; i < vocabSize; i++) {
            min = Math.min(min, tensor.getFloat(i));
        }
        float sum = 0;
        for (int i = 0; i < vocabSize; i++) {
            sum += tensor.getFloat(i);
        }
        float mean = sum / vocabSize;

        System.out.printf("[LOGITS-DEBUG-%d] Stats: max=%.6f, min=%.6f, mean=%.6f, range=%.6f%n",
                         generation, max, min, mean, max - min);
    }

    private static void debugFloatArrayLogits(FloatArray array, int generation, Vocabulary vocab) {
        int vocabSize = array.getSize();
        System.out.printf("[LOGITS-DEBUG-%d] Vocabulary size: %d%n", generation, vocabSize);

        // Find top 10 logits
        java.util.List<java.util.Map.Entry<Integer, Float>> topLogits = new java.util.ArrayList<>();
        for (int i = 0; i < vocabSize; i++) {
            topLogits.add(new java.util.AbstractMap.SimpleEntry<>(i, array.get(i)));
        }
        topLogits.sort((a, b) -> Float.compare(b.getValue(), a.getValue())); // Descending order

        // Show top 10 with token IDs and values
        System.out.printf("[LOGITS-DEBUG-%d] Top 10 raw logits:%n", generation);
        for (int i = 0; i < Math.min(10, topLogits.size()); i++) {
            var entry = topLogits.get(i);
            String tokenText = (vocab != null) ? vocab.get(entry.getKey()) : "unknown";
            System.out.printf("  [%d] Token %d ('%s'): %.6f%n", i+1, entry.getKey(), tokenText, entry.getValue());
        }

        // Show statistics
        float max = topLogits.get(0).getValue();
        float min = array.get(0);
        for (int i = 1; i < vocabSize; i++) {
            min = Math.min(min, array.get(i));
        }
        float sum = 0;
        for (int i = 0; i < vocabSize; i++) {
            sum += array.get(i);
        }
        float mean = sum / vocabSize;

        System.out.printf("[LOGITS-DEBUG-%d] Stats: max=%.6f, min=%.6f, mean=%.6f, range=%.6f%n",
                         generation, max, min, mean, max - min);
    }
}