package org.beehive.gpullama3.tokenizer.vocabulary;

import java.util.Arrays;
import java.util.Map;
import java.util.OptionalInt;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {

    // @formatter:off
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i))
        );
    }
    // @formatter:on

    public String get(int tokenIndex) {
        // Debug logging for suspicious token lookups
        if (tokenIndex == 0) {
            System.out.printf("[VOCAB-DEBUG] ⚠️ Looking up token 0 (NULL token)%n");
        }

        if (tokenIndex < 0 || tokenIndex >= tokens.length) {
            System.out.printf("[VOCAB-DEBUG] ❌ OUT OF BOUNDS: tokenIndex=%d, vocab size=%d%n", tokenIndex, tokens.length);
            return "<OUT_OF_BOUNDS>";
        }

        String result = tokens[tokenIndex];

        // Debug logging for problematic results
        if (result == null) {
            System.out.printf("[VOCAB-DEBUG] ❌ NULL token at index %d%n", tokenIndex);
        } else if (result.isEmpty()) {
            System.out.printf("[VOCAB-DEBUG] ⚠️ Empty token at index %d%n", tokenIndex);
        }

        // Verbose logging for token 0 and first few tokens
        if (tokenIndex <= 5) {
            System.out.printf("[VOCAB-DEBUG] Token[%d] → '%s'%n", tokenIndex, result);
        }

        return result;
    }

    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public static Vocabulary loadLlamaVocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }

    public static Vocabulary loadMistralVocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        Vocabulary v = new Vocabulary(tokens, scores);
        return v;
    }

    public static Vocabulary loadQwen3Vocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        return new Vocabulary(tokens, scores);
    }

    public static Vocabulary loadPhi3Vocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        return new Vocabulary(tokens, scores);
    }

    public static Vocabulary loadOlmoeVocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        return new Vocabulary(tokens, scores);
    }

    public int size() {
        return tokens.length;
    }

    /**
     * Only for Mistral.
     */
    public float getScore(int tokenIndex) {
        return scores[tokenIndex];
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Vocabulary:\n");
        sb.append("Tokens: ").append(Arrays.toString(tokens)).append("\n");
        sb.append("Scores: ").append(Arrays.toString(scores)).append("\n");
        sb.append("Token to Index Map:\n");
        tokenToIndex.forEach((token, index) -> sb.append("  ").append(token).append(" -> ").append(index).append("\n"));
        return sb.toString();
    }
}