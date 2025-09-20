package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * cl100k_base tokenizer implementation for OLMoE models.
 *
 * This is the same tokenizer used by GPT-3.5 and GPT-4 models.
 * It uses byte-level BPE with specific patterns optimized for modern language understanding.
 *
 * Key features:
 * - Vocabulary size: ~100,256 tokens
 * - Byte-level BPE encoding
 * - Optimized regex patterns for better multilingual support
 * - Special tokens for conversation formatting
 */
public class Cl100kTokenizer implements Tokenizer {

    // cl100k_base regex pattern - optimized for GPT-3.5/GPT-4 style tokenization
    private static final String CL100K_PATTERN =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;

    public Cl100kTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        System.out.println("[CL100K-TOKENIZER] Initializing cl100k_base tokenizer for OLMoE");
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(CL100K_PATTERN);

        // Load merge rules from metadata (if available)
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges;

        if (mergeLines != null) {
            merges = Arrays.stream(mergeLines).map(line -> line.split(" "))
                    .map(parts -> new Pair<>(vocabulary.getIndex(parts[0]).orElseThrow(), vocabulary.getIndex(parts[1]).orElseThrow())).toList();
        } else {
            // OLMoE may not have explicit merges in GGUF metadata, use empty list
            System.out.println("[CL100K-TOKENIZER] No tokenizer.ggml.merges found for OLMoE, using empty merges list");
            merges = new ArrayList<>();
        }

        // Initialize merges map
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            OptionalInt mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex));
            if (mergeIndex.isPresent()) {
                this.merges.put(pair, mergeIndex.getAsInt());
            }
        }

        System.out.printf("[CL100K-TOKENIZER] Initialized with vocabulary size: %d, merges: %d%n",
                         vocabulary.tokens().length, this.merges.size());
    }

    @Override
    public String regexPattern() {
        return CL100K_PATTERN;
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return getSpecialTokens().containsValue(tokenIndex);
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        // Don't display special tokens or control characters
        if (isSpecialToken(token)) {
            return false;
        }

        String tokenText = vocabulary.get(token);
        if (tokenText == null || tokenText.isEmpty()) {
            return false;
        }

        // Filter out control characters
        for (char c : tokenText.toCharArray()) {
            if (Character.isISOControl(c) && c != '\n' && c != '\r' && c != '\t') {
                return false;
            }
        }

        return true;
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        // For now, ignore allowedSpecial and use basic encoding
        return encode(text);
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        return encode(text);
    }

    private List<Integer> encode(String text) {
        System.out.printf("[CL100K-TOKENIZER] Encoding text: '%.50s%s'%n",
                         text, text.length() > 50 ? "..." : "");

        List<String> tokens = new ArrayList<>();

        // Apply regex pattern to split text
        Matcher matcher = compiledPattern.matcher(text);
        while (matcher.find()) {
            tokens.add(matcher.group());
        }

        List<Integer> result = new ArrayList<>();
        for (String token : tokens) {
            // Convert string to bytes and then apply BPE
            byte[] bytes = token.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            List<Integer> tokenIds = encodeBytesToTokens(bytes);
            result.addAll(tokenIds);
        }

        System.out.printf("[CL100K-TOKENIZER] Encoded %d characters to %d tokens%n",
                         text.length(), result.size());
        return result;
    }

    private List<Integer> encodeBytesToTokens(byte[] bytes) {
        // Convert bytes to initial token sequence (following LlamaTokenizer approach)
        List<Integer> tokens = new ArrayList<>();
        for (byte b : bytes) {
            int byteValue = Byte.toUnsignedInt(b);
            String charStr = String.valueOf((char) byteValue);
            OptionalInt tokenId = vocabulary.getIndex(charStr);
            if (tokenId.isPresent()) {
                tokens.add(tokenId.getAsInt());
            } else {
                // Handle unknown character: try fallback strategies similar to LlamaTokenizer
                // Strategy 1: Use the byte value as string
                OptionalInt byteTokenId = vocabulary.getIndex(String.valueOf(byteValue));
                if (byteTokenId.isPresent()) {
                    tokens.add(byteTokenId.getAsInt());
                } else {
                    // Strategy 2: Use unknown token (ID 0) or space token as fallback
                    OptionalInt spaceToken = vocabulary.getIndex(" ");
                    tokens.add(spaceToken.orElse(0)); // Use 0 as ultimate fallback
                }
            }
        }

        // Apply BPE merges (following LlamaTokenizer approach)
        while (tokens.size() >= 2) {
            // Find the pair with the lowest merge index (highest priority)
            Pair<Integer, Integer> bestPair = null;
            int bestMergeIndex = Integer.MAX_VALUE;

            for (int i = 0; i < tokens.size() - 1; i++) {
                Pair<Integer, Integer> pair = new Pair<>(tokens.get(i), tokens.get(i + 1));
                if (merges.containsKey(pair)) {
                    int mergeIndex = merges.get(pair);
                    if (mergeIndex < bestMergeIndex) {
                        bestPair = pair;
                        bestMergeIndex = mergeIndex;
                    }
                }
            }

            if (bestPair == null) {
                break; // No more merges possible
            }

            // Apply the best merge
            List<Integer> newTokens = new ArrayList<>();
            int i = 0;
            while (i < tokens.size()) {
                if (i < tokens.size() - 1 &&
                    tokens.get(i).equals(bestPair.first()) &&
                    tokens.get(i + 1).equals(bestPair.second())) {
                    newTokens.add(bestMergeIndex);
                    i += 2;
                } else {
                    newTokens.add(tokens.get(i));
                    i++;
                }
            }
            tokens = newTokens;
        }

        return tokens;
    }

    @Override
    public String decode(List<Integer> tokens) {
        System.out.printf("[CL100K-TOKENIZER] Decoding %d tokens%n", tokens.size());

        StringBuilder result = new StringBuilder();
        for (Integer tokenId : tokens) {
            String tokenText = vocabulary.get(tokenId);
            if (tokenText != null) {
                result.append(tokenText);
            }
        }

        String decoded = result.toString();
        System.out.printf("[CL100K-TOKENIZER] Decoded to %d characters%n", decoded.length());
        return decoded;
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        Map<String, Integer> specialTokens = new HashMap<>();

        // OLMoE uses truncated cl100k_base vocabulary (only first ~50K tokens)
        // Search for actual special tokens in the vocabulary
        int vocabSize = vocabulary.tokens().length;
        System.out.printf("[CL100K-TOKENIZER] Searching for special tokens in vocabulary of size %d%n", vocabSize);

        // Look for common special tokens in the actual vocabulary
        for (int i = 0; i < vocabSize; i++) {
            String token = vocabulary.get(i);
            if (token != null) {
                switch (token) {
                    case "<|endoftext|>" -> {
                        specialTokens.put("<|endoftext|>", i);
                        System.out.printf("[CL100K-TOKENIZER] Found <|endoftext|> at ID %d%n", i);
                    }
                    case "<|startoftext|>" -> {
                        specialTokens.put("<|startoftext|>", i);
                        System.out.printf("[CL100K-TOKENIZER] Found <|startoftext|> at ID %d%n", i);
                    }
                    case "<|im_start|>" -> {
                        specialTokens.put("<|im_start|>", i);
                        System.out.printf("[CL100K-TOKENIZER] Found <|im_start|> at ID %d%n", i);
                    }
                    case "<|im_end|>" -> {
                        specialTokens.put("<|im_end|>", i);
                        System.out.printf("[CL100K-TOKENIZER] Found <|im_end|> at ID %d%n", i);
                    }
                    case "<|fim_prefix|>" -> {
                        specialTokens.put("<|fim_prefix|>", i);
                        System.out.printf("[CL100K-TOKENIZER] Found <|fim_prefix|> at ID %d%n", i);
                    }
                    case "<|fim_middle|>" -> {
                        specialTokens.put("<|fim_middle|>", i);
                        System.out.printf("[CL100K-TOKENIZER] Found <|fim_middle|> at ID %d%n", i);
                    }
                    case "<|fim_suffix|>" -> {
                        specialTokens.put("<|fim_suffix|>", i);
                        System.out.printf("[CL100K-TOKENIZER] Found <|fim_suffix|> at ID %d%n", i);
                    }
                }
            }
        }

        // Fallback: if no special tokens found, use safe defaults within vocabulary range
        if (specialTokens.isEmpty()) {
            System.out.println("[CL100K-TOKENIZER] No standard special tokens found, using fallback strategy");
            // Use end of vocabulary for special tokens
            specialTokens.put("<|endoftext|>", vocabSize - 1);
            specialTokens.put("<|startoftext|>", vocabSize - 2);
        }

        System.out.printf("[CL100K-TOKENIZER] Loaded %d special tokens%n", specialTokens.size());
        return specialTokens;
    }

}