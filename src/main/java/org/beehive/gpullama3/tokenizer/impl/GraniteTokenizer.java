package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Tokenizer for IBM Granite models.
 * 
 * Granite uses a SentencePiece-based tokenizer with support for:
 * - Fill-in-the-Middle (FIM) tokens for code completion
 * - Structured reasoning tokens (<think>, </think>)
 * - Standard conversation tokens
 */
public class GraniteTokenizer implements Tokenizer {
    
    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;
    
    // Granite special tokens
    private static final String BOS_TOKEN = "<|begin_of_text|>";
    private static final String EOS_TOKEN = "<|end_of_text|>";
    private static final String UNK_TOKEN = "<unk>";
    private static final String PAD_TOKEN = "<pad>";
    
    // FIM (Fill-in-the-Middle) tokens for code completion
    private static final String FIM_PREFIX = "<|fim_prefix|>";
    private static final String FIM_MIDDLE = "<|fim_middle|>";
    private static final String FIM_SUFFIX = "<|fim_suffix|>";
    
    // Reasoning tokens for structured thinking
    private static final String THINK_START = "<think>";
    private static final String THINK_END = "</think>";
    
    public GraniteTokenizer(Vocabulary vocabulary) {
        this.vocabulary = vocabulary;
        this.specialTokens = new HashMap<>();
        
        // Initialize special tokens
        initializeSpecialTokens();
    }
    
    private void initializeSpecialTokens() {
        // Find and register special tokens in vocabulary
        vocabulary.getIndex(BOS_TOKEN).ifPresent(idx -> specialTokens.put(BOS_TOKEN, idx));
        vocabulary.getIndex(EOS_TOKEN).ifPresent(idx -> specialTokens.put(EOS_TOKEN, idx));
        vocabulary.getIndex(UNK_TOKEN).ifPresent(idx -> specialTokens.put(UNK_TOKEN, idx));
        vocabulary.getIndex(PAD_TOKEN).ifPresent(idx -> specialTokens.put(PAD_TOKEN, idx));
        
        // FIM tokens
        vocabulary.getIndex(FIM_PREFIX).ifPresent(idx -> specialTokens.put(FIM_PREFIX, idx));
        vocabulary.getIndex(FIM_MIDDLE).ifPresent(idx -> specialTokens.put(FIM_MIDDLE, idx));
        vocabulary.getIndex(FIM_SUFFIX).ifPresent(idx -> specialTokens.put(FIM_SUFFIX, idx));
        
        // Reasoning tokens
        vocabulary.getIndex(THINK_START).ifPresent(idx -> specialTokens.put(THINK_START, idx));
        vocabulary.getIndex(THINK_END).ifPresent(idx -> specialTokens.put(THINK_END, idx));
        
        // Alternative token formats
        vocabulary.getIndex("<bos>").ifPresent(idx -> specialTokens.put("<bos>", idx));
        vocabulary.getIndex("<eos>").ifPresent(idx -> specialTokens.put("<eos>", idx));
    }
    
    @Override
    public String regexPattern() {
        // Granite uses SentencePiece which doesn't have a regex pattern
        return null;
    }
    
    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }
    
    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }
    
    @Override
    public boolean shouldDisplayToken(int token) {
        return !isSpecialToken(token);
    }
    
    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        // Simple tokenization implementation
        // A full SentencePiece implementation would be more sophisticated
        String[] parts = text.split("(?<=\\s)|(?=\\s)|(?<=<[^>]*>)|(?=<[^>]*>)");
        List<Integer> tokens = new ArrayList<>();
        
        for (String part : parts) {
            part = part.trim();
            if (part.isEmpty()) continue;
            
            // Check if it's a special token and if it's allowed
            if (specialTokens.containsKey(part)) {
                if (allowedSpecial.contains("all") || allowedSpecial.contains(part)) {
                    tokens.add(specialTokens.get(part));
                }
            } else {
                // Regular word encoding
                tokens.add(vocabulary.getIndex(part)
                    .orElse(specialTokens.getOrDefault(UNK_TOKEN, 0)));
            }
        }
        
        return tokens;
    }
    
    @Override
    public List<Integer> encodeAsList(String text) {
        // Simple encoding allowing all special tokens
        return encode(text, Set.of("all"));
    }
    
    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder result = new StringBuilder();
        
        for (int token : tokens) {
            if (!isSpecialToken(token)) {
                String tokenStr = vocabulary.get(token);
                
                // Handle SentencePiece's ▁ prefix for spaces
                if (tokenStr.startsWith("▁")) {
                    if (result.length() > 0) {
                        result.append(" ");
                    }
                    result.append(tokenStr.substring(1));
                } else {
                    result.append(tokenStr);
                }
            } else {
                // Include special tokens for reasoning/FIM
                String tokenStr = vocabulary.get(token);
                if (tokenStr.equals(THINK_START) || tokenStr.equals(THINK_END) ||
                    tokenStr.startsWith("<|fim_")) {
                    result.append(tokenStr);
                }
            }
        }
        
        return result.toString();
    }
    
    /**
     * Checks if this tokenizer supports Fill-in-the-Middle operations.
     */
    public boolean supportsFIM() {
        return specialTokens.containsKey(FIM_PREFIX) && 
               specialTokens.containsKey(FIM_MIDDLE) && 
               specialTokens.containsKey(FIM_SUFFIX);
    }
}