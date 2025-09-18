package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Tokenizer for Gemma models.
 * 
 * Gemma uses a SentencePiece tokenizer with a large vocabulary (256K tokens).
 * This implementation provides compatibility with the GGUF format.
 */
public class GemmaTokenizer implements Tokenizer {
    
    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;
    
    // Gemma special tokens
    private static final String BOS_TOKEN = "<bos>";
    private static final String EOS_TOKEN = "<eos>";
    private static final String UNK_TOKEN = "<unk>";
    private static final String PAD_TOKEN = "<pad>";
    
    public GemmaTokenizer(Vocabulary vocabulary) {
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
        
        // Gemma may use different special token formats
        vocabulary.getIndex("<|bos|>").ifPresent(idx -> specialTokens.put("<|bos|>", idx));
        vocabulary.getIndex("<|eos|>").ifPresent(idx -> specialTokens.put("<|eos|>", idx));
        vocabulary.getIndex("<|pad|>").ifPresent(idx -> specialTokens.put("<|pad|>", idx));
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
    public String regexPattern() {
        // Gemma uses SentencePiece which doesn't have a regex pattern
        return null;
    }
    
    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        // For now, use simple whitespace tokenization with vocabulary lookup
        // A full SentencePiece implementation would be more complex
        String[] words = text.split("\\s+");
        List<Integer> tokens = new ArrayList<>();
        
        for (String word : words) {
            // Check if it's a special token and if it's allowed
            if (specialTokens.containsKey(word)) {
                if (allowedSpecial.contains("all") || allowedSpecial.contains(word)) {
                    tokens.add(specialTokens.get(word));
                }
            } else {
                // Regular word encoding
                int tokenId = vocabulary.getIndex(word)
                    .orElse(specialTokens.getOrDefault(UNK_TOKEN, 0));

                System.err.printf("[OLMOE-TOKENIZE-DEBUG] Word: '%s' -> Token ID: %d%n", word, tokenId);

                if (tokenId < 0) {
                    System.err.printf("[OLMOE-TOKENIZE-ERROR] Negative token ID %d for word '%s'%n", tokenId, word);
                    System.err.printf("[OLMOE-TOKENIZE-ERROR] UNK_TOKEN value: %s%n", specialTokens.get(UNK_TOKEN));
                }

                tokens.add(tokenId);
            }
        }
        
        return tokens;
    }
    
    @Override
    public List<Integer> encodeAsList(String text) {
        // Simple encoding without special token restrictions
        return encode(text, Set.of("all"));
    }
    
    public Vocabulary getVocabulary() {
        return vocabulary;
    }
    
    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder result = new StringBuilder();

        System.err.printf("[GEMMA-TOKENIZER] Decoding %d tokens: %s%n", tokens.size(),
                          tokens.subList(0, Math.min(10, tokens.size())).toString());

        for (int i = 0; i < tokens.size(); i++) {
            int token = tokens.get(i);
            if (!isSpecialToken(token)) {
                String tokenStr = vocabulary.get(token);

                // Handle null/missing tokens
                if (tokenStr == null) {
                    System.err.printf("[GEMMA-TOKENIZER] Warning: Missing token %d in vocabulary%n", token);
                    continue;
                }

                // Debug first 10 tokens
                if (i < 10) {
                    System.err.printf("[GEMMA-TOKENIZER] Token %d: %d -> '%s' (length=%d)%n",
                                      i, token, tokenStr.replace("\n", "\\n").replace("\t", "\\t"), tokenStr.length());
                }

                // Skip unused tokens and problematic sequences
                if (tokenStr.startsWith("<unused") ||
                    tokenStr.startsWith("<0x") ||
                    tokenStr.startsWith("�") ||
                    tokenStr.equals("") ||
                    tokenStr.matches("^<[^>]*>$")) { // Skip any <tag> format tokens
                    if (i < 10) {
                        System.err.printf("[GEMMA-TOKENIZER] Skipping problematic token: '%s'%n", tokenStr);
                    }
                    continue;
                }

                // Enhanced SentencePiece handling
                if (tokenStr.startsWith("▁")) {
                    // ▁ represents word boundary in SentencePiece
                    if (result.length() > 0) {
                        result.append(" ");
                    }
                    String content = tokenStr.substring(1);
                    if (!content.isEmpty()) {
                        result.append(content);
                    }
                    if (i < 10) {
                        System.err.printf("[GEMMA-TOKENIZER] Added space+content: '%s'%n", content);
                    }
                } else if (tokenStr.startsWith("Ġ")) {
                    // GPT-2 style space prefix (some Gemma models use this)
                    if (result.length() > 0) {
                        result.append(" ");
                    }
                    String content = tokenStr.substring(1);
                    if (!content.isEmpty()) {
                        result.append(content);
                    }
                    if (i < 10) {
                        System.err.printf("[GEMMA-TOKENIZER] Added space+content (GPT2): '%s'%n", content);
                    }
                } else {
                    // Regular token - append directly
                    if (!tokenStr.isEmpty()) {
                        result.append(tokenStr);
                        if (i < 10) {
                            System.err.printf("[GEMMA-TOKENIZER] Added directly: '%s'%n", tokenStr.replace("\n", "\\n").replace("\t", "\\t"));
                        }
                    }
                }
            } else {
                if (i < 10) {
                    System.err.printf("[GEMMA-TOKENIZER] Skipping special token %d%n", token);
                }
            }
        }

        // Clean up the result
        String output = result.toString();

        // Post-process to fix common issues
        output = output.replaceAll("\\s+", " "); // Normalize multiple spaces
        output = output.trim(); // Remove leading/trailing spaces

        System.err.printf("[GEMMA-TOKENIZER] Final output (length=%d): '%s'%n",
                          output.length(), output.length() <= 50 ? output : output.substring(0, 50) + "...");

        return output;
    }
}