package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tokenizer for Gemma models.
 *
 * Gemma uses a SentencePiece-style BPE tokenizer with a large vocabulary (256K tokens).
 * This implementation uses proper BPE tokenization with merge rules.
 */
public class GemmaTokenizer implements Tokenizer {

    // Gemma uses a similar regex pattern to Llama 3 for tokenization
    private static final String GEMMA_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;

    // Gemma special tokens
    private static final String BOS_TOKEN = "<bos>";
    private static final String EOS_TOKEN = "<eos>";
    private static final String UNK_TOKEN = "<unk>";
    private static final String PAD_TOKEN = "<pad>";

    public GemmaTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(GEMMA_PATTERN);

        // Load merge rules from metadata (if available)
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges;

        if (mergeLines != null) {
            merges = Arrays.stream(mergeLines).map(line -> line.split(" "))
                    .map(parts -> new Pair<>(vocabulary.getIndex(parts[0]).orElseThrow(), vocabulary.getIndex(parts[1]).orElseThrow())).toList();
        } else {
            // Gemma may not have explicit merges in GGUF metadata
            System.out.println("DEBUG: No tokenizer.ggml.merges found for Gemma, using empty merges list");
            merges = new ArrayList<>();
        }

        // Initialize merges map
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }

        // Initialize special tokens
        this.specialTokens = new HashMap<>();
        initializeSpecialTokens();
    }

    // Legacy constructor for backward compatibility
    public GemmaTokenizer(Vocabulary vocabulary) {
        this(new HashMap<>(), vocabulary);
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
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }
    
    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        // decode the user desire w.r.t. handling of special tokens
        Set<String> special = allowedSpecial;
        assert getSpecialTokens().keySet().containsAll(special);
        if (special.isEmpty()) {
            // shortcut: if no special tokens, just use the ordinary encoding
            return encodeOrdinary(text);
        }

        // otherwise, we have to be careful with potential special tokens in text
        // we handle special tokens by splitting the text
        // based on the occurrence of any exact match with any of the special tokens
        String specialPattern = special
                .stream()
                .map(Pattern::quote)
                .collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        // now all the special characters are separated from the rest of the text
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (special.contains(part)) {
                // this is a special token, encode it separately as a special case
                ids.add(getSpecialTokens().get(part));
            } else {
                // this is an ordinary sequence, encode it normally
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    private List<Integer> encodeOrdinary(String text) {
        // SentencePiece-style encoding for Gemma (not BPE)
        // Since Gemma doesn't have merge rules, use vocabulary lookup with fallback
        List<Integer> tokens = new ArrayList<>();

        // First try to find exact matches in vocabulary
        String remaining = text;
        while (!remaining.isEmpty()) {
            boolean found = false;

            // Try progressively shorter substrings starting from the longest
            for (int len = Math.min(remaining.length(), 50); len > 0; len--) {
                String substr = remaining.substring(0, len);
                int tokenId = vocabulary.getIndex(substr).orElse(-1);

                if (tokenId >= 0) {
                    tokens.add(tokenId);
                    remaining = remaining.substring(len);
                    found = true;
                    break;
                }
            }

            // If no exact match found, try common SentencePiece patterns
            if (!found) {
                char firstChar = remaining.charAt(0);

                // Try with SentencePiece prefix (▁ for word boundary)
                String withPrefix = "▁" + firstChar;
                int tokenId = vocabulary.getIndex(withPrefix).orElse(-1);
                if (tokenId >= 0) {
                    tokens.add(tokenId);
                    remaining = remaining.substring(1);
                    continue;
                }

                // Try single character
                tokenId = vocabulary.getIndex(String.valueOf(firstChar)).orElse(-1);
                if (tokenId >= 0) {
                    tokens.add(tokenId);
                    remaining = remaining.substring(1);
                    continue;
                }

                // Fallback to UNK token
                tokens.add(specialTokens.getOrDefault(UNK_TOKEN, 3));
                remaining = remaining.substring(1);
            }
        }

        return tokens;
    }

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
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