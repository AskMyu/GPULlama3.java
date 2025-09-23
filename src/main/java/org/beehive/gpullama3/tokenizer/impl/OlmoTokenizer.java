package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * OLMo tokenizer implementation for OLMoE models.
 *
 * This tokenizer uses the OLMo tokenization specification:
 * - Uses GPT2-style BPE with OLMo vocabulary
 * - Vocabulary size: 50,280 tokens (padded to 50,304 for efficiency)
 * - GPT2-style regex pattern (same as llama.cpp LLAMA_VOCAB_PRE_TYPE_OLMO)
 * - Special PII masking tokens: |||PHONE_NUMBER|||, |||EMAIL_ADDRESS|||, |||IP_ADDRESS|||
 * - Special tokens: EOS=50279, PAD=1
 * - Trained on Dolma dataset with PII masking
 *
 * Key differences from GPT-NeoX:
 * - Uses GPT2-style regex pattern instead of GPT-NeoX pattern
 * - Different vocabulary mapping (OLMo vs GPT-NeoX token mappings)
 * - Includes PII masking special tokens
 * - Optimized for diverse text sources in Dolma dataset
 */
public class OlmoTokenizer implements Tokenizer {

    // OLMo uses GPT2-style regex pattern - matches llama.cpp LLAMA_VOCAB_PRE_TYPE_OLMO
    // Pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    // This is the same pattern used by GPT2, MPT, and other models in llama.cpp
    private static final String OLMO_PATTERN =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";

    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<Pair<Integer, Integer>, Integer> mergePriorities;
    private final Map<String, Integer> specialTokens;

    public OlmoTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        System.out.println("[OLMO-TOKENIZER] Initializing OLMo tokenizer with GPT2-style pattern");
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(OLMO_PATTERN);

        // Load merge rules from metadata (if available)
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges;

        if (mergeLines != null) {
            System.out.println("[OLMO-TOKENIZER] Loading " + mergeLines.length + " merge rules from GGUF metadata");
            merges = Arrays.stream(mergeLines).map(line -> line.split(" "))
                    .map(parts -> new Pair<>(vocabulary.getIndex(parts[0]).orElseThrow(), vocabulary.getIndex(parts[1]).orElseThrow())).toList();
        } else {
            // OLMoE may not have explicit merges in GGUF metadata, use empty list
            System.out.println("[OLMO-TOKENIZER] No tokenizer.ggml.merges found for OLMoE, using empty merges list");
            merges = new ArrayList<>();
        }

        // Initialize merges map: pair -> result token ID, and priorities map: pair -> merge order
        this.merges = new HashMap<>();
        this.mergePriorities = new HashMap<>();
        for (int i = 0; i < merges.size(); i++) {
            Pair<Integer, Integer> pair = merges.get(i);
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            OptionalInt mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex));
            if (mergeIndex.isPresent()) {
                // Store the result token ID
                this.merges.put(pair, mergeIndex.getAsInt());
                // Store merge priority (earlier in list = higher priority = lower number)
                this.mergePriorities.put(pair, i);
            }
        }

        // Initialize special tokens
        this.specialTokens = new HashMap<>();

        // Standard OLMo special tokens
        this.specialTokens.put("<|endoftext|>", 50279);  // EOS token
        this.specialTokens.put("<|padding|>", 1);        // PAD token

        // OLMo PII masking tokens (if present in vocabulary)
        if (vocabulary.getIndex("|||PHONE_NUMBER|||").isPresent()) {
            this.specialTokens.put("|||PHONE_NUMBER|||", vocabulary.getIndex("|||PHONE_NUMBER|||").getAsInt());
            System.out.println("[OLMO-TOKENIZER] Found PII masking token: |||PHONE_NUMBER|||");
        }
        if (vocabulary.getIndex("|||EMAIL_ADDRESS|||").isPresent()) {
            this.specialTokens.put("|||EMAIL_ADDRESS|||", vocabulary.getIndex("|||EMAIL_ADDRESS|||").getAsInt());
            System.out.println("[OLMO-TOKENIZER] Found PII masking token: |||EMAIL_ADDRESS|||");
        }
        if (vocabulary.getIndex("|||IP_ADDRESS|||").isPresent()) {
            this.specialTokens.put("|||IP_ADDRESS|||", vocabulary.getIndex("|||IP_ADDRESS|||").getAsInt());
            System.out.println("[OLMO-TOKENIZER] Found PII masking token: |||IP_ADDRESS|||");
        }

        System.out.println("[OLMO-TOKENIZER] Initialized with " + vocabulary.size() + " tokens, " +
                          this.merges.size() + " merges, " + this.specialTokens.size() + " special tokens");
        System.out.println("[OLMO-TOKENIZER] Using GPT2-style pattern: " + OLMO_PATTERN);
    }

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return new HashMap<>(specialTokens);
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        // Don't display special tokens or control characters
        if (isSpecialToken(token)) {
            return false;
        }
        String tokenStr = vocabulary.get(token);
        if (tokenStr == null || tokenStr.isEmpty()) {
            return false;
        }
        // Don't display tokens that are only control characters
        return !tokenStr.chars().allMatch(Character::isISOControl);
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        return encode(text, Set.of());
    }

    public List<Integer> encodeOrdinary(String text) {
        List<Integer> ret = new ArrayList<>();
        Matcher matcher = compiledPattern.matcher(text);
        while (matcher.find()) {
            String piece = matcher.group();
            Integer token = specialTokens.get(piece);
            if (token != null) {
                ret.add(token);
            } else {
                ret.addAll(encode_chunk(piece));
            }
        }
        return ret;
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        System.out.println("[OLMO-ENCODE-DEBUG] 🔍 Starting encode for: '" + text + "'");
        System.out.println("[OLMO-ENCODE-DEBUG] Allowed special tokens: " + allowedSpecial);

        if (allowedSpecial == null) allowedSpecial = Collections.emptySet();
        List<Integer> ret = new ArrayList<>();
        int start = 0;
        while (true) {
            int nextSpecial = Integer.MAX_VALUE;
            String nextSpecialToken = null;
            for (String specialToken : specialTokens.keySet()) {
                if (allowedSpecial.contains(specialToken)) {
                    int idx = text.indexOf(specialToken, start);
                    if (idx != -1 && idx < nextSpecial) {
                        nextSpecial = idx;
                        nextSpecialToken = specialToken;
                    }
                }
            }
            if (nextSpecial == Integer.MAX_VALUE) {
                String remaining = text.substring(start);
                System.out.println("[OLMO-ENCODE-DEBUG] Encoding ordinary text: '" + remaining + "'");
                List<Integer> ordinaryTokens = encodeOrdinary(remaining);
                System.out.println("[OLMO-ENCODE-DEBUG] Ordinary tokens: " + ordinaryTokens);
                ret.addAll(ordinaryTokens);
                break;
            }
            if (nextSpecial > start) {
                String beforeSpecial = text.substring(start, nextSpecial);
                System.out.println("[OLMO-ENCODE-DEBUG] Encoding before special: '" + beforeSpecial + "'");
                List<Integer> beforeTokens = encodeOrdinary(beforeSpecial);
                System.out.println("[OLMO-ENCODE-DEBUG] Before special tokens: " + beforeTokens);
                ret.addAll(beforeTokens);
            }
            Integer specialTokenId = specialTokens.get(nextSpecialToken);
            System.out.println("[OLMO-ENCODE-DEBUG] Adding special token '" + nextSpecialToken + "' → " + specialTokenId);
            ret.add(specialTokenId);
            start = nextSpecial + nextSpecialToken.length();
        }

        System.out.println("[OLMO-ENCODE-DEBUG] ✅ Final encoded tokens: " + ret);
        System.out.println("[OLMO-ENCODE-DEBUG] 🔍 Token → Text mapping:");
        for (int i = 0; i < ret.size(); i++) {
            int tokenId = ret.get(i);
            String tokenStr = vocabulary.get(tokenId);
            System.out.printf("[OLMO-ENCODE-DEBUG] Token[%d]: %d → '%s'%n", i, tokenId, tokenStr);
        }

        return ret;
    }

    @Override
    public String decode(List<Integer> tokens) {
        System.out.println("[OLMO-DECODE-DEBUG] 🔍 Starting decode for tokens: " + tokens);

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tokens.size(); i++) {
            Integer token = tokens.get(i);
            String tokenStr = vocabulary.get(token);
            System.out.printf("[OLMO-DECODE-DEBUG] Token[%d]: %d → '%s'%n", i, token, tokenStr);

            if (tokenStr == null) {
                System.out.printf("[OLMO-DECODE-DEBUG] ❌ NULL vocabulary entry for token %d%n", token);
                tokenStr = "<UNK>";
            } else if (tokenStr.isEmpty()) {
                System.out.printf("[OLMO-DECODE-DEBUG] ⚠️ Empty vocabulary entry for token %d%n", token);
            }

            sb.append(tokenStr);
        }

        String result = sb.toString();
        System.out.println("[OLMO-DECODE-DEBUG] ✅ Final decoded text: '" + result + "'");
        return result;
    }

    public Vocabulary getVocabulary() {
        return vocabulary;
    }

    private List<Integer> encode_chunk(String chunk) {
        System.out.println("[OLMO-CHUNK-DEBUG] 🔍 Encoding chunk: '" + chunk + "'");

        // STEP 1: Check if the entire chunk exists as a token (most important!)
        OptionalInt wholeTokenId = vocabulary.getIndex(chunk);
        if (wholeTokenId.isPresent()) {
            System.out.println("[OLMO-CHUNK-DEBUG] ✅ Found whole chunk as token: " + wholeTokenId.getAsInt());
            return List.of(wholeTokenId.getAsInt());
        }

        System.out.println("[OLMO-CHUNK-DEBUG] Chunk not found as whole token, falling back to BPE");

        // STEP 2: Fall back to byte-level BPE only if whole chunk not found
        List<Integer> tokens = new ArrayList<>();
        try {
            // CRITICAL FIX: Convert to GPT2-style byte encoding
            // GPT2/OLMo uses a special byte-to-unicode mapping for all bytes
            String encoded = bytesToUnicode(chunk.getBytes("UTF-8"));
            System.out.println("[OLMO-CHUNK-DEBUG] GPT2-style encoded: '" + encoded + "'");

            // Now tokenize the unicode-encoded string character by character
            for (int i = 0; i < encoded.length(); i++) {
                String ch = encoded.substring(i, i + 1);
                OptionalInt charTokenId = vocabulary.getIndex(ch);
                if (charTokenId.isPresent()) {
                    tokens.add(charTokenId.getAsInt());
                    System.out.printf("[OLMO-CHUNK-DEBUG] Char[%d]: '%s' → token %d%n", i, ch, charTokenId.getAsInt());
                } else {
                    // Try to find a longer sequence
                    System.out.printf("[OLMO-CHUNK-DEBUG] ⚠️ Character '%s' not found individually, will rely on merges%n", ch);
                    // As last resort, use the unknown token (typically 3)
                    int unkToken = vocabulary.getIndex("<unk>").orElse(vocabulary.getIndex("UNK").orElse(3));
                    tokens.add(unkToken);
                    System.out.printf("[OLMO-CHUNK-DEBUG] Using UNK token: %d%n", unkToken);
                }
            }
        } catch (Exception e) {
            System.out.println("[OLMO-CHUNK-DEBUG] ❌ Exception during byte processing: " + e.getMessage());
            // Use unknown token instead of 0
            int unkToken = vocabulary.getIndex("<unk>").orElse(vocabulary.getIndex("UNK").orElse(3));
            return List.of(unkToken);
        }

        System.out.println("[OLMO-CHUNK-DEBUG] Initial byte tokens: " + tokens);

        // STEP 3: Apply BPE merges with proper priority
        int mergeCount = 0;
        while (tokens.size() >= 2) {
            Optional<Pair<Integer, Integer>> bestPair = findBestMergePair(tokens);
            if (bestPair.isEmpty()) {
                System.out.println("[OLMO-CHUNK-DEBUG] No more merge pairs found");
                break;
            }
            Pair<Integer, Integer> pair = bestPair.get();
            int mergedToken = merges.get(pair);
            System.out.printf("[OLMO-CHUNK-DEBUG] Merge[%d]: (%d,%d) → %d%n", mergeCount++, pair.first(), pair.second(), mergedToken);
            tokens = mergePair(tokens, pair, mergedToken);
            System.out.println("[OLMO-CHUNK-DEBUG] After merge: " + tokens);
        }

        System.out.println("[OLMO-CHUNK-DEBUG] ✅ Final chunk tokens: " + tokens);
        return tokens;
    }

    private Optional<Pair<Integer, Integer>> findBestMergePair(List<Integer> tokens) {
        Map<Pair<Integer, Integer>, Integer> counts = new HashMap<>();
        for (int i = 0; i < tokens.size() - 1; i++) {
            Pair<Integer, Integer> pair = new Pair<>(tokens.get(i), tokens.get(i + 1));
            if (merges.containsKey(pair)) {
                counts.put(pair, counts.getOrDefault(pair, 0) + 1);
            }
        }
        if (counts.isEmpty()) {
            return Optional.empty();
        }
        return counts.entrySet().stream()
                .max(Comparator.<Map.Entry<Pair<Integer, Integer>, Integer>>comparingInt(Map.Entry::getValue)
                        .thenComparing(entry -> -mergePriorities.getOrDefault(entry.getKey(), Integer.MAX_VALUE)))
                .map(Map.Entry::getKey);
    }

    private List<Integer> mergePair(List<Integer> tokens, Pair<Integer, Integer> pair, int mergedToken) {
        List<Integer> newTokens = new ArrayList<>();
        int i = 0;
        while (i < tokens.size()) {
            if (i < tokens.size() - 1 && tokens.get(i).equals(pair.first()) && tokens.get(i + 1).equals(pair.second())) {
                newTokens.add(mergedToken);
                i += 2;
            } else {
                newTokens.add(tokens.get(i));
                i++;
            }
        }
        return newTokens;
    }

    /**
     * GPT2-style byte-to-unicode encoding
     * This converts raw bytes to a unicode string using the GPT2 mapping
     * which ensures all bytes (including whitespace) can be represented
     */
    private String bytesToUnicode(byte[] bytes) {
        StringBuilder result = new StringBuilder();
        for (byte b : bytes) {
            int byteVal = b & 0xFF; // Convert to unsigned

            // GPT2 byte-to-unicode mapping
            // Printable ASCII characters map to themselves
            if (byteVal >= 33 && byteVal <= 126) {
                result.append((char) byteVal);
            }
            // Space becomes Ġ (0x0120)
            else if (byteVal == 32) {
                result.append('Ġ');
            }
            // Newline stays as-is for now (might need special handling)
            else if (byteVal == 10) {
                result.append('Ċ'); // GPT2 uses Ċ for newline
            }
            // Tab
            else if (byteVal == 9) {
                result.append('ĉ'); // GPT2 uses ĉ for tab
            }
            // Other bytes get mapped to unicode range 0x100+
            else {
                // Map to the GPT2 unicode range
                if (byteVal < 33) {
                    result.append((char) (0x100 + byteVal));
                } else if (byteVal == 127) {
                    result.append((char) 0x17F);
                } else if (byteVal > 127) {
                    result.append((char) (0x100 + byteVal));
                } else {
                    // Fallback for any special cases
                    result.append((char) (0x100 + byteVal));
                }
            }
        }
        return result.toString();
    }
}