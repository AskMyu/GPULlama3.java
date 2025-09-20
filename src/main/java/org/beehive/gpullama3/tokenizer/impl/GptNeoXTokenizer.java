package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * GPT-NeoX tokenizer implementation for OLMoE models.
 *
 * This tokenizer is based on the allenai/gpt-neox-olmo-dolma-v1_5 specification:
 * - Vocabulary size: 50,280 tokens (padded to 50,304 for efficiency)
 * - Modified GPT-NeoX BPE with PII masking tokens
 * - Special tokens: EOS=50279, PAD=1
 * - Whitespace handling optimized for code generation
 * - Trained on The Pile dataset
 *
 * Key differences from cl100k_base:
 * - Different vocabulary size and mappings
 * - Space-aware tokenization (spaces treated as part of tokens)
 * - PII masking support
 * - Optimized for diverse text sources
 */
public class GptNeoXTokenizer implements Tokenizer {

    // GPT-NeoX regex pattern - optimized for diverse text including code
    // Handles whitespace differently than cl100k_base - spaces are part of tokens
    private static final String GPTNEOX_PATTERN =
        "'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<Pair<Integer, Integer>, Integer> mergePriorities;
    private final Map<String, Integer> specialTokens;

    public GptNeoXTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        System.out.println("[GPTNEOX-TOKENIZER] Initializing GPT-NeoX tokenizer for OLMoE");
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(GPTNEOX_PATTERN);

        // Load merge rules from metadata (if available)
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges;

        if (mergeLines != null) {
            merges = Arrays.stream(mergeLines).map(line -> line.split(" "))
                    .map(parts -> new Pair<>(vocabulary.getIndex(parts[0]).orElseThrow(), vocabulary.getIndex(parts[1]).orElseThrow())).toList();
        } else {
            // OLMoE may not have explicit merges in GGUF metadata, use empty list
            System.out.println("[GPTNEOX-TOKENIZER] No tokenizer.ggml.merges found for OLMoE, using empty merges list");
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

        // Initialize special tokens based on GPT-NeoX specification
        this.specialTokens = initializeSpecialTokens();

        System.out.printf("[GPTNEOX-TOKENIZER] Initialized with vocabulary size: %d, merges: %d, special tokens: %d%n",
                         vocabulary.tokens().length, this.merges.size(), this.specialTokens.size());
    }

    private Map<String, Integer> initializeSpecialTokens() {
        Map<String, Integer> specialTokens = new HashMap<>();
        int vocabSize = vocabulary.tokens().length;

        System.out.printf("[GPTNEOX-TOKENIZER] Searching for GPT-NeoX special tokens in vocabulary of size %d%n", vocabSize);

        // Look for GPT-NeoX standard tokens and PII masking tokens
        for (int i = 0; i < vocabSize; i++) {
            String token = vocabulary.get(i);
            if (token != null) {
                switch (token) {
                    case "<|endoftext|>" -> {
                        specialTokens.put("<|endoftext|>", i);
                        System.out.printf("[GPTNEOX-TOKENIZER] Found <|endoftext|> at ID %d%n", i);
                    }
                    case "<|padding|>" -> {
                        specialTokens.put("<|padding|>", i);
                        System.out.printf("[GPTNEOX-TOKENIZER] Found <|padding|> at ID %d%n", i);
                    }
                    case "|||PHONE_NUMBER|||" -> {
                        specialTokens.put("|||PHONE_NUMBER|||", i);
                        System.out.printf("[GPTNEOX-TOKENIZER] Found PII phone token at ID %d%n", i);
                    }
                    case "|||EMAIL_ADDRESS|||" -> {
                        specialTokens.put("|||EMAIL_ADDRESS|||", i);
                        System.out.printf("[GPTNEOX-TOKENIZER] Found PII email token at ID %d%n", i);
                    }
                    case "|||IP_ADDRESS|||" -> {
                        specialTokens.put("|||IP_ADDRESS|||", i);
                        System.out.printf("[GPTNEOX-TOKENIZER] Found PII IP token at ID %d%n", i);
                    }
                }
            }
        }

        // GPT-NeoX standard special tokens based on research
        // EOS token should be at 50279 based on OLMoE specification
        if (!specialTokens.containsKey("<|endoftext|>")) {
            if (vocabSize > 50279) {
                specialTokens.put("<|endoftext|>", 50279);
                System.out.printf("[GPTNEOX-TOKENIZER] Using standard EOS token at ID 50279%n");
            } else {
                specialTokens.put("<|endoftext|>", vocabSize - 1);
                System.out.printf("[GPTNEOX-TOKENIZER] Using fallback EOS token at ID %d%n", vocabSize - 1);
            }
        }

        // PAD token should be at 1 based on OLMoE specification
        if (!specialTokens.containsKey("<|padding|>")) {
            specialTokens.put("<|padding|>", 1);
            System.out.printf("[GPTNEOX-TOKENIZER] Using standard PAD token at ID 1%n");
        }

        System.out.printf("[GPTNEOX-TOKENIZER] Loaded %d special tokens%n", specialTokens.size());
        return specialTokens;
    }

    @Override
    public String regexPattern() {
        return GPTNEOX_PATTERN;
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
        System.out.printf("[GPTNEOX-TOKENIZER] Encoding text: '%.50s%s'%n",
                         text, text.length() > 50 ? "..." : "");

        List<String> tokens = new ArrayList<>();

        // Apply regex pattern to split text (GPT-NeoX style)
        Matcher matcher = compiledPattern.matcher(text);
        while (matcher.find()) {
            tokens.add(matcher.group());
        }

        List<Integer> result = new ArrayList<>();
        for (String token : tokens) {
            // Apply GPT-NeoX BPE encoding
            List<Integer> tokenIds = encodeChunk(token);
            result.addAll(tokenIds);
        }

        System.out.printf("[GPTNEOX-TOKENIZER] Encoded %d characters to %d tokens%n",
                         text.length(), result.size());
        return result;
    }

    private List<Integer> encodeChunk(String chunk) {
        // FIXED: Proper GPT-NeoX BPE encoding starting from UTF-8 bytes
        List<Integer> ids = new ArrayList<>();

        // First, check if the entire chunk exists as a token in vocabulary
        OptionalInt wholeTokenId = vocabulary.getIndex(chunk);
        if (wholeTokenId.isPresent()) {
            ids.add(wholeTokenId.getAsInt());
            return ids;
        }

        // Convert string to UTF-8 bytes (proper BPE approach)
        try {
            byte[] bytes = chunk.getBytes("UTF-8");
            for (byte byteVal : bytes) {
                int unsignedByte = byteVal & 0xFF;
                // Map byte to token using standard byte-to-token mapping
                String byteStr = new String(new byte[]{byteVal}, "UTF-8");
                OptionalInt byteTokenId = vocabulary.getIndex(byteStr);

                if (byteTokenId.isPresent()) {
                    ids.add(byteTokenId.getAsInt());
                } else {
                    // Fallback to byte value as token ID if available
                    if (unsignedByte < vocabulary.size()) {
                        ids.add(unsignedByte);
                    } else {
                        // Ultimate fallback
                        OptionalInt unkToken = vocabulary.getIndex("<unk>");
                        ids.add(unkToken.orElse(0));
                    }
                }
            }
        } catch (Exception e) {
            // Ultimate fallback for any encoding errors
            OptionalInt unkToken = vocabulary.getIndex("<unk>");
            ids.add(unkToken.orElse(0));
            return ids;
        }

        // Apply BPE merges (GPT-NeoX style - earliest in merge list has highest priority)
        while (ids.size() >= 2) {
            Pair<Integer, Integer> bestPair = null;
            int bestPriority = Integer.MAX_VALUE;

            for (int i = 0; i < ids.size() - 1; i++) {
                Pair<Integer, Integer> pair = new Pair<>(ids.get(i), ids.get(i + 1));
                if (merges.containsKey(pair)) {
                    int priority = mergePriorities.get(pair);
                    // Lower priority number = higher priority (earlier in merge list)
                    if (priority < bestPriority) {
                        bestPair = pair;
                        bestPriority = priority;
                    }
                }
            }

            if (bestPair == null) {
                break; // No more merges possible
            }

            // Get the result token ID for this merge
            int resultTokenId = merges.get(bestPair);

            // Apply the best merge
            List<Integer> newIds = new ArrayList<>();
            int i = 0;
            while (i < ids.size()) {
                if (i < ids.size() - 1 &&
                    ids.get(i).equals(bestPair.first()) &&
                    ids.get(i + 1).equals(bestPair.second())) {
                    // CRITICAL FIX: Add result token ID, not merge priority!
                    newIds.add(resultTokenId);
                    i += 2;
                } else {
                    newIds.add(ids.get(i));
                    i++;
                }
            }
            ids = newIds;
        }

        return ids;
    }

    @Override
    public String decode(List<Integer> tokens) {
        System.out.printf("[GPTNEOX-TOKENIZER] Decoding %d tokens%n", tokens.size());

        StringBuilder result = new StringBuilder();
        for (Integer tokenId : tokens) {
            String tokenText = vocabulary.get(tokenId);
            if (tokenText != null) {
                // GPT-2/GPT-NeoX uses special byte encoding:
                // Printable ASCII stays the same
                // Spaces become 'Ġ' (U+0120)
                // Other bytes map to unicode range U+0100-U+01FF
                String decoded = decodeGPT2Bytes(tokenText);
                result.append(decoded);
            }
        }

        String decoded = result.toString();
        System.out.printf("[GPTNEOX-TOKENIZER] Decoded to %d characters%n", decoded.length());
        return decoded;
    }

    private String decodeGPT2Bytes(String text) {
        StringBuilder decoded = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            // Handle GPT-2 byte encoding
            if (c == 'Ġ') {
                // 'Ġ' (U+0120) represents space
                decoded.append(' ');
            } else if (c >= 0x100 && c <= 0x1FF) {
                // Characters in range U+0100-U+01FF map back to bytes
                // This is the GPT-2 encoding for non-printable/special bytes
                int byteValue = c - 0x100;
                decoded.append((char) byteValue);
            } else {
                // Regular characters stay as-is
                decoded.append(c);
            }
        }
        return decoded.toString();
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return new HashMap<>(specialTokens);
    }
}