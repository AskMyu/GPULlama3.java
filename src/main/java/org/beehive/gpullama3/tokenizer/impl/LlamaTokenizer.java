package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * GPT-2-style BPE tokenizer (even though it's called "llama") with an explicit merges list.
 * <p>
 * BPE (Byte Pair Encoding):
 * A sub-word tokenization algorithm that iteratively merges the most frequent pairs of symbols in a corpus to build a vocabulary of common character sequences.
 * <p>
 * GPT-2-style tokenization:
 * Applies BPE at the byte level, ensuring all UTF-8 inputs are representable and using tokens that preserve leading spaces (e.g., 'Ġthe').
 * <p>
 * Explicit merges list:
 * A fixed sequence of learned merge rules that deterministically reconstructs the tokenizer’s vocabulary during inference without retraining.
 * <p>
 * Based on <a href="https://github.com/karpathy/minbpe">minbpe</a>, algorithmically follows along the
 * <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2 tokenizer</a>
 */
public class LlamaTokenizer implements Tokenizer {
    private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    // general fields
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    // model-specific fields
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
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

    public LlamaTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        // load from metadata
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges;
        
        if (mergeLines != null) {
            merges = Arrays.stream(mergeLines).map(line -> line.split(" "))
                    .map(parts -> new Pair<>(vocabulary.getIndex(parts[0]).orElseThrow(), vocabulary.getIndex(parts[1]).orElseThrow())).toList();
        } else {
            // Some models (like LLaVA-1.5) don't have merges in metadata
            System.out.println("DEBUG: No tokenizer.ggml.merges found, using empty merges list");
            merges = new ArrayList<>();
        }
        int allTokens = vocabulary.size();
        int baseTokens = Math.min(128000, allTokens); // assume all tokens after the base ones are special, but don't exceed actual vocab size
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = (reservedSpecialTokens > 0) ? 
            Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList() :
            new ArrayList<>(); // No special tokens if baseTokens >= allTokens
        
        System.out.println("DEBUG: Vocabulary info - allTokens: " + allTokens + ", baseTokens: " + baseTokens + ", specialTokens: " + reservedSpecialTokens);

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens = IntStream.range(0, specialTokensList.size()).boxed().collect(Collectors.toMap(i -> specialTokensList.get(i), i -> baseTokens + i));

        // init tokenizer object fields
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(LLAMA_3_PATTERN);
        this.specialTokens = new HashMap<>(specialTokens);
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    private int[] encodeImpl(String text) {
        return encode(text, Set.of()).stream().mapToInt(i -> i).toArray();
    }

    /**
     * Unlike {@link #encodeOrdinary(String)}, this function handles special tokens.
     * allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
     * if none_raise, then an error is raised if any special token is encountered in text
     * this is the default tiktoken behavior right now as well
     * any other behavior is either annoying, or a major footgun.
     */
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
        // we can use re.split for this. note that surrounding the pattern with ()
        // makes it into a capturing group, so the special tokens will be included
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

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    /**
     * Encoding that ignores any special tokens.
     */
    public List<Integer> encodeOrdinary(String text) {
        // split text into chunks of text by categories defined in regex pattern
        List<String> textChunks = findAll(compiledPattern, text);
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        // return the token ids
        // let's begin. first, convert all bytes to integers in range 0..255
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            String charStr = String.valueOf((char) b);
            var tokenIndex = this.vocabulary.getIndex(charStr);
            if (tokenIndex.isPresent()) {
                ids.add(tokenIndex.getAsInt());
            } else {
                // Handle unknown character: try fallback strategies
                // Strategy 1: Use UTF-8 byte encoding
                try {
                    byte[] bytes = charStr.getBytes("UTF-8");
                    for (byte byteVal : bytes) {
                        int unsignedByte = byteVal & 0xFF;
                        var byteTokenIndex = this.vocabulary.getIndex(String.valueOf(unsignedByte));
                        if (byteTokenIndex.isPresent()) {
                            ids.add(byteTokenIndex.getAsInt());
                        } else {
                            // Strategy 2: Use unknown token (ID 0) or space token as fallback
                            var spaceToken = this.vocabulary.getIndex(" ");
                            ids.add(spaceToken.orElse(0)); // Use 0 as ultimate fallback
                        }
                    }
                } catch (Exception e) {
                    // Ultimate fallback: use unknown token
                    ids.add(0);
                }
            }
        }

        while (ids.size() >= 2) {
            // find the pair with the lowest merge index
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair = stats.keySet().stream().min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE))).orElseThrow();
            // subtle: if there are no more merges available, the key will
            // result in an inf for every single pair, and the min will be
            // just the first pair in the list, arbitrarily
            // we can detect this terminating case by a membership check
            if (!this.merges.containsKey(pair)) {
                break; // nothing else can be merged anymore
            }
            // otherwise let's merge the best pair (lowest merge index)
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            // if not at the very last position AND the pair matches, replace it
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            if (tokenString == null) {
                // Handle out-of-vocabulary tokens
                System.err.printf("[TOKENIZER] Warning: Token %d not found in vocabulary (vocab size: %d), using <UNK>%n", 
                                 token, vocabulary.size());
                tokenString = "<UNK>"; // Use unknown token placeholder
            } else {
                System.err.printf("[TOKENIZER-DEBUG] Token %d -> '%s' (length=%d)%n", 
                                 token, tokenString.replace("\n", "\\n").replace("\r", "\\r"), tokenString.length());
            }
            sb.append(tokenString);
        }
        return sb.toString();
    }

    /**
     * Returns list of utf-8 byte and a corresponding list of unicode strings.
     * The reversible bpe codes work on unicode strings.
     * This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
     * When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
     * This is a significant percentage of your normal, say, 32K bpe vocab.
     * To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
     * And avoids mapping to whitespace/control characters the bpe code barfs on.
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        // return dict(zip(bs, cs))
        return IntStream.range(0, bs.size()).boxed().collect(Collectors.toMap(bs::get, cs::get));
    }

    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet().stream().collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    public int[] encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeImpl(sb.toString());
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return Arrays.stream(encodeImpl(sb.toString())).boxed().toList();
    }

    @Override
    public String decode(List<Integer> tokens) {
        System.err.printf("[TOKENIZER-DECODE] Decoding %d tokens: %s%n", tokens.size(), tokens.toString());
        String decoded = decodeImpl(tokens);
        System.err.printf("[TOKENIZER-DECODE] Intermediate decoded string: '%s' (length=%d)%n", 
                         decoded.replace("\n", "\\n").replace("\r", "\\r"), decoded.length());
        
        int[] decodedBytesAsInts = decoded.codePoints()
            .map(codePoint -> {
                Integer byteValue = BYTE_DECODER.get(codePoint);
                if (byteValue == null) {
                    System.err.printf("[TOKENIZER] Warning: Unknown code point %d, using replacement character%n", codePoint);
                    return 0xFFFD; // Unicode replacement character
                }
                return byteValue;
            })
            .toArray();
        
        System.err.printf("[TOKENIZER-DECODE] Decoded bytes array length: %d%n", decodedBytesAsInts.length);
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < decodedBytesAsInts.length; i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        String finalResult = new String(rawBytes, StandardCharsets.UTF_8);
        System.err.printf("[TOKENIZER-DECODE] Final result: '%s' (length=%d)%n", 
                         finalResult.replace("\n", "\\n").replace("\r", "\\r"), finalResult.length());
        return finalResult;
    }
}
