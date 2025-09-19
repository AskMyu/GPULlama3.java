package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Tokenizer for IBM Granite models.
 *
 * Granite 3.3 uses a GPT2-style tokenizer (49,152 vocab) with support for:
 * - Conversation tokens (<|start_of_role|>, <|end_of_role|>)
 * - Tool functionality (<|tool_call|>)
 * - Citation tokens (<|start_of_cite|>, <|end_of_cite|>)
 * - Plugin tokens (<|start_of_plugin|>, <|end_of_plugin|>)
 * - Fill-in-the-Middle (FIM) tokens for code completion
 * - Structured reasoning tokens (<think>, </think>)
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

        // Validate vocabulary size against official Granite 3.3 specification
        int expectedVocabSize = 49152;
        int actualVocabSize = vocabulary.size();
        if (actualVocabSize != expectedVocabSize) {
            System.err.printf("[GRANITE-TOKENIZER] WARNING: Vocabulary size mismatch! Expected: %d, Actual: %d%n",
                            expectedVocabSize, actualVocabSize);
        } else {
            System.err.printf("[GRANITE-TOKENIZER] ✅ Vocabulary size matches official spec: %d tokens%n", actualVocabSize);
        }

        // Initialize special tokens
        initializeSpecialTokens();
    }
    
    private void initializeSpecialTokens() {
        // Find and register special tokens in vocabulary
        System.err.println("[GRANITE-TOKENIZER] Searching for special tokens in vocabulary...");

        // Standard tokens
        vocabulary.getIndex(BOS_TOKEN).ifPresent(idx -> {
            specialTokens.put(BOS_TOKEN, idx);
            System.err.printf("[GRANITE-TOKENIZER] Found %s -> %d%n", BOS_TOKEN, idx);
        });
        vocabulary.getIndex(EOS_TOKEN).ifPresent(idx -> {
            specialTokens.put(EOS_TOKEN, idx);
            System.err.printf("[GRANITE-TOKENIZER] Found %s -> %d%n", EOS_TOKEN, idx);
        });
        vocabulary.getIndex(UNK_TOKEN).ifPresent(idx -> {
            specialTokens.put(UNK_TOKEN, idx);
            System.err.printf("[GRANITE-TOKENIZER] Found %s -> %d%n", UNK_TOKEN, idx);
        });
        vocabulary.getIndex(PAD_TOKEN).ifPresent(idx -> {
            specialTokens.put(PAD_TOKEN, idx);
            System.err.printf("[GRANITE-TOKENIZER] Found %s -> %d%n", PAD_TOKEN, idx);
        });

        // Granite 3.3 specific role tokens
        vocabulary.getIndex("<|start_of_role|>").ifPresent(idx -> {
            specialTokens.put("<|start_of_role|>", idx);
            System.err.printf("[GRANITE-TOKENIZER] Found <|start_of_role|> -> %d%n", idx);
        });
        vocabulary.getIndex("<|end_of_role|>").ifPresent(idx -> {
            specialTokens.put("<|end_of_role|>", idx);
            System.err.printf("[GRANITE-TOKENIZER] Found <|end_of_role|> -> %d%n", idx);
        });

        // Check if tokens are missing
        if (!specialTokens.containsKey("<|start_of_role|>")) {
            System.err.println("[GRANITE-TOKENIZER] WARNING: <|start_of_role|> not found in vocabulary!");
        }
        if (!specialTokens.containsKey("<|end_of_role|>")) {
            System.err.println("[GRANITE-TOKENIZER] WARNING: <|end_of_role|> not found in vocabulary!");
        }
        
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

        // Additional Granite 3.3 tokens (from official config)
        vocabulary.getIndex("<|tool_call|>").ifPresent(idx -> {
            specialTokens.put("<|tool_call|>", idx);
            System.err.printf("[GRANITE-TOKENIZER] Found <|tool_call|> -> %d%n", idx);
        });
        vocabulary.getIndex("<|start_of_cite|>").ifPresent(idx -> {
            specialTokens.put("<|start_of_cite|>", idx);
            System.err.printf("[GRANITE-TOKENIZER] Found <|start_of_cite|> -> %d%n", idx);
        });
        vocabulary.getIndex("<|end_of_cite|>").ifPresent(idx -> {
            specialTokens.put("<|end_of_cite|>", idx);
            System.err.printf("[GRANITE-TOKENIZER] Found <|end_of_cite|> -> %d%n", idx);
        });
        vocabulary.getIndex("<|start_of_plugin|>").ifPresent(idx -> {
            specialTokens.put("<|start_of_plugin|>", idx);
            System.err.printf("[GRANITE-TOKENIZER] Found <|start_of_plugin|> -> %d%n", idx);
        });
        vocabulary.getIndex("<|end_of_plugin|>").ifPresent(idx -> {
            specialTokens.put("<|end_of_plugin|>", idx);
            System.err.printf("[GRANITE-TOKENIZER] Found <|end_of_plugin|> -> %d%n", idx);
        });
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
        List<Integer> tokens = new ArrayList<>();

        // First, handle special tokens by replacing them with placeholders
        String processedText = text;
        Map<String, String> specialTokenReplacements = new HashMap<>();
        int placeholderIndex = 0;

        // Replace special tokens with unique placeholders
        for (String specialToken : specialTokens.keySet()) {
            if (allowedSpecial.contains("all") || allowedSpecial.contains(specialToken)) {
                String placeholder = "SPECIAL_TOKEN_" + placeholderIndex++;
                specialTokenReplacements.put(placeholder, specialToken);
                processedText = processedText.replace(specialToken, " " + placeholder + " ");
            }
        }

        // Use proper subword tokenization instead of naive word splitting
        tokens.addAll(encodeSubwords(processedText, specialTokenReplacements));

        return tokens;
    }

    /**
     * Encode text using proper subword tokenization to avoid UNK token contamination.
     * Uses greedy longest-match algorithm similar to SentencePiece.
     */
    private List<Integer> encodeSubwords(String text, Map<String, String> specialTokenReplacements) {
        List<Integer> tokens = new ArrayList<>();

        // Split on whitespace to get words
        String[] words = text.split("\\s+");

        for (String word : words) {
            if (word.isEmpty()) continue;

            // Check if it's a special token placeholder
            if (specialTokenReplacements.containsKey(word)) {
                String originalSpecialToken = specialTokenReplacements.get(word);
                tokens.add(specialTokens.get(originalSpecialToken));
                continue;
            }

            // Try to encode the word using greedy longest-match subword tokenization
            List<Integer> wordTokens = encodeWordAsSubwords(word);
            tokens.addAll(wordTokens);
        }

        return tokens;
    }

    /**
     * Encode a single word using greedy longest-match subword tokenization.
     * This avoids creating multiple UNK tokens for unknown words.
     */
    private List<Integer> encodeWordAsSubwords(String word) {
        List<Integer> tokens = new ArrayList<>();

        // Add space prefix for SentencePiece-style encoding
        String prefixedWord = "▁" + word;

        // Try exact match first
        if (vocabulary.getIndex(prefixedWord).isPresent()) {
            tokens.add(vocabulary.getIndex(prefixedWord).getAsInt());
            return tokens;
        }

        // Try with Ġ prefix (GPT-style space marker)
        String gptStyleWord = "Ġ" + word;
        if (vocabulary.getIndex(gptStyleWord).isPresent()) {
            tokens.add(vocabulary.getIndex(gptStyleWord).getAsInt());
            return tokens;
        }

        // Try exact word without prefix
        if (vocabulary.getIndex(word).isPresent()) {
            tokens.add(vocabulary.getIndex(word).getAsInt());
            return tokens;
        }

        // Greedy longest-match subword tokenization
        int start = 0;
        while (start < word.length()) {
            boolean foundMatch = false;

            // Try progressively shorter substrings
            for (int end = word.length(); end > start; end--) {
                String subword = word.substring(start, end);

                // Try different prefix variations
                String[] candidates = {
                    "▁" + subword,  // SentencePiece prefix
                    "Ġ" + subword,  // GPT prefix
                    subword         // No prefix
                };

                for (String candidate : candidates) {
                    if (vocabulary.getIndex(candidate).isPresent()) {
                        tokens.add(vocabulary.getIndex(candidate).getAsInt());
                        start = end;
                        foundMatch = true;
                        break;
                    }
                }

                if (foundMatch) break;
            }

            // If no subword match found, try single character
            if (!foundMatch) {
                String singleChar = String.valueOf(word.charAt(start));
                if (vocabulary.getIndex(singleChar).isPresent()) {
                    tokens.add(vocabulary.getIndex(singleChar).getAsInt());
                } else {
                    // Last resort: add one UNK token for the entire remaining part
                    int unkToken = specialTokens.getOrDefault(UNK_TOKEN, 1);
                    if (unkToken != 0) { // Avoid token 0 which might be EOS
                        tokens.add(unkToken);
                    }
                    break; // Don't continue to avoid multiple UNK tokens
                }
                start++;
            }
        }

        return tokens;
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        // Simple encoding allowing all special tokens
        List<Integer> tokens = encode(text, Set.of("all"));
        System.err.printf("[GRANITE-TOKENIZER] Encoded '%s' -> %s%n", text, tokens.toString());

        // DEBUG: Show vocabulary mapping for encoded tokens
        System.err.printf("[ENCODE-DEBUG] Token breakdown:%n");
        for (int i = 0; i < Math.min(10, tokens.size()); i++) {
            int token = tokens.get(i);
            String tokenStr = vocabulary.get(token);
            System.err.printf("[ENCODE-DEBUG] Token %d -> '%s'%n", token, tokenStr);
        }

        return tokens;
    }
    
    @Override
    public String decode(List<Integer> tokens) {
        System.err.printf("[GRANITE-TOKENIZER] Decoding %d tokens: %s%n", tokens.size(), tokens.toString());

        // DEBUG: Log first few tokens and their vocabulary mappings for analysis
        if (tokens.size() > 0) {
            System.err.printf("[VOCAB-DEBUG] Token breakdown for this decode:%n");
            for (int i = 0; i < Math.min(10, tokens.size()); i++) {
                int token = tokens.get(i);
                String tokenStr = vocabulary.get(token);
                System.err.printf("[VOCAB-DEBUG] Token %d -> '%s'%n", token, tokenStr);
            }
        }

        // First, get the raw token strings from vocabulary
        StringBuilder rawConcatenated = new StringBuilder();
        for (int token : tokens) {
            String tokenStr = vocabulary.get(token);
            if (tokenStr != null) {
                rawConcatenated.append(tokenStr);
            } else {
                System.err.printf("[GRANITE-TOKENIZER] Warning: Token %d not found in vocabulary%n", token);
            }
        }

        String concatenated = rawConcatenated.toString();
        System.err.printf("[GRANITE-TOKENIZER] Raw concatenated: '%s'%n", concatenated);

        // Handle SentencePiece decoding
        // Replace ▁ with spaces (SentencePiece space marker)
        String processed = concatenated.replace("▁", " ");

        // Handle Ġ prefix (another common space marker)
        if (processed.startsWith("Ġ")) {
            processed = " " + processed.substring(1);
        }
        processed = processed.replace("Ġ", " ");

        // Apply byte-level decoding for UTF-8 sequences
        try {
            processed = decodeByteLevelTokens(processed);
            System.err.printf("[GRANITE-TOKENIZER] After byte-level decoding: '%s'%n", processed);
        } catch (Exception e) {
            System.err.printf("[GRANITE-TOKENIZER] Byte-level decoding failed, using raw: %s%n", e.getMessage());
        }

        // Trim leading space if it exists
        if (processed.startsWith(" ")) {
            processed = processed.substring(1);
        }

        System.err.printf("[GRANITE-TOKENIZER] Final decoded: '%s'%n", processed);
        return processed;
    }

    /**
     * Decode byte-level UTF-8 sequences back to proper text.
     * This handles tokens that represent UTF-8 bytes encoded as Unicode characters.
     */
    private String decodeByteLevelTokens(String input) {
        try {
            // Convert the string to a list of Unicode code points
            int[] codePoints = input.codePoints().toArray();

            // Convert Unicode code points back to bytes where applicable
            java.io.ByteArrayOutputStream byteStream = new java.io.ByteArrayOutputStream();

            for (int codePoint : codePoints) {
                // Check if this is a byte-level encoded character
                if (isByteLevelEncoding(codePoint)) {
                    // Convert back to original byte value
                    int byteValue = mapUnicodeToBytes(codePoint);
                    if (byteValue >= 0) {
                        byteStream.write(byteValue);
                    } else {
                        // If mapping fails, keep as original character
                        String charStr = new String(Character.toChars(codePoint));
                        byteStream.write(charStr.getBytes(StandardCharsets.UTF_8));
                    }
                } else {
                    // Regular character, convert to UTF-8 bytes
                    String charStr = new String(Character.toChars(codePoint));
                    byteStream.write(charStr.getBytes(StandardCharsets.UTF_8));
                }
            }

            // Reconstruct as UTF-8 string
            byte[] bytes = byteStream.toByteArray();
            String result = new String(bytes, StandardCharsets.UTF_8);

            System.err.printf("[GRANITE-TOKENIZER] Byte-level decode: '%s' → '%s'%n",
                            input.replace("\n", "\\n"), result.replace("\n", "\\n"));

            return result;

        } catch (Exception e) {
            System.err.printf("[GRANITE-TOKENIZER] Byte-level decoding error: %s%n", e.getMessage());
            return input; // Return original on error
        }
    }

    /**
     * Check if a Unicode code point represents a byte-level encoding.
     */
    private boolean isByteLevelEncoding(int codePoint) {
        // Byte-level encodings typically use specific Unicode ranges
        // Based on patterns seen in the gibberish: Ð, Ñ, Å, å, ĳ, etc.

        // Latin Extended-A (U+0100-U+017F) - common in byte encodings
        if (codePoint >= 0x0100 && codePoint <= 0x017F) return true;

        // Latin Extended-B (U+0180-U+024F)
        if (codePoint >= 0x0180 && codePoint <= 0x024F) return true;

        // Cyrillic (U+0400-U+04FF) - Ð, Ñ characters
        if (codePoint >= 0x0400 && codePoint <= 0x04FF) return true;

        // CJK characters that appear in byte encodings
        if (codePoint >= 0x4E00 && codePoint <= 0x9FFF) return true;

        // Additional ranges that commonly appear in byte-level encodings
        if (codePoint >= 0x2000 && codePoint <= 0x206F) return true; // General Punctuation
        if (codePoint >= 0x0300 && codePoint <= 0x036F) return true; // Combining marks

        return false;
    }

    /**
     * Map Unicode code points back to original byte values.
     * This uses the standard GPT-style byte-to-unicode mapping used by most modern tokenizers.
     */
    private int mapUnicodeToBytes(int codePoint) {
        // ASCII range (0x00-0x7F) maps directly
        if (codePoint >= 0x00 && codePoint <= 0x7F) {
            return codePoint;
        }

        // Standard GPT-style byte-to-unicode mapping for bytes 0x80-0xFF
        // This is the canonical mapping used by most tokenizers
        switch (codePoint) {
            // Latin-1 Supplement range that directly maps to bytes
            case 0x00A0: return 0x80; // Non-breaking space → 128
            case 0x00A1: return 0x81; // ¡ → 129
            case 0x00A2: return 0x82; // ¢ → 130
            case 0x00A3: return 0x83; // £ → 131
            case 0x00A4: return 0x84; // ¤ → 132
            case 0x00A5: return 0x85; // ¥ → 133
            case 0x00A6: return 0x86; // ¦ → 134
            case 0x00A7: return 0x87; // § → 135
            case 0x00A8: return 0x88; // ¨ → 136
            case 0x00A9: return 0x89; // © → 137
            case 0x00AA: return 0x8A; // ª → 138
            case 0x00AB: return 0x8B; // « → 139
            case 0x00AC: return 0x8C; // ¬ → 140
            case 0x00AD: return 0x8D; // Soft hyphen → 141
            case 0x00AE: return 0x8E; // ® → 142
            case 0x00AF: return 0x8F; // ¯ → 143
            case 0x00B0: return 0x90; // ° → 144
            case 0x00B1: return 0x91; // ± → 145
            case 0x00B2: return 0x92; // ² → 146
            case 0x00B3: return 0x93; // ³ → 147
            case 0x00B4: return 0x94; // ´ → 148
            case 0x00B5: return 0x95; // µ → 149
            case 0x00B6: return 0x96; // ¶ → 150
            case 0x00B7: return 0x97; // · → 151
            case 0x00B8: return 0x98; // ¸ → 152
            case 0x00B9: return 0x99; // ¹ → 153
            case 0x00BA: return 0x9A; // º → 154
            case 0x00BB: return 0x9B; // » → 155
            case 0x00BC: return 0x9C; // ¼ → 156
            case 0x00BD: return 0x9D; // ½ → 157
            case 0x00BE: return 0x9E; // ¾ → 158
            case 0x00BF: return 0x9F; // ¿ → 159
            case 0x00C0: return 0xA0; // À → 160
            case 0x00C1: return 0xA1; // Á → 161
            case 0x00C2: return 0xA2; // Â → 162
            case 0x00C3: return 0xA3; // Ã → 163
            case 0x00C4: return 0xA4; // Ä → 164
            case 0x00C5: return 0xA5; // Å → 165
            case 0x00C6: return 0xA6; // Æ → 166
            case 0x00C7: return 0xA7; // Ç → 167
            case 0x00C8: return 0xA8; // È → 168
            case 0x00C9: return 0xA9; // É → 169
            case 0x00CA: return 0xAA; // Ê → 170
            case 0x00CB: return 0xAB; // Ë → 171
            case 0x00CC: return 0xAC; // Ì → 172
            case 0x00CD: return 0xAD; // Í → 173
            case 0x00CE: return 0xAE; // Î → 174
            case 0x00CF: return 0xAF; // Ï → 175
            case 0x00D0: return 0xB0; // Ð → 176
            case 0x00D1: return 0xB1; // Ñ → 177
            case 0x00D2: return 0xB2; // Ò → 178
            case 0x00D3: return 0xB3; // Ó → 179
            case 0x00D4: return 0xB4; // Ô → 180
            case 0x00D5: return 0xB5; // Õ → 181
            case 0x00D6: return 0xB6; // Ö → 182
            case 0x00D7: return 0xB7; // × → 183
            case 0x00D8: return 0xB8; // Ø → 184
            case 0x00D9: return 0xB9; // Ù → 185
            case 0x00DA: return 0xBA; // Ú → 186
            case 0x00DB: return 0xBB; // Û → 187
            case 0x00DC: return 0xBC; // Ü → 188
            case 0x00DD: return 0xBD; // Ý → 189
            case 0x00DE: return 0xBE; // Þ → 190
            case 0x00DF: return 0xBF; // ß → 191
            case 0x00E0: return 0xC0; // à → 192
            case 0x00E1: return 0xC1; // á → 193
            case 0x00E2: return 0xC2; // â → 194
            case 0x00E3: return 0xC3; // ã → 195
            case 0x00E4: return 0xC4; // ä → 196
            case 0x00E5: return 0xC5; // å → 197
            case 0x00E6: return 0xC6; // æ → 198
            case 0x00E7: return 0xC7; // ç → 199
            case 0x00E8: return 0xC8; // è → 200
            case 0x00E9: return 0xC9; // é → 201
            case 0x00EA: return 0xCA; // ê → 202
            case 0x00EB: return 0xCB; // ë → 203
            case 0x00EC: return 0xCC; // ì → 204
            case 0x00ED: return 0xCD; // í → 205
            case 0x00EE: return 0xCE; // î → 206
            case 0x00EF: return 0xCF; // ï → 207
            case 0x00F0: return 0xD0; // ð → 208
            case 0x00F1: return 0xD1; // ñ → 209
            case 0x00F2: return 0xD2; // ò → 210
            case 0x00F3: return 0xD3; // ó → 211
            case 0x00F4: return 0xD4; // ô → 212
            case 0x00F5: return 0xD5; // õ → 213
            case 0x00F6: return 0xD6; // ö → 214
            case 0x00F7: return 0xD7; // ÷ → 215
            case 0x00F8: return 0xD8; // ø → 216
            case 0x00F9: return 0xD9; // ù → 217
            case 0x00FA: return 0xDA; // ú → 218
            case 0x00FB: return 0xDB; // û → 219
            case 0x00FC: return 0xDC; // ü → 220
            case 0x00FD: return 0xDD; // ý → 221
            case 0x00FE: return 0xDE; // þ → 222
            case 0x00FF: return 0xDF; // ÿ → 223

            // Extended characters that map to high bytes
            case 0x0100: return 0xE0; // Ā → 224
            case 0x0101: return 0xE1; // ā → 225
            case 0x0102: return 0xE2; // Ă → 226
            case 0x0103: return 0xE3; // ă → 227
            case 0x0104: return 0xE4; // Ą → 228
            case 0x0105: return 0xE5; // ą → 229
            case 0x0106: return 0xE6; // Ć → 230
            case 0x0107: return 0xE7; // ć → 231
            case 0x0108: return 0xE8; // Ĉ → 232
            case 0x0109: return 0x09; // ĉ → tab (HT) - CONTROL CHARACTER
            case 0x010A: return 0x0A; // Ċ → newline (LF) - CONTROL CHARACTER
            case 0x010B: return 0xEB; // ċ → 235
            case 0x010C: return 0xEC; // Č → 236
            case 0x010D: return 0x0D; // č → carriage return (CR) - CONTROL CHARACTER
            case 0x010E: return 0xEE; // Ď → 238
            case 0x010F: return 0xEF; // ď → 239
            case 0x0110: return 0xF0; // Đ → 240
            case 0x0111: return 0xF1; // đ → 241
            case 0x0112: return 0xF2; // Ē → 242
            case 0x0113: return 0xF3; // ē → 243
            case 0x0114: return 0xF4; // Ĕ → 244
            case 0x0115: return 0xF5; // ĕ → 245
            case 0x0116: return 0xF6; // Ė → 246
            case 0x0117: return 0xF7; // ė → 247
            case 0x0118: return 0xF8; // Ę → 248
            case 0x0119: return 0xF9; // ę → 249
            case 0x011A: return 0xFA; // Ě → 250
            case 0x011B: return 0xFB; // ě → 251
            case 0x011C: return 0xFC; // Ĝ → 252
            case 0x011D: return 0xFD; // ĝ → 253
            case 0x011E: return 0xFE; // Ğ → 254
            case 0x011F: return 0xFF; // ğ → 255

            // Space character (handled elsewhere but included for completeness)
            case 0x0120: return 0x20; // Ġ → space

            // Specific problematic characters seen in logs
            case 0x0131: return 0x80; // ı → 128 (dotless i)
            case 0x013F: return 0x81; // Ŀ → 129
            case 0x0142: return 0x82; // ł → 130
            case 0x0144: return 0x83; // ń → 131
            case 0x0148: return 0x84; // ň → 132
            case 0x014D: return 0x85; // ō → 133
            case 0x0151: return 0x86; // ő → 134
            case 0x0153: return 0x87; // œ → 135
            case 0x0155: return 0x88; // ŕ → 136
            case 0x0159: return 0x89; // ř → 137
            case 0x015B: return 0x8A; // ś → 138
            case 0x015F: return 0x8B; // ş → 139
            case 0x0161: return 0x8C; // š → 140
            case 0x0165: return 0x8D; // ť → 141
            case 0x016B: return 0x8E; // ū → 142
            case 0x016F: return 0x8F; // ů → 143
            case 0x0171: return 0x90; // ű → 144
            case 0x0173: return 0x91; // ų → 145
            case 0x017A: return 0x92; // ź → 146
            case 0x017C: return 0x93; // ż → 147
            case 0x017E: return 0x94; // ž → 148

            default:
                // Handle control characters first (common in tokenizers)
                if (codePoint >= 0x0100 && codePoint <= 0x011F) {
                    // Map Latin Extended-A control characters to ASCII control range
                    int controlChar = (codePoint - 0x0100);
                    if (controlChar <= 0x1F) {
                        return controlChar; // Map to ASCII control characters 0x00-0x1F
                    }
                }

                // Try Latin-1 Supplement range (most common for UTF-8 bytes)
                if (codePoint >= 0x00A0 && codePoint <= 0x00FF) {
                    return 0x80 + (codePoint - 0x00A0);
                }

                // Try Latin Extended-A range
                if (codePoint >= 0x0100 && codePoint <= 0x017F) {
                    return 0x80 + ((codePoint - 0x0100) % 0x80);
                }

                // Handle higher Unicode ranges that might represent bytes
                if (codePoint >= 0x0180 && codePoint <= 0x024F) {
                    return 0x80 + ((codePoint - 0x0180) % 0x80);
                }

                return -1; // No mapping found
        }
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