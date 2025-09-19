package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.Qwen3Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utility tailored for the Chat Markup Language (ChatML) prompt format.
 */
public class Qwen3ChatFormat implements ChatFormat {

    protected final int beginOfText;
    protected final int startHeader;
    protected final int endHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;
    protected final int endOfTextFim;
    protected final int imStart; // beginOfText
    protected final int imEnd; // endOfText
    protected final int fimPrefix;
    protected final int fimSuffix;
    protected final int fimMiddle;
    protected Qwen3Tokenizer tokenizer;
    protected ChatTokens chatTokens;

    public Qwen3ChatFormat(Qwen3Tokenizer tokenizer, ChatTokens chatTokens) {
        this.tokenizer = tokenizer;
        this.chatTokens = chatTokens;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.getOrDefault("", -1);
        this.startHeader = specialTokens.getOrDefault(chatTokens.tStartHeader(), -1);
        this.endHeader = specialTokens.getOrDefault(chatTokens.tEndHeader(), -1);
        this.endOfTurn = specialTokens.getOrDefault(chatTokens.tEndOfTurn(), -1);
        this.endOfText = specialTokens.getOrDefault(chatTokens.tEndOfText(), -1);
        this.endOfTextFim = specialTokens.getOrDefault(chatTokens.tEndOfTextFim(), -1);
        this.endOfMessage = specialTokens.getOrDefault("", -1); // Use default value if key not found

        this.imStart = startHeader;
        this.imEnd = endHeader;

        fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1);
        fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1);
        fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1);
    }

    public ChatTokens chatTokens() {
        return chatTokens;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        if (endHeader == -1) {
            // DeepSeek-R1
            String sToken = switch (message.role().name()) {
                case "system" -> null;
                case "user" -> "<｜User｜>";
                case "assistant" -> "<｜Assistant｜>";
                case "fim_prefix" -> "<|fim_prefix|>";
                case "fim_middle" -> "<|fim_middle|>";
                case "fim_suffix" -> "<|fim_suffix|>";
                default -> null;
            };
            if (sToken != null) {
                Integer token = tokenizer.getSpecialTokens().get(sToken);
                if (token != null) {
                    tokens.add(token);
                } else {
                    // Fallback: encode as regular text if special token not found
                    tokens.addAll(this.tokenizer.encodeAsList(sToken));
                }
            }
        } else if (Role.FIM_PREFIX.equals(message.role())) {
            // fill-in-the-middle, token fim_prefix.
            if (fimPrefix != -1) {
                tokens.add(fimPrefix);
            }
        } else if (Role.FIM_SUFFIX.equals(message.role())) {
            if (fimSuffix != -1) {
                tokens.add(fimSuffix);
            }
        } else if (Role.FIM_MIDDLE.equals(message.role())) {
            if (fimMiddle != -1) {
                tokens.add(fimMiddle);
            }
        } else {
            if (imStart != -1) {
                tokens.add(imStart);
            }
            tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
            tokens.addAll(this.tokenizer.encodeAsList("\n"));
        }
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        boolean isFim = Role.FIM_PREFIX.equals(message.role()) || Role.FIM_SUFFIX.equals(message.role()) || Role.FIM_MIDDLE.equals(message.role());
        if (imEnd != -1 && !isFim) {
            tokens.add(imEnd);
        }
        return tokens;
    }

    @Override
    public int getBeginOfText() {
        // For DeepSeek-R1, if beginOfText is not found, return a safe default
        if (beginOfText == -1) {
            // Try to find a suitable BOS token
            Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
            int bosToken = specialTokens.getOrDefault("<|im_start|>", -1);
            if (bosToken != -1) {
                return bosToken;
            }

            // Fallback to endOfText if available
            if (endOfText != -1) {
                return endOfText;
            }

            // Last resort: return 1 (safer than 0 which might be EOS)
            System.err.println("[QWEN3-CHAT-FORMAT] Warning: No BOS token found, using token 1");
            return 1;
        }
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        // For DeepSeek-R1, be more flexible with stop tokens
        Set<Integer> stopTokens = new java.util.HashSet<>();

        if (imEnd != -1) {
            stopTokens.add(imEnd);
        }
        if (endOfText != -1) {
            stopTokens.add(endOfText);
        }
        if (endOfTextFim != -1) {
            stopTokens.add(endOfTextFim);
        }

        // If no standard stop tokens found, try to find common alternatives
        if (stopTokens.isEmpty()) {
            Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

            // Try common end-of-sequence tokens
            int eosToken = specialTokens.getOrDefault("<|endoftext|>", -1);
            if (eosToken != -1) {
                stopTokens.add(eosToken);
            }

            // Try alternative end tokens
            int altEnd = specialTokens.getOrDefault("<｜end▁of▁sentence｜>", -1);
            if (altEnd != -1) {
                stopTokens.add(altEnd);
            }

            // Last resort: use a reasonable default (EOS token ID is often 2)
            if (stopTokens.isEmpty()) {
                System.err.println("[QWEN3-CHAT-FORMAT] Warning: No stop tokens found, using default EOS token 2");
                stopTokens.add(2);
            }
        }

        return stopTokens;
    }
}
