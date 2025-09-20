package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.Cl100kTokenizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chat format for cl100k_base tokenizer (used by OLMoE models).
 * Based on LlamaChatFormat but adapted for cl100k_base tokenization.
 */
public class Cl100kChatFormat implements ChatFormat {

    protected final Cl100kTokenizer tokenizer;
    protected final int beginOfText;
    protected final int endHeader;
    protected final int startHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;
    protected final Set<Integer> stopTokens;

    public Cl100kChatFormat(Cl100kTokenizer tokenizer) {
        this.tokenizer = tokenizer;

        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        // Use discovered special tokens or safe fallbacks
        this.beginOfText = specialTokens.getOrDefault("<|startoftext|>",
                          specialTokens.getOrDefault("<|endoftext|>", 1)); // Use endoftext as fallback
        this.endOfText = specialTokens.getOrDefault("<|endoftext|>",
                        specialTokens.keySet().iterator().hasNext() ?
                        specialTokens.values().iterator().next() : 0); // Use any available token

        // For chat headers, prefer im_start/im_end but fallback to simple tokens
        this.startHeader = specialTokens.getOrDefault("<|im_start|>", this.beginOfText);
        this.endHeader = specialTokens.getOrDefault("<|im_end|>", this.endOfText);

        this.endOfTurn = this.endHeader; // Use im_end as end of turn
        this.endOfMessage = this.endOfText; // Use endoftext as end of message

        // Only add available tokens to stop tokens
        Set<Integer> stopSet = new HashSet<>();
        if (specialTokens.containsKey("<|endoftext|>")) {
            stopSet.add(this.endOfText);
        }
        if (specialTokens.containsKey("<|im_end|>")) {
            stopSet.add(this.endHeader);
        }
        if (stopSet.isEmpty()) {
            stopSet.add(this.endOfText); // Always have at least one stop token
        }
        this.stopTokens = stopSet;

        System.out.printf("[CL100K-CHAT-FORMAT] Initialized with special tokens: " +
                         "beginOfText=%d, endOfText=%d, startHeader=%d, endHeader=%d (using %d discovered tokens)%n",
                         beginOfText, endOfText, startHeader, endHeader, specialTokens.size());
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();

        // Add start header token
        tokens.add(startHeader);

        // Add role
        tokens.addAll(tokenizer.encodeAsList(message.role().name()));

        // Add end header token
        tokens.add(endHeader);

        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = new ArrayList<>();

        // Add header
        tokens.addAll(encodeHeader(message));

        // Add newline after header
        tokens.addAll(tokenizer.encodeAsList("\n"));

        // Add content
        tokens.addAll(tokenizer.encodeAsList(message.content()));

        return tokens;
    }

    @Override
    public ChatTokens chatTokens() {
        return new ChatTokens(
            "<|im_start|>",     // tStartHeader
            "<|im_end|>",       // tEndHeader
            "<|im_end|>",       // tEndOfTurn (same as end header)
            "<|endoftext|>",    // tEndOfText
            "<|fim_suffix|>"    // tEndOfTextFim
        );
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }
}