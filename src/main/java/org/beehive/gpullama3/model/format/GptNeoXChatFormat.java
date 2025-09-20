package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.GptNeoXTokenizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chat format for GPT-NeoX tokenizer (used by OLMoE models).
 * Based on GPT-NeoX specifications with OLMoE-specific adaptations.
 */
public class GptNeoXChatFormat implements ChatFormat {

    protected final GptNeoXTokenizer tokenizer;
    protected final int beginOfText;
    protected final int endOfText;
    protected final int startHeader;
    protected final int endHeader;
    protected final int endOfTurn;
    protected final int endOfMessage;
    protected final Set<Integer> stopTokens;

    public GptNeoXChatFormat(GptNeoXTokenizer tokenizer) {
        this.tokenizer = tokenizer;

        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        // Use discovered special tokens based on GPT-NeoX/OLMoE specification
        // EOS token is the primary end-of-text token in GPT-NeoX
        this.endOfText = specialTokens.getOrDefault("<|endoftext|>", 50279);
        this.beginOfText = this.endOfText; // GPT-NeoX typically uses same token for both

        // For chat formatting, use endoftext for headers if no specific chat tokens exist
        this.startHeader = this.endOfText;
        this.endHeader = this.endOfText;

        this.endOfTurn = this.endOfText;
        this.endOfMessage = this.endOfText;

        // Stop tokens include the main EOS token
        Set<Integer> stopSet = new HashSet<>();
        stopSet.add(this.endOfText);

        // Add padding token if available
        if (specialTokens.containsKey("<|padding|>")) {
            stopSet.add(specialTokens.get("<|padding|>"));
        }

        this.stopTokens = stopSet;

        System.out.printf("[GPTNEOX-CHAT-FORMAT] Initialized with special tokens: " +
                         "beginOfText=%d, endOfText=%d, startHeader=%d, endHeader=%d (using %d discovered tokens)%n",
                         beginOfText, endOfText, startHeader, endHeader, specialTokens.size());
    }

    @Override
    public ChatTokens chatTokens() {
        return new ChatTokens(
            "<|endoftext|>",    // tStartHeader (using EOS as generic delimiter)
            "<|endoftext|>",    // tEndHeader
            "<|endoftext|>",    // tEndOfTurn
            "<|endoftext|>",    // tEndOfText
            "<|endoftext|>"     // tEndOfTextFim (no specific FIM token in GPT-NeoX)
        );
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();

        // Simple header format for GPT-NeoX: role name
        tokens.addAll(tokenizer.encodeAsList(message.role().name()));
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
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }
}