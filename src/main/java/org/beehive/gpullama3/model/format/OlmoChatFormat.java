package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.Tokenizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chat format for OLMo tokenizer (used by OLMoE models).
 * Based on OLMo specifications with Tulu chat template format.
 */
public class OlmoChatFormat implements ChatFormat {

    protected final Tokenizer tokenizer;
    protected final int beginOfText;
    protected final int endOfText;
    protected final int startHeader;
    protected final int endHeader;
    protected final int endOfTurn;
    protected final int endOfMessage;
    protected final Set<Integer> stopTokens;

    public OlmoChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;

        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        // OLMo special tokens
        this.beginOfText = specialTokens.getOrDefault("<|endoftext|>", 50279);  // OLMo uses EOS as BOT
        this.endOfText = specialTokens.getOrDefault("<|endoftext|>", 50279);
        this.startHeader = specialTokens.getOrDefault("<|user|>", -1);
        this.endHeader = specialTokens.getOrDefault("<|assistant|>", -1);
        this.endOfTurn = endOfText;  // Use EOS for end of turn
        this.endOfMessage = endOfText;

        // Configure stop tokens for OLMo
        this.stopTokens = new HashSet<>();
        stopTokens.add(endOfText);
        stopTokens.add(endOfTurn);

        System.out.printf("[OLMO-CHAT-FORMAT] Initialized OLMo chat format: " +
                         "beginOfText=%d, endOfText=%d, startHeader=%d, endHeader=%d (using %d discovered tokens)%n",
                         beginOfText, endOfText, startHeader, endHeader, specialTokens.size());

        // Debug: Check what tokens 29 and 237 decode to (llama.cpp uses 29 for '<', we use 237)
        System.out.println("[OLMO-CHAT-FORMAT] 🔍 Token 29 decodes to: '" + tokenizer.decode(List.of(29)) + "'");
        System.out.println("[OLMO-CHAT-FORMAT] 🔍 Token 237 decodes to: '" + tokenizer.decode(List.of(237)) + "'");

        // Test encoding of '<' character specifically
        List<Integer> bracketTokens = tokenizer.encodeAsList("<");
        System.out.println("[OLMO-CHAT-FORMAT] 🔍 '<' encodes to tokens: " + bracketTokens);
    }

    @Override
    public ChatTokens chatTokens() {
        return new ChatTokens(
            "<|user|>",         // tStartHeader
            "<|assistant|>",    // tEndHeader
            "",                 // tEndOfTurn (handled by newlines)
            "<|endoftext|>",    // tEndOfText
            "<|endoftext|>"     // tEndOfTextFim
        );
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();

        // Start with user/assistant header
        if (message.role() == Role.USER) {
            tokens.addAll(tokenizer.encodeAsList("<|user|>\n"));
        } else if (message.role() == Role.ASSISTANT) {
            tokens.addAll(tokenizer.encodeAsList("<|assistant|>\n"));
        } else {
            // Fallback for system messages
            tokens.addAll(tokenizer.encodeAsList("<|user|>\n"));
        }

        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        System.out.println("[OLMO-CHAT-DEBUG] 🔍 Encoding message: role=" + message.role() + ", content='" + message.content() + "'");

        List<Integer> tokens = new ArrayList<>();

        // Add header
        List<Integer> headerTokens = encodeHeader(message);
        System.out.println("[OLMO-CHAT-DEBUG] Header tokens: " + headerTokens);
        tokens.addAll(headerTokens);

        // Add content
        List<Integer> contentTokens = tokenizer.encodeAsList(message.content());
        System.out.println("[OLMO-CHAT-DEBUG] Content tokens for '" + message.content() + "': " + contentTokens);
        tokens.addAll(contentTokens);

        // Add newline
        List<Integer> newlineTokens = tokenizer.encodeAsList("\n");
        System.out.println("[OLMO-CHAT-DEBUG] Newline tokens: " + newlineTokens);
        tokens.addAll(newlineTokens);

        System.out.println("[OLMO-CHAT-DEBUG] ✅ Final message tokens: " + tokens);
        System.out.println("[OLMO-CHAT-DEBUG] 🔍 Token breakdown:");
        for (int i = 0; i < tokens.size(); i++) {
            int tokenId = tokens.get(i);
            String decoded = tokenizer.decode(List.of(tokenId));
            System.out.printf("[OLMO-CHAT-DEBUG] Token[%d]: %d → '%s'%n", i, tokenId, decoded);
        }

        return tokens;
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return new HashSet<>(stopTokens);
    }
}