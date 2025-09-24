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
    protected final ChatTokens chatTokens;

    public OlmoChatFormat(Tokenizer tokenizer) {
        this(tokenizer, null);
    }

    public OlmoChatFormat(Tokenizer tokenizer, ChatTokens chatTokens) {
        this.tokenizer = tokenizer;
        this.chatTokens = chatTokens;

        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        // Use chatTokens if provided, otherwise use OLMo defaults
        if (chatTokens != null) {
            System.out.println("[OLMO-CHAT-FORMAT] Using custom chat tokens for Tulu template");
            // For Tulu template, we don't need special token IDs - we'll encode the strings directly
            this.beginOfText = specialTokens.getOrDefault("<|endoftext|>", 50279);
            this.endOfText = specialTokens.getOrDefault("<|endoftext|>", 50279);
            this.startHeader = -1;  // Will use string encoding
            this.endHeader = -1;    // Will use string encoding
            this.endOfTurn = endOfText;
            this.endOfMessage = endOfText;
        } else {
            // Original OLMo special tokens
            this.beginOfText = specialTokens.getOrDefault("<|endoftext|>", 50279);  // OLMo uses EOS as BOT
            this.endOfText = specialTokens.getOrDefault("<|endoftext|>", 50279);
            this.startHeader = specialTokens.getOrDefault("<|user|>", -1);
            this.endHeader = specialTokens.getOrDefault("<|assistant|>", -1);
            this.endOfTurn = endOfText;  // Use EOS for end of turn
            this.endOfMessage = endOfText;
        }

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
        System.out.println("[OLMO-CHAT-DEBUG] 🎯 encodeHeader called with role: " + message.role() + ", content: '" + message.content() + "'");
        List<Integer> tokens = new ArrayList<>();

        // Start with user/assistant header
        if (message.role() == Role.USER) {
            System.out.println("[OLMO-CHAT-DEBUG] Adding USER header tokens");
            tokens.addAll(tokenizer.encodeAsList("<|user|>\n"));
        } else if (message.role() == Role.ASSISTANT) {
            System.out.println("[OLMO-CHAT-DEBUG] 🔥 ADDING ASSISTANT HEADER TOKENS!");
            List<Integer> assistantTokens = tokenizer.encodeAsList("<|assistant|>\n");
            System.out.println("[OLMO-CHAT-DEBUG] Assistant tokens: " + assistantTokens);
            tokens.addAll(assistantTokens);
        } else {
            System.out.println("[OLMO-CHAT-DEBUG] Adding fallback USER header for role: " + message.role());
            // Fallback for system messages
            tokens.addAll(tokenizer.encodeAsList("<|user|>\n"));
        }

        System.out.println("[OLMO-CHAT-DEBUG] encodeHeader returning " + tokens.size() + " tokens: " + tokens);
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

    /**
     * Encodes the conversation including the assistant prompt to trigger model response.
     * This method should be called when preparing the final prompt for generation.
     */
    public List<Integer> encodeConversation(List<Message> messages) {
        System.out.println("[OLMO-CHAT-DEBUG] 🔍 Encoding conversation with " + messages.size() + " messages");

        List<Integer> tokens = new ArrayList<>();

        // Encode all messages
        for (Message message : messages) {
            tokens.addAll(encodeMessage(message));
        }

        // CRITICAL: Add the assistant prompt to trigger model response
        List<Integer> assistantPrompt = tokenizer.encodeAsList("<|assistant|>\n");
        System.out.println("[OLMO-CHAT-DEBUG] 🎯 Adding assistant prompt tokens: " + assistantPrompt);
        tokens.addAll(assistantPrompt);

        System.out.println("[OLMO-CHAT-DEBUG] ✅ Final conversation tokens count: " + tokens.size());
        System.out.println("[OLMO-CHAT-DEBUG] 🔍 Final conversation tokens: " + tokens);

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