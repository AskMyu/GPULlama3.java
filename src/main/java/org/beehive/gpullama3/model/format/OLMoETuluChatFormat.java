package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.GptNeoXTokenizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chat format for OLMoE Instruct models using the Tulu chat template.
 *
 * The Tulu template format is:
 * <|user|>
 * {user_message}
 *
 * <|assistant|>
 * {assistant_response}
 *
 * This format is specifically designed for OLMoE Instruct models that were
 * fine-tuned using the Tulu instruction dataset.
 */
public class OLMoETuluChatFormat implements ChatFormat {

    protected final GptNeoXTokenizer tokenizer;
    protected final int beginOfText;
    protected final int endOfText;
    protected final Set<Integer> stopTokens;

    // Tulu template tokens
    protected final List<Integer> userToken;
    protected final List<Integer> assistantToken;
    protected final List<Integer> systemToken;
    protected final List<Integer> newlineToken;

    public OLMoETuluChatFormat(GptNeoXTokenizer tokenizer) {
        this.tokenizer = tokenizer;

        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        // Use discovered special tokens based on GPT-NeoX/OLMoE specification
        this.endOfText = specialTokens.getOrDefault("<|endoftext|>", 50279);
        this.beginOfText = this.endOfText; // GPT-NeoX typically uses same token for both

        // Pre-encode the Tulu chat template tokens
        this.userToken = tokenizer.encodeAsList("<|user|>");
        this.assistantToken = tokenizer.encodeAsList("<|assistant|>");
        this.systemToken = tokenizer.encodeAsList("<|system|>");
        this.newlineToken = tokenizer.encodeAsList("\n");

        // Stop tokens include the main EOS token
        Set<Integer> stopSet = new HashSet<>();
        stopSet.add(this.endOfText);

        // Add padding token if available
        if (specialTokens.containsKey("<|padding|>")) {
            stopSet.add(specialTokens.get("<|padding|>"));
        }

        this.stopTokens = stopSet;

        System.out.printf("[OLMOE-TULU-CHAT-FORMAT] Initialized Tulu chat template: " +
                         "user_tokens=%s, assistant_tokens=%s, system_tokens=%s%n",
                         userToken, assistantToken, systemToken);
    }

    @Override
    public ChatTokens chatTokens() {
        return new ChatTokens(
            "<|user|>",         // tStartHeader
            "",                 // tEndHeader (no explicit end header in Tulu)
            "",                 // tEndOfTurn (handled by newlines)
            "<|endoftext|>",    // tEndOfText
            "<|endoftext|>"     // tEndOfTextFim
        );
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();

        System.err.printf("[OLMOE-TULU-DEBUG] encodeHeader called for role='%s'%n", message.role().name());

        // Add the appropriate role token based on the Tulu template
        switch (message.role().name()) {
            case "system" -> {
                tokens.addAll(systemToken);
                System.err.printf("[OLMOE-TULU-DEBUG] Added system tokens: %s%n", systemToken);
            }
            case "user" -> {
                tokens.addAll(userToken);
                System.err.printf("[OLMOE-TULU-DEBUG] Added user tokens: %s%n", userToken);
            }
            case "assistant" -> {
                tokens.addAll(assistantToken);
                System.err.printf("[OLMOE-TULU-DEBUG] Added assistant tokens: %s%n", assistantToken);
            }
            default -> {
                // Fallback to user for unknown roles
                System.err.printf("[OLMOE-TULU-CHAT-FORMAT] WARNING: Unknown role '%s', using 'user'%n",
                                message.role().name());
                tokens.addAll(userToken);
            }
        }

        // Add newline after role token
        tokens.addAll(newlineToken);
        System.err.printf("[OLMOE-TULU-DEBUG] Added newline tokens: %s%n", newlineToken);
        System.err.printf("[OLMOE-TULU-DEBUG] encodeHeader returning %d tokens: %s%n", tokens.size(), tokens);

        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = new ArrayList<>();

        System.err.printf("[OLMOE-TULU-DEBUG] encodeMessage called for role='%s', content='%s'%n",
                         message.role().name(), message.content());

        // Add role header (e.g., "<|user|>\n")
        tokens.addAll(encodeHeader(message));

        // Add message content
        List<Integer> contentTokens = tokenizer.encodeAsList(message.content());
        System.err.printf("[OLMOE-TULU-DEBUG] Content '%s' encoded to %d tokens: %s%n",
                         message.content(), contentTokens.size(), contentTokens);
        tokens.addAll(contentTokens);

        // Add double newline to separate messages (Tulu format convention)
        tokens.addAll(newlineToken);
        tokens.addAll(newlineToken);
        System.err.printf("[OLMOE-TULU-DEBUG] Added double newline, total tokens: %d%n", tokens.size());

        System.err.printf("[OLMOE-TULU-DEBUG] encodeMessage returning %d tokens: %s%n", tokens.size(), tokens);
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