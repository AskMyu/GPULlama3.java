package org.beehive.gpullama3;

import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;

public record Options(Path modelPath, String prompt, String systemPrompt, String suffix, boolean interactive, float temperature, float topp, int topK, long seed, int maxTokens, boolean stream, boolean echo,
                      boolean useTornadovm, Path imagePath) {

    public static final int DEFAULT_MAX_TOKENS = 1024;

    public Options {
        require(interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
        require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
        require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        require(topK > 0, "Invalid argument: --top-k must be positive");
    }

    static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            System.out.println("ERROR " + messageFormat.formatted(args));
            System.out.println();
            printUsage(System.out);
            System.exit(-1);
        }
    }

    private static boolean getDefaultTornadoVM() {
        // TornadoVM is enabled by default - only disabled when explicitly set to false
        String property = System.getProperty("use.tornadovm");
        boolean result;
        if (property == null) {
            result = true; // Default to TornadoVM when not specified
            System.err.println("[TORNADOVM-DEBUG] use.tornadovm property not set, defaulting to TRUE");
        } else {
            result = Boolean.parseBoolean(property);
            System.err.printf("[TORNADOVM-DEBUG] use.tornadovm property='%s', parsed to %b%n", property, result);
        }
        return result;
    }

    static void printUsage(PrintStream out) {
        out.println("Usage:  jbang Llama3.java [options]");
        out.println();
        out.println("Options:");
        out.println("  --model, -m <path>            required, path to .gguf file");
        out.println("  --interactive, --chat, -i     run in chat mode");
        out.println("  --instruct                    run in instruct (once) mode, default mode");
        out.println("  --prompt, -p <string>         input prompt");
        out.println("  --system-prompt, -sp <string> (optional) system prompt (Llama models)");
        out.println("  --suffix <string>             suffix for fill-in-the-middle request (Codestral)");
        out.println("  --temperature, -temp <float>  temperature in [0,inf], default 1.0 (Gemma optimized)");
        out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
        out.println("  --top-k <int>                 k value in top-k sampling, default 64 (Gemma optimized)");
        out.println("  --seed <long>                 random seed, default System.nanoTime()");
        out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS);
        out.println("  --stream <boolean>            print tokens during generation; may cause encoding artifacts for non ASCII text, default true");
        out.println("  --echo <boolean>              print ALL tokens to stderr, if true, recommended to set --stream=false, default false");
        out.println("  --use-tornadovm <boolean>     use TornadoVM GPU acceleration, default true (set to false to disable)");
        out.println("  --image <path>                path to input image file for multimodal inference (JPEG/PNG)");
        out.println();
    }

    public static Options getDefaultOptions() {
        String prompt = "Tell me a story with Java"; // Hardcoded for testing
        String systemPrompt = null;
        String suffix = null;
        float temperature = 1.0f;  // Google official recommendation for Gemma models
        float topp = 0.95f;
        int topK = 64;  // Google official recommendation for Gemma models
        Path modelPath = null;
        long seed = System.nanoTime();
        int maxTokens = DEFAULT_MAX_TOKENS;
        boolean interactive = false;
        boolean stream = true;
        boolean echo = false;
        boolean useTornadoVM = getDefaultTornadoVM();

        return new Options(modelPath, prompt, systemPrompt, suffix, interactive, temperature, topp, topK, seed, maxTokens, stream, echo, useTornadoVM, null);
    }

    /**
     * Get conservative sampling parameters optimized for Granite models.
     * These parameters prioritize more deterministic, focused responses.
     */
    public static Options withConservativeSampling(Options original) {
        // Conservative parameters for better response quality
        float conservativeTemperature = 0.2f;  // Much lower for more focused responses
        float conservativeTopp = 0.7f;         // More selective nucleus sampling
        int conservativeTopK = 20;             // Smaller candidate pool

        System.err.printf("[CONSERVATIVE-SAMPLING] Applied to Granite: temp=%.2f, top-p=%.2f, top-k=%d%n",
                        conservativeTemperature, conservativeTopp, conservativeTopK);

        return new Options(original.modelPath, original.prompt, original.systemPrompt, original.suffix,
                         original.interactive, conservativeTemperature, conservativeTopp, conservativeTopK,
                         original.seed, original.maxTokens, original.stream, original.echo,
                         original.useTornadovm, original.imagePath);
    }

    public static Options parseOptions(String[] args) {
        String prompt = "Tell me a story with Java"; // Hardcoded for testing
        String systemPrompt = null;
        String suffix = null;
        float temperature = 1.0f;  // Google official recommendation for Gemma models
        float topp = 0.95f;
        int topK = 64;  // Google official recommendation for Gemma models
        Path modelPath = null;
        long seed = System.nanoTime();
        int maxTokens = DEFAULT_MAX_TOKENS;
        boolean interactive = false;
        boolean stream = false;
        boolean echo = false;
        Boolean useTornadovm = null; // null means not specified via command line
        Path imagePath = null;

        for (int i = 0; i < args.length; i++) {
            String optionName = args[i];
            require(optionName.startsWith("-"), "Invalid option %s", optionName);
            switch (optionName) {
                case "--interactive", "--chat", "-i" -> interactive = true;
                case "--instruct" -> interactive = false;
                case "--help", "-h" -> {
                    printUsage(System.out);
                    System.exit(0);
                }
                default -> {
                    String nextArg;
                    if (optionName.contains("=")) {
                        String[] parts = optionName.split("=", 2);
                        optionName = parts[0];
                        nextArg = parts[1];
                    } else {
                        require(i + 1 < args.length, "Missing argument for option %s", optionName);
                        nextArg = args[i + 1];
                        i += 1; // skip arg
                    }
                    switch (optionName) {
                        case "--prompt", "-p" -> prompt = nextArg;
                        case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                        case "--suffix" -> suffix = nextArg;
                        case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                        case "--top-p" -> topp = Float.parseFloat(nextArg);
                        case "--top-k" -> topK = Integer.parseInt(nextArg);
                        case "--model", "-m" -> modelPath = Paths.get(nextArg);
                        case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                        case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                        case "--stream" -> stream = Boolean.parseBoolean(nextArg);
                        case "--echo" -> echo = Boolean.parseBoolean(nextArg);
                        case "--use-tornadovm" -> useTornadovm = Boolean.parseBoolean(nextArg);
                        case "--image" -> imagePath = Paths.get(nextArg);
                        default -> require(false, "Unknown option: %s", optionName);
                    }
                }
            }
        }

        require(modelPath != null, "Missing argument: --model <path> is required");

        if (useTornadovm == null) {
            useTornadovm = getDefaultTornadoVM();
        }

        return new Options(modelPath, prompt, systemPrompt, suffix, interactive, temperature, topp, topK, seed, maxTokens, stream, echo, useTornadovm, imagePath);
    }
}
