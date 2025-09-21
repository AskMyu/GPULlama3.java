package org.beehive.gpullama3.inference.operation;

import org.beehive.gpullama3.core.types.Pair;

public final class RoPE {
    public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta,
            boolean ropeScaling, float scaleFactor, float loFreqFactor, float hiFreqFactor, float oldContextLength) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                // CRITICAL FIX: RoPE frequency formula was wrong
                // OLD (wrong): freq = 1.0 / pow(theta, i / headSize) where i = 0,2,4,6...
                // NEW (correct): freq = 1.0 / pow(theta, (i/2) / (headSize/2)) = 1.0 / pow(theta, i / headSize)
                // But since i increments by 2, we need (i/2) / (headSize/2) = i / headSize
                // Actually the issue is different - we need proper dimension index
                int freqIndex = i / 2; // Convert pair index to frequency index
                float freq = (float) (1.0 / Math.pow(theta, (2.0 * freqIndex) / (double) headSize));
                if (ropeScaling) {
                    // Llama 3.1 scaling
                    float loFreqWavelen = oldContextLength / loFreqFactor;
                    float hiFreqWavelen = oldContextLength / hiFreqFactor;
                    float wavelen = (float) (2.0 * Math.PI / freq);
                    if (wavelen < hiFreqWavelen) {
                        freq = freq;
                    } else if (wavelen > loFreqWavelen) {
                        freq = freq / scaleFactor;
                    } else {
                        float smooth = (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor);
                        freq = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
                    }
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);

                // Debug: Log frequency computation for first few positions/frequencies
                if (pos <= 1 && freqIndex <= 4) {
                    System.err.printf("[ROPE-COMPUTE-DEBUG] pos=%d, freqIndex=%d, freq=%.6f, val=%.6f, cos=%.6f, sin=%.6f%n",
                        pos, freqIndex, freq, val, cr[n], ci[n]);
                }

                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }
}