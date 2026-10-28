/**
 * AudioWorklet Processors for DeepFilterNet Live Voice Calls.
 * Includes:
 * 1. CaptureProcessor: collects 128-sample render slices into 1920-sample PCM16 frames (40 ms @ 48 kHz).
 * 2. PlaybackProcessor: ultra-low-latency jitter buffer managing smooth playback, underruns, and buffer metrics.
 */

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSize = 1920; // 40 ms at 48 kHz
    this.buffer = new Float32Array(this.frameSize);
    this.bufferIndex = 0;
    this.meterCounter = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const channelData = input[0];
    const len = channelData.length;

    let peak = 0;
    let sumSquares = 0;

    for (let i = 0; i < len; i++) {
      const sample = channelData[i];
      const abs = Math.abs(sample);
      if (abs > peak) peak = abs;
      sumSquares += sample * sample;

      this.buffer[this.bufferIndex++] = sample;

      if (this.bufferIndex >= this.frameSize) {
        // Quantize Float32 [-1.0, 1.0] to Int16 [-32768, 32767]
        const pcm16 = new Int16Array(this.frameSize);
        for (let j = 0; j < this.frameSize; j++) {
          const s = Math.max(-1.0, Math.min(1.0, this.buffer[j]));
          pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Transfer raw Int16Array buffer to main thread
        this.port.postMessage(
          {
            type: "chunk",
            pcm16: pcm16.buffer,
          },
          [pcm16.buffer]
        );

        this.bufferIndex = 0;
      }
    }

    this.meterCounter++;
    if (this.meterCounter >= 4) {
      this.meterCounter = 0;
      const rms = Math.sqrt(sumSquares / len);
      this.port.postMessage({
        type: "meter",
        peak: peak,
        rms: rms,
      });
    }

    return true;
  }
}

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Circular ring buffer capacity: 48,000 samples (1.0 second of audio)
    this.capacity = 48000;
    this.buffer = new Float32Array(this.capacity);
    this.writeIndex = 0;
    this.readIndex = 0;
    this.available = 0;

    // Fast interactive buffering: start playing as soon as 1 frame (1920 samples = 40 ms) arrives
    this.targetBufferSamples = 1920;
    this.isBuffering = true;
    this.underrunCount = 0;
    this.meterCounter = 0;

    this.port.onmessage = (e) => {
      const msg = e.data;
      if (!msg) return;

      if (msg.type === "enqueue") {
        const pcm16 = new Int16Array(msg.pcm16);
        const numSamples = pcm16.length;

        for (let i = 0; i < numSamples; i++) {
          const raw = pcm16[i];
          const sample = raw < 0 ? raw / 32768.0 : raw / 32767.0;
          this.buffer[this.writeIndex] = sample;
          this.writeIndex = (this.writeIndex + 1) % this.capacity;
        }

        this.available += numSamples;
        if (this.available > this.capacity) {
          // Overflow: advance read pointer
          this.readIndex = (this.writeIndex - this.capacity + this.capacity) % this.capacity;
          this.available = this.capacity;
        }

        if (this.isBuffering && this.available >= this.targetBufferSamples) {
          this.isBuffering = false;
        }
      } else if (msg.type === "set_target_buffer_ms") {
        const ms = Math.max(20, Math.min(500, msg.target_ms));
        this.targetBufferSamples = Math.round((ms / 1000.0) * 48000);
      } else if (msg.type === "reset") {
        this.readIndex = 0;
        this.writeIndex = 0;
        this.available = 0;
        this.isBuffering = true;
        this.underrunCount = 0;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const output = outputs[0];
    if (!output || !output[0]) return true;

    const channel = output[0];
    const quantum = channel.length; // Standard 128 samples

    // Initial pre-buffering
    if (this.isBuffering) {
      if (this.available >= this.targetBufferSamples) {
        this.isBuffering = false;
      } else {
        channel.fill(0);
        return true;
      }
    }

    if (this.available === 0) {
      channel.fill(0);
      this.underrunCount++;
      return true;
    }

    const toRead = Math.min(quantum, this.available);
    let sumSquares = 0;
    let peak = 0;

    for (let i = 0; i < toRead; i++) {
      const s = this.buffer[this.readIndex];
      this.readIndex = (this.readIndex + 1) % this.capacity;
      channel[i] = s;

      const abs = Math.abs(s);
      if (abs > peak) peak = abs;
      sumSquares += s * s;
    }

    for (let i = toRead; i < quantum; i++) {
      channel[i] = 0;
    }

    this.available -= toRead;

    this.meterCounter++;
    if (this.meterCounter >= 4) {
      this.meterCounter = 0;
      const rms = Math.sqrt(sumSquares / quantum);
      this.port.postMessage({
        type: "meter",
        rms: rms,
        peak: peak,
        available: this.available,
        underruns: this.underrunCount,
      });
    }

    return true;
  }
}

// Register both hyphenated and PascalCase names to guarantee instant cross-browser resolution
registerProcessor("capture-processor", CaptureProcessor);
registerProcessor("CaptureProcessor", CaptureProcessor);
registerProcessor("playback-processor", PlaybackProcessor);
registerProcessor("PlaybackProcessor", PlaybackProcessor);
