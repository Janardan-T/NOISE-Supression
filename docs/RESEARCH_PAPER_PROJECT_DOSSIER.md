# RESEARCH PAPER PROJECT DOSSIER: NOISESHIELD
## A Comprehensive Scientific & Engineering Brief for Academic Paper Co-Authorship & LLM Prompting

> **Purpose of this Document:**  
> This dossier contains the complete theoretical, architectural, algorithmic, empirical, and production engineering details of **NoiseShield**. It is specifically structured for direct upload or ingestion into external advanced AI models (e.g., ChatGPT Plus / GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) to co-author, draft, format, and refine high-impact academic research papers for premier venues such as **IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)**, **IEEE ICASSP**, **INTERSPEECH**, or the **Journal of the Audio Engineering Society (JAES)**.

---

# TABLE OF CONTENTS
1. [Paper Metadata, Taxonomy & Executive Summary](#1-paper-metadata-taxonomy--executive-summary)
2. [Target Venue Guidelines & Paper Structure Outline](#2-target-venue-guidelines--paper-structure-outline)
3. [Theoretical Foundations & Mathematical Formulations](#3-theoretical-foundations--mathematical-formulations)
4. [System Architecture & Production Engineering Implementation](#4-system-architecture--production-engineering-implementation)
5. [Underground Engineering: Forensic Analysis of 6 Critical Production Defects](#5-underground-engineering-forensic-analysis-of-6-critical-production-defects)
6. [Experimental Setup, Datasets & Acoustic Methodologies](#6-experimental-setup-datasets--acoustic-methodologies)
7. [Comprehensive Quantitative Results & Empirical Benchmarks](#7-comprehensive-quantitative-results--empirical-benchmarks)
8. [Acoustic & Perceptual Discussions (Anomalies, Room Tone & Naturalness)](#8-acoustic--perceptual-discussions-anomalies-room-tone--naturalness)
9. [Complete LaTeX Equation Bank & Formulations](#9-complete-latex-equation-bank--formulations)
10. [Complete BibTeX Reference Library](#10-complete-bibtex-reference-library)
11. [Turn-by-Turn LLM Prompting Playbook](#11-turn-by-turn-llm-prompting-playbook)
12. [Project Codebase & File Manifest](#12-project-codebase--file-manifest)

---

# 1. PAPER METADATA, TAXONOMY & EXECUTIVE SUMMARY

### 1.1 Candidate Titles
- **Primary (Systems & Applications):**  
  *NoiseShield: Full-Duplex 48 kHz Neural Speech Enhancement for Live Web Telephony via DeepFilterNet3 and AudioWorklets*
- **Alternative (Theoretical & Signal Processing):**  
  *Real-Time Fullband Complex Deep Filtering in Asynchronous Web Architectures: Design, Latency Profiling, and In-Situ Perceptual Evaluation*
- **Alternative (Engineering & Applied):**  
  *Zero-Configuration Real-Time Acoustic Noise Suppression: Bridging High-Fidelity Fullband Deep Filtering with Consumer Mobile Web Telephony*

### 1.2 Author Context & Project Repository
- **Authors:** Engineering & Applied Audio Intelligence Research Team
- **Affiliation:** Audio Intelligence Systems Laboratory
- **Repository Location:** `e:\bnsup`
- **Key Modules:** `src/call/call_server.py`, `src/call/static/worklets.js`, `src/call/static/app.js`, `src/realtime/server.py`, `src/denoise.py`, `src/evaluate.py`, `src/tradeoff.py`

### 1.3 Abstract Draft (Academic Standard)
> Real-time telecommunication in untreated acoustic environments suffers from diverse stationary and non-stationary acoustic noise, degrading intelligibility and inducing acute listener fatigue ("Zoom fatigue"). Conventional DSP estimators (e.g., spectral subtraction, Wiener filtering) suffer from musical noise artifacts and phase degradation, while modern deep learning approaches traditionally demand GPU acceleration or restrict bandwidth to 16 kHz narrowband/wideband signals. This paper presents **NoiseShield**, an open-source, full-duplex live telecommunication studio delivering native fullband (48 kHz) neural speech enhancement on standard consumer multi-core CPUs without dedicated hardware acceleration. Powered by **DeepFilterNet3**, NoiseShield combines Equivalent Rectangular Bandwidth (ERB) filterbank gain estimation across 32 psychoacoustic bands with Complex Deep Filtering across the 0–8 kHz spectrum, directly predicting complex linear prediction coefficients to restore both magnitude and phase. 
> 
> The system integrates a high-throughput asynchronous Python server (FastAPI, WebSockets, PyTorch CPU with thread pinning and 200 ms stateful context re-priming) with a browser client utilizing dedicated `AudioWorklet` processors for jitter buffering and non-blocking PCM16 binary streaming. Integrated automated Cloudflare tunneling and dynamic QR code generation enable zero-configuration, cross-platform telephony across mobile Safari, mobile Chrome, and desktop environments. We document the forensic root-cause analysis of six silent-audio failure modes in binary alignment and browser thread scheduling. Comprehensive empirical evaluation across 45 multi-condition test sets (LibriSpeech + VoiceBank-DEMAND) demonstrates a **+11.88 dB** improvement in SI-SDR, a **+0.95** increase in DNSMOS Background Quality, and **>30 dB** of noise-floor suppression in authentic HVAC field recordings with a real-time computation factor (RTF) of 0.615 (24.6 ms computation per 40.0 ms frame hop).

### 1.4 ACM/IEEE Taxonomy & Keywords
- **Keywords:** Neural Speech Enhancement, DeepFilterNet3, Complex Linear Prediction, Web Audio API, AudioWorklets, Full-Duplex Telephony, Real-Time Systems, Psychoacoustics, Perceptual Speech Quality.
- **ACM CCS (2012):** Computing methodologies ~ Machine learning ~ Machine learning approaches ~ Neural networks; Applied computing ~ Sound and music computing; Software and its engineering ~ Real-time systems software.
- **IEEE Keywords:** Speech enhancement, Real-time systems, Acoustics, Audio streaming, Neural networks, Internet telephony.

---

# 2. TARGET VENUE GUIDELINES & PAPER STRUCTURE OUTLINE

### 2.1 Recommended Publication Venues
1. **IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)**  
   *Focus:* Deep mathematical formulation of complex filtering, fullband psychoacoustics, exhaustive statistical evaluation across 45 conditions, and ablation of stateful context re-priming. (Target: Regular 10–13 page paper).
2. **INTERSPEECH (ISCA) / IEEE ICASSP**  
   *Focus:* Real-time low-latency CPU implementation, 48 kHz fullband streaming architecture, AudioWorklet jitter buffer design, and production engineering bug resolution. (Target: 4-page + 1-page references conference paper).
3. **Journal of the Audio Engineering Society (JAES)**  
   *Focus:* Applied studio-quality telephony, naturalness preservation (dead-silence fatigue, dry-mix blending, attenuation limiting), and zero-configuration mobile ingress.

### 2.2 Standard IEEE Transactions Paper Outline
```
1. INTRODUCTION
   1.1 The Acoustic Crisis in Ubiquitous Remote Work & Telephony
   1.2 Psychoacoustic Impact of Noise & Cognitive Listening Fatigue
   1.3 Why Fullband (48 kHz) Audio is Imperative for Natural Voice
   1.4 Contributions of this Work
2. RELATED WORK & PREVIOUS PARADIGMS
   2.1 Classical DSP: Spectral Subtraction, MMSE-STSA, and Phase Limitations
   2.2 First-Generation Deep Learning: Ideal Ratio Masking & The 16 kHz Bottleneck
   2.3 Modern Complex Spectral & Linear Prediction Neural Architectures
3. MATHEMATICAL & THEORETICAL FORMULATION OF DEEPFILTERNET3
   3.1 The Signal Degradation Model in STFT Domain
   3.2 Psychoacoustic Frequency Warping: 32 ERB Filterbanks
   3.3 Complex Deep Filtering: Formulation of Multi-Frame Complex Linear Prediction
   3.4 Power-Law Compressed Multi-Resolution Loss Optimization
4. NOISESHIELD SYSTEM ARCHITECTURE & ENGINEERING IMPLEMENTATION
   4.1 Client-Side Real-Time Web Audio & AudioWorklet Pipelines
   4.2 Custom 12-Byte / 16-Byte Big-Endian Binary Transport Protocol
   4.3 Stateful Context Re-Priming Engine (StreamDenoiser with 200 ms History)
   4.4 Server-Side Concurrency, PyTorch Thread Pinning, & Dual-Path Distribution
   4.5 Zero-Configuration Tunneling & Cross-Platform Mobile Ingress
5. FORENSIC POST-MORTEM: OVERCOMING 6 CRITICAL PRODUCTION DEFECTS
   5.1 Binary Header Alignment & Endianness Serialization Drift
   5.2 Uncaught DataView RangeError on Ingress Decoding
   5.3 AudioWorklet Case-Sensitive Name Collision
   5.4 Jitter Buffer Lockout & Starvation Recovery
   5.5 The Near-Silence Clipping Trap (RMS Zeroing vs. Transparent Forwarding)
   5.6 Mobile AudioContext Autoplay Suspension & Script Caching
6. EXPERIMENTAL SETUP & EVALUATION METHODOLOGY
   6.1 Datasets: LibriSpeech, VoiceBank-DEMAND, & Authentic Field Recordings
   6.2 Noise Types & Acoustic Stress Testing (White, Pink, 50/60 Hz Hum, Babble)
   6.3 Objective Perceptual Metrics (PESQ-WB, STOI, SI-SDR, DNSMOS, RTF)
7. QUANTITATIVE RESULTS & PERFORMANCE ANALYSIS
   7.1 Multi-Condition Benchmark (45 Synthetic & Real Acoustic Scenarios)
   7.2 Noise-Specific Breakdown & Analysis of Extreme Conditions
   7.3 Authentic Field Recording Suppression (Unreferenced HVAC Analysis)
   7.4 CPU Frame Latency, Real-Time Factor (RTF), & Thread Contention Benchmarks
8. ACOUSTIC ABLATIONS & PERCEPTUAL NATURALNESS STUDY
   8.1 The "Dead-Silence Fatigue" Paradox: Attenuation Limiting & Dry-Mix Control
   8.2 The Clean Reference Room-Tone Artifact in Objective PESQ Scoring
   8.3 The Real-Time vs. Offline Phenomenon on Low-Frequency & Babble Signals
9. FUTURE RESEARCH DIRECTIONS
   9.1 In-Browser Neural Inference (WASM / WebGPU)
   9.2 Selective Forwarding Units (SFU) for N-Way Spatial Conferences
   9.3 Target Speaker Extraction via Acoustic Voice Fingerprinting
   9.4 Neural Acoustic Echo Cancellation (AEC)
10. CONCLUSION & OPEN SCIENCE
```

---

# 3. THEORETICAL FOUNDATIONS & MATHEMATICAL FORMULATIONS

### 3.1 The Acoustic Signal Degradation Model
In time domain, the contaminated microphone signal $y(t)$ comprises clean target speech $s(t)$, additive acoustic noise $n(t)$, and room reverberation $h(t)$:
$$y(t) = s(t) * h(t) + n(t)$$
Assuming localized anechoic or lightly reverberant environments, the short-time Fourier transform (STFT) with analysis window $w(\tau)$ maps the 48 kHz continuous sequence into discrete time-frequency bins:
$$Y(k, m) = S(k, m) + N(k, m) = |Y(k, m)| e^{j \angle Y(k, m)}$$
where $k \in \{0, 1, \dots, K-1\}$ represents the discrete frequency bin (for FFT size $N_{\text{FFT}} = 960$, $K = 481$), and $m \in \mathbb{Z}$ is the temporal frame index with frame hop $R = 480$ samples ($10.0\text{ ms}$).

### 3.2 The Fundamental Failure Modes of Classical DSP
1. **Spectral Subtraction (Boll, 1979):**
   $$|\hat{S}(k, m)| = \max\left( |Y(k, m)|^\gamma - \alpha |\hat{N}(k)|^\gamma, \; \beta |Y(k, m)|^\gamma \right)^{1/\gamma}$$
   *Failure:* The expectation $\mathbb{E}[|N(k)|^2]$ diverges from instantaneous noise variance. Random residual peaks appear across isolated time-frequency cells, producing synthetic "musical noise."
2. **Wiener & MMSE Estimators (Ephraim & Malah, 1984):**
   $$\hat{S}(k, m) = G(k, m) Y(k, m), \quad G(k, m) = \frac{\xi(k, m)}{1 + \xi(k, m)}$$
   *Failure:* Relies on stationary noise tracking via minimum statistics. Transient non-stationary bursts (keyboard clicks, vehicle horns) violate stationarity, causing speech attenuation or pass-through leakage.
3. **The Phase Neglect Problem:**
   $$\hat{S}(k, m) = |\hat{S}(k, m)| \cdot e^{j \angle Y(k, m)}$$
   *Failure:* Modifying only magnitude while preserving corrupt noisy phase $\angle Y(k, m)$ creates severe roughness, hollow timbres, and distortion at low SNRs ($< 5\text{ dB}$).

### 3.3 Psychoacoustic ERB Filterbank Dimensionality Reduction
Human hearing frequency resolution adheres to the Equivalent Rectangular Bandwidth (ERB) scale (Glasberg & Moore, 1990):
$$\text{ERB}(f) = 24.7 \left( 4.37 \cdot \frac{f}{1000} + 1 \right)$$
The continuous frequency $f$ (in Hz) is warped to the ERB index scale $G_{\text{ERB}}(f)$:
$$G_{\text{ERB}}(f) = 21.4 \log_{10} \left( 0.00437 f + 1 \right)$$
DeepFilterNet3 maps $K = 481$ linear STFT bins to $B = 32$ psychoacoustic bands using triangular bandpass filters $W(b, k)$:
$$E(b, m) = \sum_{k=0}^{K-1} W(b, k) |Y(k, m)|^2, \quad b \in \{0, 1, \dots, 31\}$$
This yields a **$15\times$ to $30\times$ operational reduction**, compressing the full 48 kHz envelope into 32 compact features per frame, easily modeled by lightweight recurrent units (Grouped GRUs).

### 3.4 Complex Deep Filtering Formulation
To reconstruct both magnitude and phase without spectral synthesis artifacts, DeepFilterNet3 applies **Complex Linear Prediction (Deep Filtering)** to the lower frequency region ($k \le K_{\text{DF}}$, typically covering $0\text{ to } 8\text{ kHz}$):
$$\hat{S}_{\text{DF}}(k, m) = \sum_{l=0}^{L_p} C(k, m, l)^* \cdot Y(k, m - l)$$
where:
- $C(k, m, l) \in \mathbb{C}$ is a complex filter coefficient dynamically predicted by the neural network for frequency $k$, current frame $m$, and time lag $l$.
- $L_p$ is the causal filter order (number of past context frames, e.g., $L_p = 4$ or $5$).
- $(\cdot)^*$ denotes the complex conjugate.
- Future lookahead $L_f = 0$ (strictly causal for interactive telephony).

Decomposing into real and imaginary Cartesian components:
$$C(k, m, l) = C_r + j C_i, \quad Y(k, m - l) = Y_r + j Y_i$$
$$C^* \cdot Y = (C_r - j C_i)(Y_r + j Y_i) = (C_r Y_r + C_i Y_i) + j (C_r Y_i - C_i Y_r)$$
By rotating the observed noisy vector in the complex plane, Complex Deep Filtering reconstructs the periodic harmonic structure of clean speech formants while attenuating uncorrelated noise, completely eliminating musical noise and phase roughness.

### 3.5 Composite Multi-Resolution Objective Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{spec}} + \alpha \mathcal{L}_{\text{phase}} + \beta \mathcal{L}_{\text{multi-STFT}}$$
$$\mathcal{L}_{\text{spec}} = \frac{1}{M \cdot K} \sum_{m=1}^{M} \sum_{k=1}^{K} \left| |S(k, m)|^{0.3} - |\hat{S}(k, m)|^{0.3} \right|^2$$
Power-law compression $(\cdot)^{0.3}$ compresses dynamic range, matching acoustic sound pressure perception (Stevens' Power Law) and heavily weighting quiet consonant definitions.

---

# 4. SYSTEM ARCHITECTURE & PRODUCTION ENGINEERING IMPLEMENTATION

```
====================================================================================================
NOISESHIELD FULL-DUPLEX 48 kHz ARCHITECTURAL TOPOLOGY
====================================================================================================

 [ CLIENT BROWSER (Desktop/Phone) ]
 +------------------------------------------------------------------------------------------------+
 | Audio Context (48 kHz, LatencyHint: "interactive")                                            |
 |   |                                                                                            |
 |   v                                                                                            |
 | Microphone Node (echoCancellation: true, noiseSuppression: false, autoGainControl: false)     |
 |   |                                                                                            |
 |   v                                                                                            |
 | CaptureProcessor (AudioWorkletThread)                                                          |
 |   - Aggregates 128-sample render quanta into exact 1,920-sample frames (40.0 ms @ 48 kHz)      |
 |   - Quantizes Float32 [-1.0, 1.0] -> Int16 [-32768, 32767]                                     |
 |   - Zero-copy transfers ArrayBuffer via port.postMessage()                                     |
 +---|--------------------------------------------------------------------------------------------+
     |
     v Int16Array (1920 samples)
 +------------------------------------------------------------------------------------------------+
 | Client Transport Layer (Main UI Thread: app.js)                                               |
 |   - Serializes 12-Byte Big-Endian Header (>IIBBH): seq (u32), ts (u32), mode (u8), flags (u8)   |
 |   - Appends 3,840 bytes raw PCM audio (Total frame: 3,852 bytes)                               |
 +---|--------------------------------------------------------------------------------------------+
     |
     | [ Asynchronous WebSocket: wss:// via Cloudflare Tunnel / Local LAN ]
     v
 [ FASTAPI BACKEND SERVER (Python 3.10+, PyTorch CPU) ]
 +------------------------------------------------------------------------------------------------+
 | WebSocket Ingress: /ws/room/{room_code}                                                        |
 |   - Deserializes 12-byte header with struct.unpack_from(">IIBBH")                              |
 |   - Verifies 3,840-byte audio payload integrity                                               |
 |   - Enqueues into participant.frame_queue (asyncio.Queue, maxsize=30)                          |
 +---|--------------------------------------------------------------------------------------------+
     |
     v
 +------------------------------------------------------------------------------------------------+
 | Room Audio Worker: room_audio_processor()                                                      |
 |   - Overload Check: If queue.qsize() >= 2 -> Set bypass flag 0x02                             |
 |   - Silence Detection: If RMS < 20.0 -> Transparent passthrough (bypass flag 0x01)             |
 |   - Neural Offload: ThreadPoolExecutor(max_workers=4) with single-thread PyTorch               |
 |                                                                                                |
 |   +-- StreamDenoiser Engine (src/realtime/server.py) ---------------------------------------+  |
 |   |   - Maintains 200 ms (9,600 samples) rolling context buffer                             |  |
 |   |   - Shifts context: rolling_buffer = cat([rolling_buffer[:, chunk_len:], new_chunk])   |  |
 |   |   - Executes DeepFilterNet3 CPU inference (~24.6 ms)                                    |  |
 |   |   - Slices enhanced slice: out = enhanced[:, -chunk_len:]                               |  |
 |   +-----------------------------------------------------------------------------------------+  |
 +---|--------------------------------------------------------------------------------------------+
     |
     +-----------------------------------------+-----------------------------------------+
     | Path 1: Telemetry ACK to Sender         | Path 2: Clean Audio Relay to Remote Peer |
     | Header: >IIfBBH (16 bytes)              | Header: >IIfBBH (16 bytes)              |
     | Payload: 0 bytes audio                  | Payload: 3,840 bytes clean Int16 audio  |
     | Used for Round-Trip Time (RTT) tracking | Total: 3,856 bytes                      |
     +-----------------------------------------+-----------------------------------------+
                                                   |
                                                   v
 [ REMOTE PEER BROWSER CLIENT ]
 +------------------------------------------------------------------------------------------------+
 | Client Receiver (app.js): handleIncomingAudioPacket()                                          |
 |   - Unpacks 16-byte server header (>IIfBBH)                                                    |
 |   - Extracts 3,840 bytes clean Int16 PCM audio                                                 |
 |   - Posts Int16 buffer to PlaybackProcessor AudioWorklet                                       |
 +---|--------------------------------------------------------------------------------------------+
     |
     v
 +------------------------------------------------------------------------------------------------+
 | PlaybackProcessor (AudioWorkletThread: worklets.js)                                            |
 |   - Ingests into 48,000-sample circular ring buffer (1.0 second headroom)                      |
 |   - Jitter buffer startup threshold: 1,920 samples (40.0 ms)                                   |
 |   - Delivers seamless 128-sample render slices to browser DAC audio output                     |
 +------------------------------------------------------------------------------------------------+
====================================================================================================
```

### 4.1 Custom Binary Packet Serialization Specification
To avoid Base64 text inflation ($+33\%$ bandwidth overhead) and JSON serialization CPU bottlenecks, NoiseShield implements a custom binary framing protocol using network byte order (Big-Endian `>`):

#### Client-to-Server Binary Frame (3,852 Bytes Total)
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Sequence Number (uint32)                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Client Timestamp (uint32 ms)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Mode (uint8)|  Flags (uint8)|        Reserved (uint16)      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|        Raw PCM16 Audio Payload (1,920 samples = 3,840 bytes)  |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Server-to-Client Binary Frame (3,856 Bytes for Peer Relay; 16 Bytes for Sender ACK)
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Sequence Number (uint32)                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Client Timestamp (uint32 ms)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                Server Processing Time (float32 ms)            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Mode (uint8)|  Flags (uint8)|        Reserved (uint16)      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|     Clean Enhanced PCM16 Audio Payload (3,840 bytes)          |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### 4.2 Stateful Context Re-Priming Engine (`StreamDenoiser`)
In streaming neural processing, recurrent networks (GRUs) suffer from state-reset boundary artifacts if executed on discrete frames without context. NoiseShield introduces **Context Re-Priming**:
```python
class StreamDenoiser:
    def __init__(self, context_buf_size: int = 9600):  # 200 ms history @ 48 kHz
        self.context_buf_size = context_buf_size
        self.rolling_buffer = torch.zeros(1, context_buf_size)

    def process_chunk_bytes(self, raw_bytes: bytes, model, df_state, atten_lim_db=None) -> bytes:
        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        chunk_len = audio_tensor.shape[1]

        # Shift trailing context and append new incoming chunk
        self.rolling_buffer = torch.cat([self.rolling_buffer[:, chunk_len:], audio_tensor], dim=-1)

        # Enhance composite rolling buffer
        enhanced_buf = enhance(model=model, df_state=df_state, audio=self.rolling_buffer, pad=True, atten_lim_db=atten_lim_db)

        # Extract only the newly enhanced chunk
        enhanced_tensor = enhanced_buf[:, -chunk_len:]
        out_int16 = (enhanced_tensor.squeeze(0).cpu().numpy().clip(-1.0, 1.0) * 32767.0).astype(np.int16)
        return out_int16.tobytes()
```

---

# 5. UNDERGROUND ENGINEERING: FORENSIC ANALYSIS OF 6 CRITICAL PRODUCTION DEFECTS

During system integration, complete room signaling and WebSocket handshakes succeeded, but calls suffered from **complete transmission silence or white static**. Forensic analysis uncovered and resolved six subtle system defects:

```
+====================================================================================================+
| SUMMARY TABLE: PRODUCTION FAILURE MODES & RESOLUTIONS                                             |
+====================================================================================================+
| Issue ID | Fault Domain      | Root Cause                               | Observable Symptom      |
+----------+-------------------+------------------------------------------+-------------------------+
| BUG-01   | Binary Framing    | 16-byte Little-Endian Header on Client   | White noise static;     |
|          |                   | vs 12-byte Big-Endian Header on Server   | Server stuck in bypass  |
+----------+-------------------+------------------------------------------+-------------------------+
| BUG-02   | Client Deserial.  | DataView(buf, 0, 16).getUint8(16)        | Uncaught RangeError;    |
|          |                   | Index 16 out of bounds [0..15]           | Playback aborts         |
+----------+-------------------+------------------------------------------+-------------------------+
| BUG-03   | Worklet Lifecycle | "capture-processor" in worklets.js vs    | InvalidStateError;      |
|          |                   | "CaptureProcessor" in app.js             | Audio graph dead        |
+----------+-------------------+------------------------------------------+-------------------------+
| BUG-04   | Jitter Buffer     | Buffer dropped < 128 samples forced full | Perpetual silence on    |
|          |                   | re-buffer lockout (required 3840 smp)    | real-world networks     |
+----------+-------------------+------------------------------------------+-------------------------+
| BUG-05   | Server DSP        | SILENCE_RMS_THRESHOLD = 100.0 zeroed out | Quiet speech sentences  |
|          |                   | normal speech (-50 dBFS to -42 dBFS)     | completely muted        |
+----------+-------------------+------------------------------------------+-------------------------+
| BUG-06   | Mobile OS Sandbox | AudioContext suspended on iOS/Android;   | Phones connect but hear |
|          |                   | aggressive browser HTTP caching of scripts| and transmit nothing   |
+====================================================================================================+
```

### Forensic Walkthroughs

#### BUG-01: Endianness & Binary Header Alignment Drift
- **Root Cause:** In `app.js`, `sendAudioChunk()` initially serialized client timestamps using a 64-bit float (`Float64`), creating a 16-byte header. In `call_server.py`, the server unpacked `>IIBBH` (12 bytes).
- **Physical Consequence:** Unpacking 12 bytes shifted the audio payload start offset by $4\text{ bytes}$ ($3,844\text{ bytes}$ remaining). This broke the 16-bit integer parity ($2\text{ bytes/sample}$), swapping high and low bytes across the entire PCM stream and transforming human voice into loud digital white noise. Additionally, byte offset 8 was decoded as mode $214$ instead of $0$, permanently tripping the bypass mechanism.
- **Resolution:** Explicitly unified client and server to 12-byte Big-Endian packing via `DataView.setUint32(offset, val, false)`.

#### BUG-02: Client Ingress RangeError Out-of-Bounds
- **Root Cause:** When receiving server audio, `app.js` sliced a 16-byte header: `const view = new DataView(buffer, 0, 16);`. It then evaluated `const flags = view.getUint8(16);`.
- **Physical Consequence:** In JavaScript, `DataView` indices are $0$-indexed up to $\text{byteLength} - 1$. Index 16 on a 16-byte view throws an unhandled `RangeError`. This exception killed the message handler before the audio payload could be passed to the audio node, resulting in total silence.
- **Resolution:** Re-indexed struct offsets: sequence at 0, timestamp at 4, processing time at 8, mode at 12, flags at 13.

#### BUG-03: Worklet Class Name Collision
- **Root Cause:** In `worklets.js`, the worklet processor was registered with kebab-case: `registerProcessor("capture-processor", CaptureProcessor)`. In `app.js`, it was instantiated with PascalCase: `new AudioWorkletNode(ctx, "CaptureProcessor")`.
- **Physical Consequence:** W3C Web Audio specification mandates exact case matching. The mismatch threw `InvalidStateError`, preventing the capture graph from connecting.
- **Resolution:** Registered both naming variants in `worklets.js`.

#### BUG-04: Jitter Buffer Lockout & Starvation Recovery
- **Root Cause:** `PlaybackProcessor` entered an aggressive buffering state whenever buffer availability dropped below a single 128-sample render slice, refusing to output audio until $3,840\text{ samples}$ ($80\text{ ms}$) re-accumulated.
- **Physical Consequence:** Natural network arrival jitter caused momentary buffer starvation, locking the player into constant re-buffering cycles and silencing speech.
- **Resolution:** Lowered threshold to $1,920\text{ samples}$ ($40\text{ ms}$) and implemented non-blocking zero-filling on partial render quanta ($0 < \text{avail} < 128$) without triggering re-buffering mode.

#### BUG-05: Excessive Silence Suppression (RMS Hard Zeroing)
- **Root Cause:** In `call_server.py`, frames with $\text{RMS} < 100.0$ were replaced with `np.zeros_like(audio_np).tobytes()`.
- **Physical Consequence:** Typical unamplified consumer laptop/phone microphones produce conversational speech at $-48\text{ dBFS}$ to $-42\text{ dBFS}$ ($\text{RMS} \approx 30\text{--}80$). The server actively erased legitimate spoken words.
- **Resolution:** Lowered threshold to $\text{RMS} = 20.0$ ($\approx -64\text{ dBFS}$) and forwarded raw audio transparently rather than muting it.

#### BUG-06: Mobile Autoplay Suspension & Worklet Caching
- **Root Cause:** iOS Safari and Android Chrome initialize `AudioContext` in `"suspended"` mode unless unlocked by a direct user gesture. Additionally, mobile browsers cached older worklet scripts aggressively.
- **Physical Consequence:** Mobile devices connected via Cloudflare tunnel established WebSocket signaling but could neither send nor receive audio.
- **Resolution:** Bound automatic resume handlers to `touchstart`, `pointerdown`, and `click` on the `window` object, and appended cache-busting query parameters (`?v=12`) to script imports.

---

# 6. EXPERIMENTAL SETUP, DATASETS & ACOUSTIC METHODOLOGIES

### 6.1 Evaluation Corpora
1. **LibriSpeech (test-clean):** 5 distinct speakers (3 female, 2 male: IDs 374, 2277, 3660, 6930, 7902) resampled to 48 kHz.
2. **VoiceBank-DEMAND:** Real-world background noise environments (Clip A: `p232_001` public transit, Clip B: `p232_013` domestic cafeteria, Clip C: `p232_024` street traffic).
3. **Synthetic Stress Suite:** Gaussian white noise, pink noise ($1/f$ spectral roll-off), and 50/60 Hz powerline hum with harmonics up to 1 kHz.
4. **Same-Speaker Babble (Worst-Case Formant Masking):** Synthesized using 5 overlapping time-shifted copies of the target speaker's own voice ($0.55\text{s}, 1.15\text{s}, 1.75\text{s}, 2.35\text{s}$) with gains of $1.00$ and $0.85$. Target and interferers share identical pitch, vocal tract geometry, and formant distributions.
5. **Authentic Field Recording:** Unreferenced 48 kHz field recording (`samples/real_noisy_speech.wav`) containing continuous HVAC air-conditioning rumble and office background noise.

### 6.2 Standard Objective Metrics
- **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio, dB):**
  $$\text{SI-SDR} = 10 \log_{10} \left( \frac{\|\alpha s\|^2}{\|\hat{s} - \alpha s\|^2} \right), \quad \alpha = \frac{\hat{s}^T s}{\|s\|^2}$$
- **PESQ (Perceptual Evaluation of Speech Quality - ITU-T P.862.2 Wideband):** Validated mapping from $-0.5$ to $4.5$.
- **STOI (Short-Time Objective Intelligibility):** Correlation metric from $0.0$ to $1.0$.
- **DNSMOS P.835 (Deep Noise Suppression MOS):** Predicts subjective quality on a 1.0–5.0 scale across Speech (SIG), Background (BAK), and Overall (OVRL).
- **Real-Time Factor (RTF):**
  $$\text{RTF} = \frac{\text{Processing Time (ms)}}{40.0\text{ ms Hop}}$$

---

# 7. COMPREHENSIVE QUANTITATIVE RESULTS & EMPIRICAL BENCHMARKS

### 7.1 Multi-Condition Benchmark (45 Synthetic & Real Acoustic Scenarios)
Evaluated across 45 conditions ($5\text{ LibriSpeech speakers} \times 3\text{ Noise Types} \times 3\text{ SNR levels} [0\text{ dB}, 5\text{ dB}, 10\text{ dB}]$).

```
TABLE 1: Multi-Condition Acoustic Benchmark (Mean +/- Standard Deviation Across 45 Test Items)
+-------------------------+--------------------+------------------+-----------------+-----------------+-----------------+
| System / Configuration  | Delta SI-SDR (dB)  | SI-SDR (dB)      | PESQ-Wideband   | DNSMOS Overall  | DNSMOS Backgrnd |
+-------------------------+--------------------+------------------+-----------------+-----------------+-----------------+
| Noisy Baseline (Input)  |        ---         |  5.69 +/- 4.15   |  2.23 +/- 0.76  |  2.74 +/- 0.65  |  3.19 +/- 0.88  |
| NoiseShield Default     |  +11.88 +/- 6.47   | 17.57 +/- 6.11   |  3.02 +/- 0.76  |  3.35 +/- 0.18  |  4.14 +/- 0.15  |
| NoiseShield (Atten 24dB)|  +11.38 +/- 5.69   | 17.06 +/- 5.46   |  3.07 +/- 0.71  |  3.29 +/- 0.17  |  4.03 +/- 0.23  |
| NoiseShield (Atten 18dB)|  +10.31 +/- 4.55   | 16.00 +/- 4.68   |  3.04 +/- 0.69  |  3.24 +/- 0.19  |  3.94 +/- 0.28  |
| NoiseShield (Mix 0.10)  |  +10.77 +/- 5.01   | 16.46 +/- 4.97   |  3.06 +/- 0.69  |  3.26 +/- 0.19  |  3.97 +/- 0.26  |
+-------------------------+--------------------+------------------+-----------------+-----------------+-----------------+
```

### 7.2 Specific Noise Profiles on Real Human Speech (Live Streaming Mode)
Evaluated on speech-active frames of real human speech:

```
TABLE 2: Performance Breakdown by Specific Acoustic Noise Category
+--------------------------------+---------------+--------------------+---------------+---------------+---------------+
| Noise Environment & Input SNR  | Mode          | Delta SI-SDR (dB)  | STOI Score    | PESQ Score    | CPU Frame Lat |
+--------------------------------+---------------+--------------------+---------------+---------------+---------------+
| Real Speech + White Noise 0 dB | Enhanced (RT) |      +6.68 dB      |     0.907     |     1.17      |    35.2 ms    |
| Real Speech + White Noise 5 dB | Enhanced (RT) |      +4.92 dB      |     0.947     |     1.30      |    29.5 ms    |
| Real Speech + White Noise 10dB | Enhanced (RT) |      +2.55 dB      |     0.967     |     1.40      |    31.0 ms    |
| Real Speech + Pink Noise 0 dB  | Enhanced (RT) |      +6.65 dB      |     0.926     |     1.25      |    25.8 ms    |
| Real Speech + Pink Noise 5 dB  | Enhanced (RT) |      +4.88 dB      |     0.953     |     1.41      |    30.6 ms    |
| Real Speech + Pink Noise 10 dB | Enhanced (RT) |      +1.69 dB      |     0.963     |     1.46      |    32.7 ms    |
| Real Speech + Powerline Hum 0dB| Enhanced (RT) |     +11.18 dB      |     0.950     |     1.48      |    29.3 ms    |
| Real Speech + Powerline Hum 5dB| Enhanced (RT) |      +7.07 dB      |     0.958     |     1.54      |    31.6 ms    |
| Real Speech + Same-Speaker 0dB | Enhanced (RT) |      +2.11 dB      |     0.784     |     1.14      |    31.2 ms    |
| Real Speech + Same-Speaker 5dB | Enhanced (RT) |      +2.35 dB      |     0.902     |     1.20      |    24.7 ms    |
| Real Speech + Same-Speaker 10dB| Enhanced (RT) |      +0.97 dB      |     0.950     |     1.38      |    24.3 ms    |
+--------------------------------+---------------+--------------------+---------------+---------------+---------------+
```

### 7.3 Authentic Unreferenced Field Recording (Continuous HVAC Noise)
Evaluated on an authentic field recording (`samples/real_noisy_speech.wav`):

```
TABLE 3: Suppression Metrics on Authentic Real-World Field Recording
+---------------------------+-----------------------------------+-----------------------+--------------------+----------------+
| Processing Implementation | Noise-Floor Reduction (Quiet 10%) | Speech-Level Change   | Envelope Tracing r | Frame Latency  |
+---------------------------+-----------------------------------+-----------------------+--------------------+----------------+
| Baseline (Raw Recording)  |       -49.98 dBFS (Unfiltered)    |       0.00 dB         |       1.0000       |      ---       |
| Offline Batch Enhancement |       -77.65 dBFS (+27.68 dB)     |      -0.74 dB         |       0.9707       |      ---       |
| Live Real-Time Streaming  |       -80.22 dBFS (+30.25 dB)     |      -1.32 dB         |       0.9581       |    34.4 ms     |
+---------------------------+-----------------------------------+-----------------------+--------------------+----------------+
```
- **Suppression:** Real-time neural streaming pushed background noise down by **+30.25 dB**, rendering air conditioning rumble completely inaudible.
- **Speech Preservation:** Active speech envelope correlation remained extremely high at **0.9581**, with speech level attenuated by only $1.32\text{ dB}$, confirming clean preservation without clipping.

### 7.4 Latency & Computational Concurrency Benchmarks
Benchmarked using `src/call/benchmark_concurrency.py` on an x86_64 CPU ($40.0\text{ ms}$ budget):

```
TABLE 4: Real-Time Factor (RTF) & Latency Profiling
+--------------------------------------------------+---------------+---------------+---------------+---------------+---------------+
| Execution Workload Scenario                      | Average (ms)  | Median (ms)   | 95th %ile(ms) | Frame Budget% | Real-Time Fac |
+--------------------------------------------------+---------------+---------------+---------------+---------------+---------------+
| 1 Active Audio Stream (200 ms Context Buffer)    |   24.62 ms    |   24.18 ms    |   27.80 ms    |     61.5%     |     0.615     |
| 1 Active Audio Stream (120 ms Short Context)     |   21.84 ms    |   21.30 ms    |   24.10 ms    |     54.6%     |     0.546     |
| 2 Concurrent Active Streams (Dual Voice Call)    |   38.10 ms*   |   37.85 ms    |   41.20 ms    |   47.6%/frame |     0.476     |
| Near-Silent Frame (Inference Bypassed, RMS < 20) |    0.012 ms   |    0.010 ms   |    0.015 ms   |    < 0.1%     |    < 0.001    |
+--------------------------------------------------+---------------+---------------+---------------+---------------+---------------+
*Note: 38.10 ms represents the aggregate turnaround for both streams processed via ThreadPoolExecutor.
```
- **Thread Pinning Benefit:** Explicitly enforcing `torch.set_num_threads(1)` eliminated intra-op OpenMP thread contention across concurrent workers, reducing 95th percentile latency jitter by $38\%$.

---

# 8. ACOUSTIC & PERCEPTUAL DISCUSSIONS (ANOMALIES, ROOM TONE & NATURALNESS)

### 8.1 The "Dead-Silence Fatigue" Paradox & Naturalness Controls
In real voice communication, complete mathematical elimination of background noise (reducing ambient noise to absolute digital zero, $- \infty\text{ dBFS}$) creates **Dead-Silence Fatigue**. Human callers unconsciously interpret the total absence of acoustic room reflections as a dropped call or frozen connection, leading to repetitive checking ("Can you hear me?").
- **Solution:** NoiseShield implements Attenuation Limiting (`atten_lim_db = 18 dB` or `24 dB`) and Dry-Mix Blending (`dry_mix = 0.10`).
- **Empirical Finding:** As shown in Table 1, setting `atten_lim_db = 24 dB` retains high intelligibility ($\text{PESQ} = 3.07$, $\Delta\text{SI-SDR} = +11.38\text{ dB}$) while providing a quiet, natural acoustic floor that eliminates dead-silence anxiety.

### 8.2 The Clean Reference Room-Tone Artifact in Objective PESQ Scoring
During initial benchmarks on `real_clean_speech_48k.wav`, PESQ scores for enhanced speech occasionally fell below the noisy baseline (e.g., pink noise at 10 dB). 
- **Forensic Diagnosis:** The target reference file `real_clean_speech_48k.wav` was recorded in an authentic domestic room containing ambient room tone ($-44.3\text{ dBFS}$ leading, $-50.9\text{ dBFS}$ during pauses). 
- **Perceptual Metric Bias:** DeepFilterNet effectively suppressed this ambient room tone. However, PESQ performs strict psychoacoustic time-frequency alignment against the imperfect reference. PESQ penalized the removal of reference room tone as a "distortion," artificially degrading the score.
- **Verification:** When re-evaluated against studio-clean anechoic LibriSpeech references, enhanced PESQ consistently outperformed the noisy baseline across all 9 conditions ($+0.52\text{ to } +1.42\text{ MOS}$ improvement), verifying that the discrepancy was reference-noise bias rather than an algorithmic flaw.

### 8.3 Unexplained Phenomenon: Real-Time vs. Offline on Low-Frequency & Babble Signals
- **Observation:** On Pink Noise ($0\text{ dB}$: $+6.99\text{ dB}$ RT vs $+6.34\text{ dB}$ Offline) and Same-Speaker Babble ($0\text{ dB}$: $+3.58\text{ dB}$ RT vs $+1.32\text{ dB}$ Offline; $5\text{ dB}$: $+2.85\text{ dB}$ RT vs $+0.50\text{ dB}$ Offline), real-time streaming achieved higher $\Delta\text{SNR}$ and $\Delta\text{SI-SDR}$ than offline batch enhancement.
- **Scientific Status:** *[Unexplained Observation]* — We hypothesize that sliding-window context re-priming ($200\text{ ms}$ history) continually shifts the recurrent state, preventing over-adaptation to stationary assumptions and allowing the complex linear filter to react more aggressively to speech-like interferers. This provides a compelling topic for future internal state probing.

---

# 9. COMPLETE LATEX EQUATION BANK & FORMULATIONS

```latex
% Signal Degradation Model
y(t) = s(t) * h(t) + n(t)
Y(k, m) = S(k, m) + N(k, m) = |Y(k, m)| e^{j \angle Y(k, m)}

% Equivalent Rectangular Bandwidth (Glasberg & Moore, 1990)
\text{ERB}(f) = 24.7 \left( 4.37 \cdot \frac{f}{1000} + 1 \right)
G_{\text{ERB}}(f) = 21.4 \log_{10} \left( 0.00437 f + 1 \right)
E(b, m) = \sum_{k=0}^{K-1} W(b, k) |Y(k, m)|^2, \quad b \in \{0, 1, \dots, 31\}

% Complex Deep Filtering Formulation
\hat{S}_{\text{DF}}(k, m) = \sum_{l=0}^{L_p} C(k, m, l)^* \cdot Y(k, m - l)

% Cartesian Decomposition of Complex Linear Prediction
C = C_r + j C_i, \quad Y = Y_r + j Y_i
C^* \cdot Y = (C_r Y_r + C_i Y_i) + j (C_r Y_i - C_i Y_r)

% Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
\alpha = \frac{\hat{s}^T s}{\|s\|^2}
e_{\text{target}} = \alpha s, \quad e_{\text{res}} = \hat{s} - \alpha s
\text{SI-SDR} = 10 \log_{10} \left( \frac{\|e_{\text{target}}\|^2}{\|e_{\text{res}}\|^2} \right)

% Multi-Resolution Power-Law Loss Function
\mathcal{L}_{\text{spec}} = \frac{1}{M \cdot K} \sum_{m=1}^{M} \sum_{k=1}^{K} \left| |S(k, m)|^{0.3} - |\hat{S}(k, m)|^{0.3} \right|^2

% Real-Time Factor (RTF)
\text{RTF} = \frac{T_{\text{proc}}}{T_{\text{frame}}} = \frac{T_{\text{proc}}}{40.0\text{ ms}}

% Dry-Mix Blending Formulation
x_{\text{out}}(t) = (1 - \alpha) \cdot \hat{s}(t) + \alpha \cdot y(t), \quad \alpha \in [0.0, 1.0]
```

---

# 10. COMPLETE BIBTEX REFERENCE LIBRARY

```bibtex
@article{schroter2023deepfilternet,
  title={{DeepFilterNet}: A Low-Complexity Speech Enhancement Framework for Full-Band Audio Based on Deep Filtering},
  author={Schr{\"o}ter, Hendrik and Rosenkranz, Tobias and Escalante-B., Alberto N. and Maier, Andreas},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={31},
  pages={1024--1035},
  year={2023},
  publisher={IEEE},
  doi={10.1109/TASLP.2023.3243936}
}

@inproceedings{schroter2022deepfilternet2,
  title={{DeepFilterNet2}: Towards Real-Time Speech Enhancement on Embedded Platforms for Full-Band Audio},
  author={Schr{\"o}ter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
  booktitle={Proc. Interspeech 2022},
  pages={4117--4121},
  year={2022},
  doi={10.21437/Interspeech.2022-10825}
}

@article{boll1979suppression,
  title={Suppression of acoustic noise in speech using spectral subtraction},
  author={Boll, Steven F.},
  journal={IEEE Transactions on Acoustics, Speech, and Signal Processing},
  volume={27},
  number={2},
  pages={113--120},
  year={1979},
  doi={10.1109/TASSP.1979.1163209}
}

@article{ephraim1984speech,
  title={Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator},
  author={Ephraim, Yariv and Malah, David},
  journal={IEEE Transactions on Acoustics, Speech, and Signal Processing},
  volume={32},
  number={6},
  pages={1109--1121},
  year={1984},
  doi={10.1109/TASSP.1984.1164453}
}

@inproceedings{valin2018hybrid,
  title={A hybrid {DSP}/deep learning approach to real-time full-band speech enhancement},
  author={Valin, Jean-Marc},
  booktitle={2018 IEEE 20th International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--5},
  year={2018},
  organization={IEEE},
  doi={10.1109/MMSP.2018.8547078}
}

@inproceedings{rix2001perceptual,
  title={Perceptual evaluation of speech quality ({PESQ})---a new method for speech quality assessment of telephone networks and codecs},
  author={Rix, Antony W. and Beerends, John G. and Hollier, Michael P. and Hekstra, Andries P.},
  booktitle={2001 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  volume={2},
  pages={749--752},
  year={2001},
  organization={IEEE},
  doi={10.1109/ICASSP.2001.941023}
}

@article{taal2011algorithm,
  title={An algorithm for estimating the intelligibility of speech signals from short-time objective intelligibility measure ({STOI})},
  author={Taal, Cees H. and Hendriks, Richard C. and Heusdens, Richard and Jensen, Jesper},
  journal={IEEE Transactions on Audio, Speech, and Language Processing},
  volume={19},
  number={5},
  pages={1125--1136},
  year={2011},
  doi={10.1109/TASL.2010.2076594}
}

@inproceedings{reddy2022dnsmos,
  title={{DNSMOS P.835}: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors},
  author={Reddy, Chandan KA and Gopal, Vishak and Cutler, Ross},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={886--890},
  year={2022},
  organization={IEEE},
  doi={10.1109/ICASSP43922.2022.9746387}
}

@misc{itu2007p8622,
  title={{Recommendation P.862.2}: Wideband extension to Recommendation P.862 for the assessment of wideband speech codecs},
  author={{International Telecommunication Union}},
  year={2007},
  publisher={ITU-T}
}

@misc{w3c2021webaudio,
  title={{Web Audio API}: {W3C} Recommendation},
  author={{World Wide Web Consortium}},
  year={2021},
  howpublished={\url{https://www.w3.org/TR/webaudio/}}
}
```

---

# 11. TURN-BY-TURN LLM PROMPTING PLAYBOOK

When uploading or pasting this dossier into ChatGPT, Claude 3.5 Sonnet, or Gemini 1.5 Pro, use the following structured prompt sequences to generate the academic paper section by section.

### Turn 1: System Persona & Architecture Ingestion
```text
PROMPT:
You are a Senior Principal Research Scientist and IEEE Fellow specializing in Neural Audio Signal Processing, Speech Communications, and Real-Time Systems Engineering. 
I have uploaded the comprehensive technical dossier of our project, NoiseShield: Full-Duplex Neural Speech Enhancement at 48 kHz for Live Web Telephony.
Carefully review the entire document, including the theoretical derivations, system architecture, forensic bug analyses, and empirical benchmark tables across 45 acoustic conditions.
Please acknowledge ingestion by outlining the 5 core scientific and engineering contributions of this project that make it suitable for publication in IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP).
```

### Turn 2: Abstract & Introduction
```text
PROMPT:
Using the dossier, write Section 1: "Introduction & Motivation" and Section 2: "Background & Prior Art" in high-level, rigorous academic LaTeX. 
Key requirements:
1. Ground the motivation in Cognitive Load Theory and psychoacoustic listening fatigue ("Zoom fatigue").
2. Clearly contrast fullband 48 kHz audio against historical 16 kHz narrowband limitations.
3. Formally analyze the failure modes of classical DSP (spectral subtraction, Wiener filtering, phase distortion).
4. Explicitly list our 4 main scientific and systems contributions with bold bullet points.
```

### Turn 3: Theoretical Formulation of Complex Deep Filtering
```text
PROMPT:
Draft Section 3: "Mathematical Formulation & Deep Filtering Theory".
Key requirements:
1. Formulate the signal model in the STFT domain.
2. Provide the complete Glasberg & Moore (1990) Equivalent Rectangular Bandwidth (ERB) derivations, showing how the 48 kHz spectrum is compressed into 32 bands.
3. Detail the Complex Deep Filtering equations, showing real and imaginary Cartesian decomposition and explaining how complex linear prediction reconstructs both magnitude and phase.
4. Detail the multi-resolution power-law compressed loss function. Provide all LaTeX equations cleanly formatted.
```

### Turn 4: Systems Engineering & Production Post-Mortem
```text
PROMPT:
Draft Section 4: "Full-Duplex Web Telephony Architecture & Engineering Forensics".
Key requirements:
1. Detail the client-side AudioWorklet architecture (CaptureProcessor and PlaybackProcessor circular ring buffer).
2. Document the custom 12-byte / 16-byte Big-Endian binary framing protocol over WebSockets.
3. Detail the StreamDenoiser stateful context re-priming mechanism (200 ms rolling buffer) and why stateless execution causes click artifacts.
4. Convert Section 5 of the dossier (the forensic post-mortem of the 6 production defects) into an insightful "Systems Engineering & Failure Mode Analysis" subsection, explaining how endianness drift, DataView boundaries, worklet lifecycle, and silence zeroing were diagnosed and solved.
```

### Turn 5: Quantitative Results, Tables & Acoustic Discussion
```text
PROMPT:
Draft Section 5: "Experimental Evaluation & Empirical Findings" and Section 6: "Acoustic Discussion & Ablation Study".
Key requirements:
1. Format Table 1 (45-condition benchmark), Table 2 (noise categories), Table 3 (authentic HVAC field recording), and Table 4 (latency/RTF) into pristine LaTeX table syntax.
2. Analyze the SI-SDR (+11.88 dB), PESQ (3.02), and DNSMOS BAK (4.14) improvements.
3. Provide an in-depth discussion on:
   a) The "Dead-Silence Fatigue" paradox and naturalness control (attenuation limits and dry mix).
   b) The reference room-tone artifact in objective PESQ evaluation.
   c) The unexplained phenomenon where real-time streaming outperformed offline batch on pink noise and same-speaker babble.
```

### Turn 6: Conclusion, Future Work & BibTeX Generation
```text
PROMPT:
Draft Section 7: "Future Directions & Open Research Problems" and Section 8: "Conclusion". Then, format the complete paper references into standard IEEEtran BibTeX format matching all cited papers in the dossier.
```

---

# 12. PROJECT CODEBASE & FILE MANIFEST

```
e:\bnsup\
├── docs\
│   ├── RESEARCH_PAPER_PROJECT_DOSSIER.md    <-- Master Academic Paper Dossier (This Document)
│   ├── noiseshield_comprehensive_research_report.md <-- Extended technical report
│   ├── eval_spec_powerline_hum_0dB.png      <-- Powerline hum spectrogram analysis
│   ├── eval_spec_real_recording_noref.png   <-- Authentic HVAC suppression spectrogram
│   └── eval_spec_white_noise_0dB.png        <-- White noise suppression spectrogram
├── src\
│   ├── call\
│   │   ├── call_server.py                   <-- Core FastAPI server, WebSockets, room signaling
│   │   ├── database.py                      <-- SQLite user presence & call duration persistence
│   │   ├── benchmark_concurrency.py         <-- RTF and multi-stream latency benchmarking
│   │   ├── static\
│   │   │   ├── app.js                       <-- UI state machine, binary serializer, oscilloscope
│   │   │   ├── worklets.js                  <-- CaptureProcessor & PlaybackProcessor AudioWorklets
│   │   │   └── index.html                   <-- Modern dark-mode telecommunication studio UI
│   │   ├── test_features.py                 <-- Automated feature & regression test suite
│   │   ├── test_room_relay.py               <-- End-to-end binary WebSocket relay validation
│   │   └── test_browser_room_call.py        <-- Headless Playwright multi-browser call validation
│   ├── realtime\
│   │   └── server.py                        <-- StreamDenoiser rolling context buffer manager
│   ├── denoise.py                           <-- DeepFilterNet3 loader, batch processing & atten limits
│   ├── evaluate.py                          <-- Multi-metric evaluation engine (PESQ, STOI, SI-SDR)
│   └── tradeoff.py                          <-- Attenuation limit and dry-mix Pareto analysis
├── outputs\
│   ├── evaluation_results.md                <-- Raw empirical test scores & markdown tables
│   └── noiseshield.db                       <-- Call session & presence database
├── samples\
│   ├── real_clean_speech_48k.wav            <-- 48 kHz clean speech reference
│   └── real_noisy_speech.wav                <-- 48 kHz authentic HVAC field recording
├── requirements.txt                         <-- PyTorch, DeepFilterNet, FastAPI, librosa, pesq, pystoi
└── run_all.ps1                              <-- Comprehensive automation & verification script
```

---
*End of Dossier.*
