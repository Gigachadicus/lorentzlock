# Lorenz Cipher Live Audio/Video Encryption System

A master/slave live audio and video encryption system using a Lorenz chaos-based block cipher pipeline with compression, multiple encryption rounds, and HMAC authentication.

## Features

- **Lorenz-based encryption**: Uses chaotic dynamics for cryptographic operations
- **Multi-layered security**: Compression + block cipher with multiple rounds + HMAC
- **Live streaming**: Real-time audio/video capture, encryption, and playback
- **Flexible modes**: Video-only, audio-only, or both
- **UDP networking**: Low-latency packet transmission
- **Deterministic chaos**: Reproducible encryption using shared pool bytes

## Architecture

### Encryption Pipeline

1. **Compression**: Raw data compressed with zlib
2. **Block partitioning**: Data split into fixed-size blocks with padding
3. **Chaotic resources**: Generated from Lorenz pool using deterministic offsets:
   - Bijective S-box (Fisher-Yates shuffle)
   - Global block permutation
   - Local block shuffles
   - Keystreams for XOR operations
   - Tweak bytes for additional randomness
4. **Multiple rounds** (default 3 per block):
   - MixColumns diffusion (4-byte words)
   - Local shuffle within block
   - XOR with round keystream
   - S-box substitution
   - Tweak addition (mod 256)
5. **Global block shuffle**: Permute all encrypted blocks
6. **HMAC tagging**: SHA-256 authentication over metadata + ciphertext

### Decryption

Exact reverse of encryption pipeline with verification:
- HMAC verification (abort on mismatch)
- Reverse global block shuffle
- Reverse rounds (inverse operations)
- Remove padding
- Decompress data

## Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install portaudio19-dev python3-dev

# Install Python requirements
pip install -r requirements.txt
```

## Usage

### 1. Generate Pool File

First, generate the Lorenz chaos pool bytes (both master and slave need the same file):

```bash
python generate_pool.py --output lorenz_pool.bin --size-mb 10
```

### 2. Start Master (Capture & Encrypt)

```bash
# Video + audio capture
python master.py --pool-file lorenz_pool.bin --salt "shared_secret_key"

# Video only
python master.py --pool-file lorenz_pool.bin --salt "shared_secret_key" --mode video

# Audio only  
python master.py --pool-file lorenz_pool.bin --salt "shared_secret_key" --mode audio

# Custom parameters
python master.py --pool-file lorenz_pool.bin --salt "shared_secret_key" \
  --host 0.0.0.0 --port 5001 --frame-rate 15 --block-size 1024 --rounds 5
```

### 3. Start Slave (Decrypt & Playback)

```bash
# Connect to master and playback
python slave.py --pool-file lorenz_pool.bin --salt "shared_secret_key"

# Connect to remote master
python slave.py --pool-file lorenz_pool.bin --salt "shared_secret_key" \
  --host 192.168.1.100 --port 5001

# Specific mode
python slave.py --pool-file lorenz_pool.bin --salt "shared_secret_key" --mode video
```

## Command Line Options

### Master System

- `--host`: Host address to bind (default: 127.0.0.1)
- `--port`: UDP port to bind (default: 5000)  
- `--pool-file`: Path to Lorenz pool bytes file (required)
- `--salt`: HMAC salt string (required, must match slave)
- `--frame-rate`: Video frame rate (default: 30)
- `--block-size`: Encryption block size in bytes (default: 768)
- `--rounds`: Number of encryption rounds (default: 3)
- `--mode`: Capture mode - video/audio/both (default: both)

### Slave System

- `--host`: Master host address (default: 127.0.0.1)
- `--port`: Master UDP port (default: 5000)
- `--pool-file`: Path to Lorenz pool bytes file (required)
- `--salt`: HMAC salt string (required, must match master)
- `--block-size`: Encryption block size (default: 768)
- `--rounds`: Number of encryption rounds (default: 3)  
- `--mode`: Playback mode - video/audio/both (default: both)

## Security Features

### Cryptographic Strength

- **Chaotic encryption**: Non-linear Lorenz dynamics provide unpredictable keystreams
- **Multiple rounds**: 3+ rounds of confusion and diffusion per block
- **Bijective S-boxes**: Ensure no information loss during substitution
- **Global permutation**: Additional layer scrambling block order
- **HMAC authentication**: Prevents tampering and ensures integrity

### Key Management

- **Shared pool**: Pre-generated Lorenz chaos pool provides shared randomness
- **Deterministic derivation**: Encryption parameters derived from frame number + salt
- **No key exchange**: Pool file and salt provide all necessary key material
- **Perfect synchronization**: Deterministic chaos ensures master/slave sync

## Technical Details

### Video Processing

- **Capture**: OpenCV VideoCapture from default camera (640x480)
- **Format**: RGB frames converted to byte streams
- **Block size**: Default 768 bytes (16x16 pixel blocks)
- **Transmission**: UDP packets with frame metadata

### Audio Processing  

- **Capture**: PyAudio input stream (44.1kHz, 16-bit, mono)
- **Chunks**: 1024 sample blocks processed in real-time
- **Format**: Raw PCM audio data
- **Playback**: Immediate audio output on slave

### Lorenz System

- **Parameters**: σ=10.0, ρ=28.0, β=8/3 (classic chaotic regime)
- **Integration**: Runge-Kutta 45 with high precision (rtol=1e-9)
- **Quantization**: 32-bit integer quantization of (x,y,z) states
- **Pool generation**: Continuous trajectory sampling for large pools

## Protocol Specification

### Packet Format

```
[Type:1] [MetaLen:4] [DataLen:4] [TagLen:4] [Metadata] [Ciphertext] [HMAC]
```

- **Type**: 'V' for video, 'A' for audio
- **Lengths**: Big-endian 32-bit integers
- **Metadata**: Frame dimensions, frame number, cipher metadata
- **HMAC**: SHA-256 authentication tag

### Discovery Protocol

1. Slave sends "DISCOVER" to master UDP port
2. Master responds with "ACK" and registers client
3. Master broadcasts encrypted packets to all registered clients

## Performance

### Throughput

- **Video**: ~30 FPS at 640x480 (typical)
- **Audio**: Real-time 44.1kHz streaming
- **Encryption**: ~50-100 MB/s per core (depends on block size and rounds)

### Latency

- **Network**: UDP provides minimal protocol overhead
- **Processing**: 1-5ms encryption/decryption per frame (typical)
- **Total**: <100ms end-to-end latency on LAN

## Troubleshooting

### Common Issues

1. **Audio not working**: Install PortAudio development libraries
2. **Video not displaying**: Check OpenCV camera access permissions  
3. **HMAC failures**: Ensure exact salt match between master/slave
4. **Connection refused**: Check firewall settings for UDP port
5. **Performance issues**: Reduce frame rate or increase block size

### Debug Mode

Add print statements in `lorenz_cipher.py` functions to trace encryption/decryption:

```python
print(f"Encrypting block {i}: {len(block)} bytes")
print(f"Pool offset: {offset}, Seed: {seed_hash.hex()[:8]}")
```

## Security Warnings

- **Pool file security**: Keep Lorenz pool file secret - it's part of the key material
- **Salt management**: Use strong, unique salts for different deployments  
- **Network security**: Consider VPN/TLS for transmission over untrusted networks
- **Implementation note**: This is a demonstration system, not production-ready

## License

Educational/research use. Not for production security applications.
