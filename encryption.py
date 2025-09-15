import numpy as np
import zlib
import hashlib
import hmac
import struct
import random
import pickle
from scipy.fft import dct

class EncryptionSystem:
    def __init__(self, k_master=b'master_key_32_bytes_long_12345678'):
        self.k_master = k_master
        self.frame_counter = 0
        self.audio_counter = 0
        
        # Load Lorenz states
        try:
            with open('lorenz_states.pkl', 'rb') as f:
                self.lorenz_states = pickle.load(f)
            print(f"Loaded {len(self.lorenz_states)} Lorenz states for encryption")
        except FileNotFoundError:
            print("Warning: lorenz_states.pkl not found. Run lorenz.py first!")
            self.lorenz_states = None
    
    def fisher_yates_sbox(self, seed):
        """Generate S-box using Fisher-Yates shuffle with given seed"""
        random.seed(seed)
        sbox = list(range(256))
        for i in range(255, 0, -1):
            j = random.randint(0, i)
            sbox[i], sbox[j] = sbox[j], sbox[i]
        return sbox
    
    def sha256_counter_keystream(self, key, counter, length):
        """Generate keystream using SHA256 in counter mode"""
        keystream = b''
        block_size = 32  # SHA256 output size
        blocks_needed = (length + block_size - 1) // block_size
        
        for i in range(blocks_needed):
            counter_bytes = struct.pack('>Q', counter + i)
            hash_input = key + counter_bytes
            block = hashlib.sha256(hash_input).digest()
            keystream += block
        
        return keystream[:length]
    
    def encrypt_frame(self, frame_data):
        """Encrypt video frame using Lorenz-based encryption"""
        if self.lorenz_states is None:
            raise ValueError("Lorenz states not loaded")
        
        # Compress frame data
        compressed = zlib.compress(frame_data, level=6)
        
        # Get Lorenz state for this frame
        lorenz_state = self.lorenz_states[self.frame_counter % len(self.lorenz_states)]
        
        # Generate frame seed using Lorenz state and master key
        lorenz_bytes = struct.pack('>ddd', lorenz_state[0], lorenz_state[1], lorenz_state[2])
        frame_seed_hash = hashlib.sha256(self.k_master + lorenz_bytes + struct.pack('>I', self.frame_counter)).digest()
        frame_seed = struct.unpack('>I', frame_seed_hash[:4])[0]
        
        # Generate S-box using Fisher-Yates
        sbox = self.fisher_yates_sbox(frame_seed)
        
        # Generate nonce and keystream
        frame_nonce = struct.pack('>Q', self.frame_counter)
        keystream = self.sha256_counter_keystream(self.k_master + frame_nonce, 0, len(compressed))
        
        # Encrypt with S-box, XOR, and feedback chain
        encrypted = bytearray()
        prev_byte = 0x5A  # Fixed initialization vector
        
        for i, byte in enumerate(compressed):
            # S-box substitution
            sub_byte = sbox[byte]
            # XOR with keystream
            xor_byte = sub_byte ^ keystream[i % len(keystream)]
            # Feedback chain
            chain_byte = xor_byte ^ prev_byte
            encrypted.append(chain_byte)
            prev_byte = chain_byte
        
        # Generate HMAC for authentication
        auth_data = struct.pack('>I', self.frame_counter) + frame_nonce + bytes(encrypted)
        auth_tag = hmac.new(self.k_master, auth_data, hashlib.sha256).digest()
        
        result = {
            'ciphertext': bytes(encrypted),
            'frame_no': self.frame_counter,
            'frame_nonce': frame_nonce,
            'auth_tag': auth_tag
        }
        
        self.frame_counter += 1
        return result
    
    def encrypt_audio(self, audio_data):
        """Encrypt audio block using Lorenz-based encryption with DCT pipeline"""
        if self.lorenz_states is None:
            raise ValueError("Lorenz states not loaded")
        
        # Step 1: Convert audio to float and apply DCT
        audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float64)
        dct_data = dct(audio_float, type=2)
        print(f"Audio {self.audio_counter}: DCT applied, size: {len(dct_data)}")
        
        # Step 2: Get Lorenz state for this audio block
        lorenz_state = self.lorenz_states[self.audio_counter % len(self.lorenz_states)]
        
        # Step 3: Mix DCT data with Lorenz state
        mixed_data = dct_data + lorenz_state[0] * 0.1  # Small mixing factor
        
        # Step 4: Generate audio seed (different from video)
        lorenz_bytes = struct.pack('>ddd', lorenz_state[0], lorenz_state[1], lorenz_state[2])
        seed_input = self.k_master[4:8] + lorenz_bytes + struct.pack('>I', self.audio_counter)  # Different key slice
        audio_seed = struct.unpack('>I', hashlib.sha256(seed_input).digest()[:4])[0]
        
        # Step 5: Generate S-box
        sbox = self.fisher_yates_sbox(audio_seed)
        
        # Step 6: Convert mixed data to bytes for encryption
        audio_bytes = mixed_data.astype(np.float32).tobytes()
        audio_nonce = struct.pack('>Q', self.audio_counter)
        keystream = self.sha256_counter_keystream(self.k_master + audio_nonce, 0, len(audio_bytes))
        
        # Step 7: Encrypt with different pipeline for audio: XOR -> S-box -> Feedback
        encrypted = bytearray()
        prev_byte = 0xA5  # Different IV for audio
        
        for i, byte in enumerate(audio_bytes):
            # XOR with keystream first (different order than video)
            xor_byte = byte ^ keystream[i % len(keystream)]
            # S-box substitution
            sub_byte = sbox[xor_byte]
            # Feedback chain
            chain_byte = sub_byte ^ prev_byte
            encrypted.append(chain_byte)
            prev_byte = chain_byte
        
        # Step 8: Generate HMAC
        auth_data = struct.pack('>I', self.audio_counter) + audio_nonce + bytes(encrypted)
        auth_tag = hmac.new(self.k_master, auth_data, hashlib.sha256).digest()
        
        result = {
            'ciphertext': bytes(encrypted),
            'frame_no': self.audio_counter,
            'frame_nonce': audio_nonce,
            'auth_tag': auth_tag,
            'lorenz_state': lorenz_state,  # Include for debugging/saving
            'original_size': len(audio_data)
        }
        
        self.audio_counter += 1
        return result
