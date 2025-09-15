import numpy as np
import zlib
import hashlib
import hmac
import struct
import random
import pickle
from scipy.fft import idct

class DecryptionSystem:
    def __init__(self, k_master=b'master_key_32_bytes_long_12345678'):
        self.k_master = k_master
        
        # Load the same Lorenz states as encryption
        try:
            with open('lorenz_states.pkl', 'rb') as f:
                self.lorenz_states = pickle.load(f)
            print(f"Loaded {len(self.lorenz_states)} Lorenz states for decryption")
        except FileNotFoundError:
            print("Error: lorenz_states.pkl not found. Run lorenz.py first!")
            self.lorenz_states = None
    
    def fisher_yates_sbox(self, seed):
        """Generate S-box using Fisher-Yates shuffle with given seed"""
        random.seed(seed)
        sbox = list(range(256))
        for i in range(255, 0, -1):
            j = random.randint(0, i)
            sbox[i], sbox[j] = sbox[j], sbox[i]
        return sbox
    
    def inverse_sbox(self, sbox):
        """Generate inverse S-box"""
        inv_sbox = [0] * 256
        for i in range(256):
            inv_sbox[sbox[i]] = i
        return inv_sbox
    
    def sha256_counter_keystream(self, key, counter, length):
        """Generate keystream using SHA256 in counter mode"""
        keystream = b''
        block_size = 32
        blocks_needed = (length + block_size - 1) // block_size
        
        for i in range(blocks_needed):
            counter_bytes = struct.pack('>Q', counter + i)
            hash_input = key + counter_bytes
            block = hashlib.sha256(hash_input).digest()
            keystream += block
        
        return keystream[:length]
    
    def decrypt_frame(self, encrypted_packet):
        """Decrypt video frame"""
        if self.lorenz_states is None:
            raise ValueError("Lorenz states not loaded")
        
        ciphertext = encrypted_packet['ciphertext']
        frame_no = encrypted_packet['frame_no']
        frame_nonce = encrypted_packet['frame_nonce']
        auth_tag = encrypted_packet['auth_tag']
        
        # Verify HMAC first
        auth_data = struct.pack('>I', frame_no) + frame_nonce + ciphertext
        expected_tag = hmac.new(self.k_master, auth_data, hashlib.sha256).digest()
        
        if not hmac.compare_digest(auth_tag, expected_tag):
            raise ValueError("Frame authentication failed")
        
        # Get same Lorenz state as encryption
        lorenz_state = self.lorenz_states[frame_no % len(self.lorenz_states)]
        
        # Generate same frame seed
        lorenz_bytes = struct.pack('>ddd', lorenz_state[0], lorenz_state[1], lorenz_state[2])
        frame_seed_hash = hashlib.sha256(self.k_master + lorenz_bytes + struct.pack('>I', frame_no)).digest()
        frame_seed = struct.unpack('>I', frame_seed_hash[:4])[0]
        
        # Generate same S-box and inverse
        sbox = self.fisher_yates_sbox(frame_seed)
        inv_sbox = self.inverse_sbox(sbox)
        
        # Generate same keystream
        keystream = self.sha256_counter_keystream(self.k_master + frame_nonce, 0, len(ciphertext))
        
        # Decrypt by reversing the encryption process
        decrypted = bytearray()
        prev_byte = 0x5A  # Same IV as encryption
        
        for i, byte in enumerate(ciphertext):
            # Reverse feedback chain
            chain_byte = byte ^ prev_byte
            # Reverse XOR with keystream
            xor_byte = chain_byte ^ keystream[i % len(keystream)]
            # Reverse S-box substitution
            original_byte = inv_sbox[xor_byte]
            decrypted.append(original_byte)
            prev_byte = byte
        
        # Decompress
        decompressed = zlib.decompress(bytes(decrypted))
        return decompressed
    
    def decrypt_audio(self, encrypted_packet):
        """Decrypt audio block following exact reverse pipeline"""
        if self.lorenz_states is None:
            raise ValueError("Lorenz states not loaded")
        
        ciphertext = encrypted_packet['ciphertext']
        frame_no = encrypted_packet['frame_no']
        frame_nonce = encrypted_packet['frame_nonce']
        auth_tag = encrypted_packet['auth_tag']
        
        # Step 1: Verify HMAC first
        auth_data = struct.pack('>I', frame_no) + frame_nonce + ciphertext
        expected_tag = hmac.new(self.k_master, auth_data, hashlib.sha256).digest()
        
        if not hmac.compare_digest(auth_tag, expected_tag):
            raise ValueError("Audio authentication failed")
        
        # Step 2: Get same Lorenz state as encryption
        lorenz_state = self.lorenz_states[frame_no % len(self.lorenz_states)]
        
        # Step 3: Generate same audio seed (different from video)
        lorenz_bytes = struct.pack('>ddd', lorenz_state[0], lorenz_state[1], lorenz_state[2])
        seed_input = self.k_master[4:8] + lorenz_bytes + struct.pack('>I', frame_no)
        audio_seed = struct.unpack('>I', hashlib.sha256(seed_input).digest()[:4])[0]
        
        # Step 4: Generate same S-box and inverse
        sbox = self.fisher_yates_sbox(audio_seed)
        inv_sbox = self.inverse_sbox(sbox)
        
        # Step 5: Generate same keystream
        keystream = self.sha256_counter_keystream(self.k_master + frame_nonce, 0, len(ciphertext))
        
        # Step 6: Decrypt by reversing: Feedback -> Inverse S-box -> XOR
        decrypted = bytearray()
        prev_byte = 0xA5  # Same IV as audio encryption
        
        for i, byte in enumerate(ciphertext):
            # Reverse feedback chain
            chain_byte = byte ^ prev_byte
            # Reverse S-box substitution
            xor_byte = inv_sbox[chain_byte]
            # Reverse XOR with keystream
            original_byte = xor_byte ^ keystream[i % len(keystream)]
            decrypted.append(original_byte)
            prev_byte = byte  # Use original ciphertext byte for next iteration
        
        # Step 7: Convert back to float64 and reverse Lorenz mixing
        try:
            mixed_data = np.frombuffer(bytes(decrypted), dtype=np.float32).astype(np.float64)
            dct_data = mixed_data - lorenz_state[0] * 0.1  # Reverse the mixing
            
            # Step 8: Apply inverse DCT
            audio_float = idct(dct_data, type=2)
            
            # Step 9: Convert back to int16
            audio_int16 = np.clip(audio_float, -32768, 32767).astype(np.int16)
            
            print(f"Audio {frame_no}: Decrypted successfully")
            return audio_int16.tobytes()
        except Exception as e:
            raise ValueError(f"Audio reconstruction failed for block {frame_no}: {e}")
