# cipher_system.py - Chaotic Video Encryption Pipeline
import hashlib
import hmac
import struct
import numpy as np
from typing import Tuple, List
from scipy.fft import dct, idct


def derive_seed_hash(frame_no: int, salt: bytes, final_lorenz_state: Tuple[float, float, float] = None) -> bytes:
    """Derive seed hash using Lorenz final state + frame number + salt"""
    if final_lorenz_state is None:
        # Default final state if not provided
        final_lorenz_state = (1.0, 1.0, 1.0)
    
    x, y, z = final_lorenz_state
    # Pack Lorenz state and frame number
    lorenz_bytes = struct.pack('>ddd', x, y, z)  # double precision
    frame_bytes = struct.pack('>Q', frame_no)
    
    # SHA256(X_last || Y_last || Z_last || f || SALT)
    data = lorenz_bytes + frame_bytes + salt
    return hashlib.sha256(data).digest()


def get_pool_offset(seed_hash: bytes, pool_size: int) -> int:
    """Get deterministic offset into chaos pool"""
    hash_int = struct.unpack('>Q', seed_hash[:8])[0]
    return hash_int % (pool_size - 10000)  # Leave safety margin


def build_sbox_from_chaos(pool_bytes: bytes, offset: int) -> Tuple[List[int], List[int]]:
    """Build bijective S-box using Fisher-Yates shuffle with chaotic bytes"""
    sbox = list(range(256))
    
    # Fisher-Yates shuffle using chaotic bytes
    for i in range(255, 0, -1):
        if offset + (255 - i) < len(pool_bytes):
            chaos_byte = pool_bytes[offset + (255 - i)]
            j = chaos_byte % (i + 1)
            sbox[i], sbox[j] = sbox[j], sbox[i]
    
    # Build inverse S-box
    inv_sbox = [0] * 256
    for i in range(256):
        inv_sbox[sbox[i]] = i
    
    return sbox, inv_sbox


def build_permutation_from_chaos(pool_bytes: bytes, offset: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build permutation using Fisher-Yates with chaotic bytes"""
    perm = np.arange(size)
    
    # Fisher-Yates shuffle
    for i in range(size - 1, 0, -1):
        if offset + (size - 1 - i) < len(pool_bytes):
            chaos_byte = pool_bytes[offset + (size - 1 - i)]
            j = chaos_byte % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
    
    # Build inverse permutation
    inv_perm = np.argsort(perm)
    return perm, inv_perm


def generate_chaotic_keystream(pool_bytes: bytes, offset: int, length: int) -> bytes:
    """Generate keystream from chaotic pool"""
    keystream = bytearray()
    
    # Use chaotic bytes directly, wrapping around if needed
    for i in range(length):
        idx = (offset + i) % len(pool_bytes)
        keystream.append(pool_bytes[idx])
    
    return bytes(keystream)


class VideoChaoticCipher:
    """Complete chaotic video encryption pipeline following specified stages"""
    
    def __init__(self, pool_bytes: bytes):
        self.pool_bytes = pool_bytes
        self.pool_size = len(pool_bytes)
    
    def encrypt_frame(self, frame_bytes: bytes, seed_hash: bytes, frame_no: int, 
                     frame_shape: Tuple[int, int, int]) -> Tuple[bytes, bytes]:
        """
        Complete chaotic encryption pipeline:
        1. S-box substitution
        2. Block permutation  
        3. Pixel permutation
        4. Diffusion with chaining
        """
        height, width, channels = frame_shape
        
        # Convert to numpy array for processing
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)
        
        # Get chaotic resources from pool
        pool_offset = get_pool_offset(seed_hash, self.pool_size)
        
        # STAGE 1: S-BOX SUBSTITUTION
        sbox, _ = build_sbox_from_chaos(self.pool_bytes, pool_offset)
        
        # Apply S-box to each channel separately
        for c in range(channels):
            for y in range(height):
                for x in range(width):
                    frame[y, x, c] = sbox[frame[y, x, c]]
        
        # STAGE 2: BLOCK PERMUTATION (16x16 blocks)
        frame = self._apply_block_permutation(frame, pool_offset + 1000)
        
        # STAGE 3: PIXEL PERMUTATION (within blocks)  
        frame = self._apply_pixel_permutation(frame, pool_offset + 2000)
        
        # STAGE 4: DIFFUSION WITH CHAINING
        frame_flat = frame.flatten()
        keystream = generate_chaotic_keystream(self.pool_bytes, pool_offset + 3000, len(frame_flat))
        
        # Chained diffusion: enc[i] = (pixel[i] XOR ks[i] + enc[i-1]) mod 256
        encrypted_flat = np.zeros_like(frame_flat)
        prev_enc = 0
        
        for i in range(len(frame_flat)):
            encrypted_flat[i] = (frame_flat[i] ^ keystream[i] + prev_enc) % 256
            prev_enc = encrypted_flat[i]
        
        # Pack metadata (frame dimensions)
        metadata = struct.pack('>III', height, width, channels)
        
        return encrypted_flat.tobytes(), metadata
    
    def decrypt_frame(self, encrypted_data: bytes, metadata: bytes, seed_hash: bytes, 
                     frame_no: int) -> bytes:
        """Reverse all encryption stages in opposite order"""
        
        # Unpack metadata
        height, width, channels = struct.unpack('>III', metadata)
        frame_shape = (height, width, channels)
        
        encrypted_flat = np.frombuffer(encrypted_data, dtype=np.uint8)
        
        # Get same chaotic resources
        pool_offset = get_pool_offset(seed_hash, self.pool_size)
        
        # STAGE 4 REVERSE: Undo diffusion with chaining
        keystream = generate_chaotic_keystream(self.pool_bytes, pool_offset + 3000, len(encrypted_flat))
        
        decrypted_flat = np.zeros_like(encrypted_flat)
        prev_enc = 0
        
        for i in range(len(encrypted_flat)):
            # Reverse: pixel[i] = (enc[i] - enc[i-1]) XOR ks[i]
            temp = (encrypted_flat[i] - prev_enc) % 256
            decrypted_flat[i] = temp ^ keystream[i]
            prev_enc = encrypted_flat[i]
        
        # Reshape back to frame
        frame = decrypted_flat.reshape(frame_shape)
        
        # STAGE 3 REVERSE: Undo pixel permutation
        frame = self._reverse_pixel_permutation(frame, pool_offset + 2000)
        
        # STAGE 2 REVERSE: Undo block permutation
        frame = self._reverse_block_permutation(frame, pool_offset + 1000)
        
        # STAGE 1 REVERSE: Undo S-box substitution
        _, inv_sbox = build_sbox_from_chaos(self.pool_bytes, pool_offset)
        
        for c in range(frame.shape[2]):
            for y in range(frame.shape[0]):
                for x in range(frame.shape[1]):
                    frame[y, x, c] = inv_sbox[frame[y, x, c]]
        
        return frame.tobytes()
    
    def _apply_block_permutation(self, frame: np.ndarray, chaos_offset: int) -> np.ndarray:
        """Permute 16x16 blocks using chaotic randomness"""
        height, width, channels = frame.shape
        block_size = 16
        
        # Create list of blocks
        blocks = []
        positions = []
        
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = frame[i:min(i+block_size, height), j:min(j+block_size, width)]
                blocks.append(block.copy())
                positions.append((i, j))
        
        # Permute blocks using chaos
        if len(blocks) > 1:
            perm, _ = build_permutation_from_chaos(self.pool_bytes, chaos_offset, len(blocks))
            blocks_permuted = [blocks[perm[i]] for i in range(len(blocks))]
        else:
            blocks_permuted = blocks
        
        # Reconstruct frame
        result = np.zeros_like(frame)
        for idx, (i, j) in enumerate(positions):
            if idx < len(blocks_permuted):
                block = blocks_permuted[idx]
                h_end = min(i + block.shape[0], height)
                w_end = min(j + block.shape[1], width)
                result[i:h_end, j:w_end] = block
        
        return result
    
    def _apply_pixel_permutation(self, frame: np.ndarray, chaos_offset: int) -> np.ndarray:
        """Permute pixels within each 16x16 block"""
        height, width, channels = frame.shape
        block_size = 16
        
        result = frame.copy()
        
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Extract block
                block = frame[i:min(i+block_size, height), j:min(j+block_size, width)]
                
                if block.size > 1:  # Only permute if block has multiple pixels
                    # Permute each channel separately
                    for c in range(channels):
                        channel_data = block[:, :, c].flatten()
                        if len(channel_data) > 1:
                            # Build permutation for this block
                            block_offset = chaos_offset + (i * width + j) % 1000
                            perm, _ = build_permutation_from_chaos(self.pool_bytes, block_offset, len(channel_data))
                            
                            # Apply permutation
                            channel_permuted = channel_data[perm]
                            block[:, :, c] = channel_permuted.reshape(block[:, :, c].shape)
                
                # Put block back
                h_end = min(i + block.shape[0], height)
                w_end = min(j + block.shape[1], width)
                result[i:h_end, j:w_end] = block
        
        return result
    
    def _reverse_block_permutation(self, frame: np.ndarray, chaos_offset: int) -> np.ndarray:
        """Reverse block permutation"""
        height, width, channels = frame.shape
        block_size = 16
        
        # Create list of blocks and positions
        blocks = []
        positions = []
        
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block = frame[i:min(i+block_size, height), j:min(j+block_size, width)]
                blocks.append(block.copy())
                positions.append((i, j))
        
        # Build inverse permutation
        if len(blocks) > 1:
            _, inv_perm = build_permutation_from_chaos(self.pool_bytes, chaos_offset, len(blocks))
            blocks_restored = [blocks[inv_perm[i]] for i in range(len(blocks))]
        else:
            blocks_restored = blocks
        
        # Reconstruct frame
        result = np.zeros_like(frame)
        for idx, (i, j) in enumerate(positions):
            if idx < len(blocks_restored):
                block = blocks_restored[idx]
                h_end = min(i + block.shape[0], height)
                w_end = min(j + block.shape[1], width)
                result[i:h_end, j:w_end] = block
        
        return result
    
    def _reverse_pixel_permutation(self, frame: np.ndarray, chaos_offset: int) -> np.ndarray:
        """Reverse pixel permutation within blocks"""
        height, width, channels = frame.shape
        block_size = 16
        
        result = frame.copy()
        
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Extract block
                block = frame[i:min(i+block_size, height), j:min(j+block_size, width)]
                
                if block.size > 1:
                    # Reverse permute each channel
                    for c in range(channels):
                        channel_data = block[:, :, c].flatten()
                        if len(channel_data) > 1:
                            # Build inverse permutation for this block
                            block_offset = chaos_offset + (i * width + j) % 1000
                            _, inv_perm = build_permutation_from_chaos(self.pool_bytes, block_offset, len(channel_data))
                            
                            # Apply inverse permutation
                            channel_restored = channel_data[inv_perm]
                            block[:, :, c] = channel_restored.reshape(block[:, :, c].shape)
                
                # Put block back
                h_end = min(i + block.shape[0], height)
                w_end = min(j + block.shape[1], width)
                result[i:h_end, j:w_end] = block
        
        return result


# Audio encryption functions
def encrypt_audio_dct_double(audio_segment: np.ndarray, seed_hash: bytes) -> Tuple[bytes, bytes]:
    """Double DCT audio encryption"""
    # Convert to float
    audio_float = audio_segment.astype(np.float64)
    
    # First DCT
    dct1 = dct(audio_float, norm='ortho')
    
    # Second DCT
    dct2 = dct(dct1, norm='ortho')
    
    # Generate key from seed
    key_seed = struct.unpack('>Q', seed_hash[:8])[0]
    np.random.seed(key_seed & 0xFFFFFFFF)
    chaos_key = np.random.random(len(dct2)) * 1000 - 500
    
    # Encrypt: add chaos key
    encrypted_dct = dct2 + chaos_key
    
    # Convert to int16 for transmission
    encrypted_audio = np.clip(encrypted_dct, -32768, 32767).astype(np.int16)
    
    # Metadata
    metadata = struct.pack('>QI', key_seed, len(audio_segment))
    
    return encrypted_audio.tobytes(), metadata


def decrypt_audio_dct_double(encrypted_data: bytes, metadata: bytes) -> np.ndarray:
    """Double DCT audio decryption"""
    # Unpack metadata
    key_seed, frame_len = struct.unpack('>QI', metadata)
    
    # Convert back to float
    encrypted_audio = np.frombuffer(encrypted_data, dtype=np.int16).astype(np.float64)
    
    # Generate same key
    np.random.seed(key_seed & 0xFFFFFFFF)
    chaos_key = np.random.random(len(encrypted_audio)) * 1000 - 500
    
    # Decrypt: subtract key
    decrypted_dct2 = encrypted_audio - chaos_key
    
    # Reverse second DCT
    decrypted_dct1 = idct(decrypted_dct2, norm='ortho')
    
    # Reverse first DCT
    decrypted_audio = idct(decrypted_dct1, norm='ortho')
    
    # Convert back to int16
    result = np.clip(decrypted_audio, -32768, 32767).astype(np.int16)
    
    return result


def compute_hmac(metadata: bytes, ciphertext: bytes, salt: bytes) -> bytes:
    """Compute HMAC-SHA256 authentication tag"""
    data = metadata + ciphertext
    return hmac.new(salt, data, hashlib.sha256).digest()


def verify_hmac(metadata: bytes, ciphertext: bytes, tag: bytes, salt: bytes) -> bool:
    """Verify HMAC authentication tag"""
    expected_tag = compute_hmac(metadata, ciphertext, salt)
    return hmac.compare_digest(tag, expected_tag)
