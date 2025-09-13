# cipher_system.py
import hashlib
import hmac
import struct
import zlib
import numpy as np
from typing import Tuple, List, Optional, Union


def load_pool_bytes(filename: str) -> bytes:
    """Load pre-generated Lorenz pool bytes"""
    with open(filename, 'rb') as f:
        return f.read()


def derive_seed_hash(frame_no: int, salt: bytes) -> bytes:
    """Derive seed hash from frame number and salt"""
    data = struct.pack('>Q', frame_no) + salt
    return hashlib.sha256(data).digest()


def get_pool_offset(seed_hash: bytes, pool_size: int) -> int:
    """Get deterministic offset into pool bytes"""
    hash_int = struct.unpack('>Q', seed_hash[:8])[0]
    return hash_int % (pool_size - 10000)  # Ensure enough bytes available


def build_sbox(pool_bytes: bytes, offset: int) -> Tuple[List[int], List[int]]:
    """Build bijective S-box using Fisher-Yates shuffle"""
    sbox = list(range(256))
    inv_sbox = [0] * 256
    
    # Use pool bytes as random source for Fisher-Yates
    for i in range(255, 0, -1):
        j = pool_bytes[offset + (255 - i)] % (i + 1)
        sbox[i], sbox[j] = sbox[j], sbox[i]
    
    # Build inverse S-box
    for i in range(256):
        inv_sbox[sbox[i]] = i
    
    return sbox, inv_sbox


def build_permutation(pool_bytes: bytes, offset: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build permutation using Fisher-Yates shuffle"""
    perm = np.arange(size)
    
    # Fisher-Yates shuffle
    for i in range(size - 1, 0, -1):
        if offset + (size - 1 - i) < len(pool_bytes):
            j = pool_bytes[offset + (size - 1 - i)] % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
    
    # Build inverse permutation
    inv_perm = np.argsort(perm)
    
    return perm, inv_perm


def generate_keystream(seed: bytes, length: int) -> bytes:
    """Generate keystream using SHA256 chaining"""
    keystream = bytearray()
    current = seed
    
    while len(keystream) < length:
        current = hashlib.sha256(current).digest()
        keystream.extend(current)
    
    return bytes(keystream[:length])


# ============================
# VIDEO ENCRYPTION PIPELINE (2D SPATIAL)
# ============================

class VideoChaoticCipher:
    """2D Spatial Video Encryption using Chaotic Lorenz Pool"""
    
    def __init__(self, pool_bytes: bytes):
        self.pool_bytes = pool_bytes
        self.pool_size = len(pool_bytes)
    
    def _apply_block_permutation(self, frame: np.ndarray, perm: np.ndarray) -> np.ndarray:
        """Apply block-level permutation to video frame"""
        height, width, channels = frame.shape
        block_height = max(1, height // 8)  # 8x8 blocks
        block_width = max(1, width // 8)
        
        # Create blocks
        blocks = []
        for i in range(0, height, block_height):
            for j in range(0, width, block_width):
                block = frame[i:i+block_height, j:j+block_width]
                blocks.append(block)
        
        # Permute blocks
        if len(blocks) > 1 and len(perm) >= len(blocks):
            perm_blocks = [blocks[perm[i] % len(blocks)] for i in range(len(blocks))]
        else:
            perm_blocks = blocks
        
        # Reconstruct frame
        result = np.zeros_like(frame)
        idx = 0
        for i in range(0, height, block_height):
            for j in range(0, width, block_width):
                if idx < len(perm_blocks):
                    block = perm_blocks[idx]
                    h_end = min(i + block.shape[0], height)
                    w_end = min(j + block.shape[1], width)
                    result[i:h_end, j:w_end] = block[:h_end-i, :w_end-j]
                    idx += 1
        
        return result
    
    def _apply_pixel_permutation(self, block: np.ndarray, perm: np.ndarray) -> np.ndarray:
        """Apply pixel-level permutation within blocks"""
        shape = block.shape
        flat = block.flatten()
        
        if len(perm) >= len(flat):
            perm_indices = perm[:len(flat)]
            permuted_flat = flat[perm_indices]
        else:
            permuted_flat = flat
        
        return permuted_flat.reshape(shape)
    
    def encrypt_frame(self, frame_bytes: bytes, seed_hash: bytes, frame_no: int, 
                     frame_shape: tuple) -> Tuple[bytes, bytes]:
        """Encrypt video frame using 2D spatial pipeline"""
        height, width, channels = frame_shape
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)
        
        # Get chaotic resources
        pool_offset = get_pool_offset(seed_hash, self.pool_size)
        
        # Step 1: Build S-box for substitution
        sbox, _ = build_sbox(self.pool_bytes, pool_offset)
        
        # Step 2: Block-level permutation
        block_perm, _ = build_permutation(self.pool_bytes, pool_offset + 500, 64)
        frame = self._apply_block_permutation(frame, block_perm)
        
        # Step 3: Pixel-level permutation within blocks
        pixel_perm, _ = build_permutation(self.pool_bytes, pool_offset + 1000, 
                                         min(1024, frame.size))
        frame = self._apply_pixel_permutation(frame, pixel_perm)
        
        # Step 4: S-box substitution
        frame_flat = frame.flatten()
        for i in range(len(frame_flat)):
            frame_flat[i] = sbox[frame_flat[i]]
        
        # Step 5: Diffusion with chaotic keystream
        keystream_seed = self.pool_bytes[pool_offset+2000:pool_offset+2032]
        keystream = generate_keystream(keystream_seed, len(frame_flat))
        
        for i in range(len(frame_flat)):
            frame_flat[i] = (frame_flat[i] ^ keystream[i]) & 0xFF
        
        # Pack metadata
        metadata = struct.pack('>III', height, width, channels)
        
        return frame_flat.tobytes(), metadata
    
    def decrypt_frame(self, encrypted_data: bytes, metadata: bytes, seed_hash: bytes, 
                     frame_no: int) -> bytes:
        """Decrypt video frame (reverse of encrypt_frame)"""
        # Unpack metadata
        height, width, channels = struct.unpack('>III', metadata)
        frame_shape = (height, width, channels)
        
        frame_flat = np.frombuffer(encrypted_data, dtype=np.uint8).copy()
        
        # Get chaotic resources (same as encryption)
        pool_offset = get_pool_offset(seed_hash, self.pool_size)
        sbox, inv_sbox = build_sbox(self.pool_bytes, pool_offset)
        
        # Step 5 (reverse): Reverse diffusion
        keystream_seed = self.pool_bytes[pool_offset+2000:pool_offset+2032]
        keystream = generate_keystream(keystream_seed, len(frame_flat))
        
        for i in range(len(frame_flat)):
            frame_flat[i] = (frame_flat[i] ^ keystream[i]) & 0xFF
        
        # Step 4 (reverse): Reverse S-box substitution
        for i in range(len(frame_flat)):
            frame_flat[i] = inv_sbox[frame_flat[i]]
        
        # Reshape back to frame
        frame = frame_flat.reshape(frame_shape)
        
        # Step 3 (reverse): Reverse pixel-level permutation
        pixel_perm, inv_pixel_perm = build_permutation(self.pool_bytes, pool_offset + 1000, 
                                                      min(1024, frame.size))
        frame = self._reverse_pixel_permutation(frame, inv_pixel_perm)
        
        # Step 2 (reverse): Reverse block-level permutation
        block_perm, inv_block_perm = build_permutation(self.pool_bytes, pool_offset + 500, 64)
        frame = self._reverse_block_permutation(frame, inv_block_perm)
        
        return frame.tobytes()
    
    def _reverse_block_permutation(self, frame: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
        """Reverse block-level permutation"""
        height, width, channels = frame.shape
        block_height = max(1, height // 8)
        block_width = max(1, width // 8)
        
        # Create blocks
        blocks = []
        for i in range(0, height, block_height):
            for j in range(0, width, block_width):
                block = frame[i:i+block_height, j:j+block_width]
                blocks.append(block)
        
        # Reverse permute blocks
        if len(blocks) > 1 and len(inv_perm) >= len(blocks):
            unperm_blocks = [None] * len(blocks)
            for i in range(len(blocks)):
                unperm_blocks[inv_perm[i] % len(blocks)] = blocks[i]
            # Fill any None values
            for i in range(len(unperm_blocks)):
                if unperm_blocks[i] is None:
                    unperm_blocks[i] = blocks[i % len(blocks)]
        else:
            unperm_blocks = blocks
        
        # Reconstruct frame
        result = np.zeros_like(frame)
        idx = 0
        for i in range(0, height, block_height):
            for j in range(0, width, block_width):
                if idx < len(unperm_blocks):
                    block = unperm_blocks[idx]
                    h_end = min(i + block.shape[0], height)
                    w_end = min(j + block.shape[1], width)
                    result[i:h_end, j:w_end] = block[:h_end-i, :w_end-j]
                    idx += 1
        
        return result
    
    def _reverse_pixel_permutation(self, block: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
        """Reverse pixel-level permutation"""
        shape = block.shape
        flat = block.flatten()
        
        if len(inv_perm) >= len(flat):
            inv_perm_indices = inv_perm[:len(flat)]
            unpermuted_flat = np.zeros_like(flat)
            unpermuted_flat[inv_perm_indices] = flat
        else:
            unpermuted_flat = flat
        
        return unpermuted_flat.reshape(shape)


# ============================
# AUDIO ENCRYPTION PIPELINE (1D TEMPORAL)
# ============================

class AudioChaoticCipher:
    """1D Temporal Audio Encryption using Chaotic Lorenz Pipeline"""
    
    def __init__(self, pool_bytes: bytes, salt: bytes):
        self.pool_bytes = pool_bytes
        self.salt = salt
        self.pool_size = len(pool_bytes)
    
    def _derive_frame_key(self, frame_no: int, last_state: Optional[bytes] = None) -> bytes:
        """Derive unique key for each audio frame"""
        if last_state is None:
            last_state = b'\x00' * 12  # Default state
        
        data = last_state + struct.pack('>Q', frame_no) + self.salt
        return hashlib.sha256(data).digest()
    
    def _get_keystream(self, seed: bytes, length: int) -> np.ndarray:
        """Generate keystream from chaotic pool"""
        # Use seed to determine pool offset
        offset = struct.unpack('>Q', seed[:8])[0] % (self.pool_size - length * 2)
        
        # Extract keystream bytes and convert to int16
        keystream_bytes = self.pool_bytes[offset:offset + length * 2]
        keystream = []
        for i in range(0, len(keystream_bytes), 2):
            if i + 1 < len(keystream_bytes):
                val = struct.unpack('>H', keystream_bytes[i:i+2])[0]
                # Convert to signed 16-bit range, scaled to avoid overflow
                keystream.append((val - 32768) // 8)
        
        return np.array(keystream[:length], dtype=np.int16)
    
    def _get_permutation(self, seed: bytes, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate permutation from chaotic pool"""
        offset = (struct.unpack('>Q', seed[8:16])[0] % (self.pool_size - length)) 
        perm_bytes = self.pool_bytes[offset:offset + length]
        
        # Create permutation using Fisher-Yates shuffle
        perm = np.arange(length)
        for i in range(length - 1, 0, -1):
            j = perm_bytes[length - 1 - i] % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        
        # Create inverse permutation
        inv_perm = np.argsort(perm)
        
        return perm, inv_perm
    
    def encrypt_frame(self, audio_frame: np.ndarray, frame_no: int, 
                     last_state: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encrypt single audio frame using 1D temporal pipeline"""
        # Ensure int16 format
        if audio_frame.dtype != np.int16:
            audio_frame = audio_frame.astype(np.int16)
        
        frame_len = len(audio_frame)
        
        # Derive frame key
        frame_key = self._derive_frame_key(frame_no, last_state)
        
        # Stage 1: Stream cipher (XOR with keystream)
        keystream = self._get_keystream(frame_key, frame_len)
        encrypted = audio_frame.astype(np.int32)
        for i in range(len(encrypted)):
            encrypted[i] = (encrypted[i] + keystream[i]) & 0xFFFF
            if encrypted[i] > 32767:
                encrypted[i] -= 65536
        encrypted = encrypted.astype(np.int16)
        
        # Stage 2: Frame-level permutation (sample scrambling)
        perm, _ = self._get_permutation(frame_key, frame_len)
        permuted = encrypted[perm]
        
        # Stage 3: Temporal diffusion (each sample depends on previous)
        diffused = permuted.copy().astype(np.int32)
        for i in range(1, len(diffused)):
            diffused[i] = (diffused[i] + (diffused[i-1] // 4)) & 0xFFFF
            if diffused[i] > 32767:
                diffused[i] -= 65536
        
        # Convert back to int16
        diffused = diffused.astype(np.int16)
        
        # Return encrypted frame and current state for next frame
        current_state = frame_key[:12]  # Use part of key as state
        
        return diffused.tobytes(), current_state
    
    def decrypt_frame(self, encrypted_data: bytes, frame_no: int,
                     last_state: Optional[bytes] = None) -> Tuple[np.ndarray, bytes]:
        """Decrypt single audio frame (reverse of encrypt_frame)"""
        # Convert bytes back to int16 array
        encrypted_frame = np.frombuffer(encrypted_data, dtype=np.int16)
        frame_len = len(encrypted_frame)
        
        # Derive same frame key
        frame_key = self._derive_frame_key(frame_no, last_state)
        
        # Stage 3 (reverse): Reverse temporal diffusion
        diffused = encrypted_frame.astype(np.int32)
        for i in range(frame_len - 1, 0, -1):
            diffused[i] = (diffused[i] - (diffused[i-1] // 4)) & 0xFFFF
            if diffused[i] > 32767:
                diffused[i] -= 65536
        
        # Convert back to int16
        undiffused = diffused.astype(np.int16)
        
        # Stage 2 (reverse): Reverse frame-level permutation
        perm, inv_perm = self._get_permutation(frame_key, frame_len)
        unpermuted = undiffused[inv_perm]
        
        # Stage 1 (reverse): Reverse stream cipher
        keystream = self._get_keystream(frame_key, frame_len)
        decrypted = unpermuted.astype(np.int32)
        for i in range(len(decrypted)):
            decrypted[i] = (decrypted[i] - keystream[i]) & 0xFFFF
            if decrypted[i] > 32767:
                decrypted[i] -= 65536
        decrypted = decrypted.astype(np.int16)
        
        # Return decrypted frame and current state
        current_state = frame_key[:12]
        
        return decrypted, current_state


# ============================
# PUBLIC INTERFACE FUNCTIONS
# ============================

def encrypt_video_data(data: bytes, pool_bytes: bytes, seed_hash: bytes, frame_no: int, 
                      frame_shape: tuple) -> Tuple[bytes, bytes]:
    """Encrypt VIDEO data using 2D spatial pipeline"""
    cipher = VideoChaoticCipher(pool_bytes)
    return cipher.encrypt_frame(data, seed_hash, frame_no, frame_shape)


def decrypt_video_data(ciphertext: bytes, metadata: bytes, pool_bytes: bytes, 
                      seed_hash: bytes, frame_no: int) -> bytes:
    """Decrypt VIDEO data using 2D spatial pipeline"""
    cipher = VideoChaoticCipher(pool_bytes)
    return cipher.decrypt_frame(ciphertext, metadata, seed_hash, frame_no)


def encrypt_audio_data(audio_frame: np.ndarray, pool_bytes: bytes, seed_hash: bytes, 
                      frame_no: int, last_state: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Encrypt AUDIO data using 1D temporal pipeline"""
    cipher = AudioChaoticCipher(pool_bytes, seed_hash[:16])  # Use part of seed as salt
    encrypted_data, new_state = cipher.encrypt_frame(audio_frame, frame_no, last_state)
    
    # Pack metadata: [frame_no(8)] [state_len(4)] [state]
    metadata = (struct.pack('>Q', frame_no) + 
               struct.pack('>I', len(new_state)) + 
               new_state)
    
    return encrypted_data, metadata


def decrypt_audio_data(encrypted_data: bytes, metadata: bytes, pool_bytes: bytes, 
                      seed_hash: bytes) -> Tuple[np.ndarray, bytes]:
    """Decrypt AUDIO data using 1D temporal pipeline"""
    # Unpack metadata
    frame_no = struct.unpack('>Q', metadata[:8])[0]
    state_len = struct.unpack('>I', metadata[8:12])[0]
    last_state = metadata[12:12+state_len] if state_len > 0 else None
    
    cipher = AudioChaoticCipher(pool_bytes, seed_hash[:16])  # Use part of seed as salt
    decrypted_frame, new_state = cipher.decrypt_frame(encrypted_data, frame_no, last_state)
    
    return decrypted_frame, new_state


def compute_hmac(metadata: bytes, ciphertext: bytes, salt: bytes) -> bytes:
    """Compute HMAC-SHA256 over metadata + ciphertext"""
    data = metadata + ciphertext
    return hmac.new(salt, data, hashlib.sha256).digest()


def verify_hmac(metadata: bytes, ciphertext: bytes, tag: bytes, salt: bytes) -> bool:
    """Verify HMAC tag"""
    expected_tag = compute_hmac(metadata, ciphertext, salt)
    return hmac.compare_digest(tag, expected_tag)