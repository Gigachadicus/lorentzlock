import hashlib
import hmac
import struct
import zlib
import numpy as np
from typing import Tuple, List


def quantize_lorenz_state(state: np.ndarray, bits: int = 32) -> int:
    """Quantize Lorenz state to integer for deterministic operations"""
    x, y, z = state
    # Normalize to [0, 1] range using known Lorenz bounds
    x_norm = (x + 30) / 60  # Lorenz x typically in [-30, 30]
    y_norm = (y + 30) / 60  # Lorenz y typically in [-30, 30] 
    z_norm = z / 60         # Lorenz z typically in [0, 60]
    
    # Clamp to [0, 1]
    x_norm = max(0, min(1, x_norm))
    y_norm = max(0, min(1, y_norm))
    z_norm = max(0, min(1, z_norm))
    
    # Convert to integer
    max_val = (1 << bits) - 1
    return int((x_norm + y_norm + z_norm) / 3 * max_val)


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


def build_permutation(pool_bytes: bytes, offset: int, size: int) -> Tuple[List[int], List[int]]:
    """Build permutation using Fisher-Yates shuffle"""
    perm = list(range(size))
    inv_perm = [0] * size
    
    # Fisher-Yates shuffle
    for i in range(size - 1, 0, -1):
        if offset + (size - 1 - i) < len(pool_bytes):
            j = pool_bytes[offset + (size - 1 - i)] % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
    
    # Build inverse permutation
    for i in range(size):
        inv_perm[perm[i]] = i
    
    return perm, inv_perm


def generate_keystream(seed: bytes, length: int) -> bytes:
    """Generate keystream using SHA256 chaining"""
    keystream = bytearray()
    current = seed
    
    while len(keystream) < length:
        current = hashlib.sha256(current).digest()
        keystream.extend(current)
    
    return bytes(keystream[:length])


def mixcolumns_forward(block: bytes) -> bytes:
    """Forward MixColumns operation on 4-byte words"""
    if len(block) % 4 != 0:
        raise ValueError("Block size must be multiple of 4")
    
    result = bytearray()
    
    for i in range(0, len(block), 4):
        word = block[i:i+4]
        a, b, c, d = word
        
        # MixColumns-like transformation
        new_a = (2*a + 3*b + c + d) & 0xFF
        new_b = (a + 2*b + 3*c + d) & 0xFF  
        new_c = (a + b + 2*c + 3*d) & 0xFF
        new_d = (3*a + b + c + 2*d) & 0xFF
        
        result.extend([new_a, new_b, new_c, new_d])
    
    return bytes(result)


def mixcolumns_inverse(block: bytes) -> bytes:
    """Inverse MixColumns operation"""
    if len(block) % 4 != 0:
        raise ValueError("Block size must be multiple of 4")
    
    result = bytearray()
    
    for i in range(0, len(block), 4):
        word = block[i:i+4]
        a, b, c, d = word
        
        # Inverse MixColumns transformation
        new_a = (14*a + 11*b + 13*c + 9*d) & 0xFF
        new_b = (9*a + 14*b + 11*c + 13*d) & 0xFF
        new_c = (13*a + 9*b + 14*c + 11*d) & 0xFF
        new_d = (11*a + 13*b + 9*c + 14*d) & 0xFF
        
        result.extend([new_a, new_b, new_c, new_d])
    
    return bytes(result)


def encrypt_block(block: bytes, pool_bytes: bytes, offset: int, rounds: int = 3) -> bytes:
    """Encrypt single block using Lorenz cipher pipeline"""
    block_size = len(block)
    
    # Build cipher resources
    sbox, _ = build_sbox(pool_bytes, offset)
    local_perm, _ = build_permutation(pool_bytes, offset + 256, block_size)
    
    # Generate keystream and tweaks
    seed = pool_bytes[offset:offset+32]
    keystream = generate_keystream(seed, block_size * rounds)
    tweak_bytes = pool_bytes[offset+300:offset+300+block_size][:block_size]
    
    data = bytearray(block)
    
    # Apply rounds
    for r in range(rounds):
        # MixColumns diffusion
        if len(data) % 4 == 0:
            data = bytearray(mixcolumns_forward(bytes(data)))
        
        # Local shuffle
        temp = bytearray(len(data))
        for i in range(len(data)):
            temp[local_perm[i]] = data[i]
        data = temp
        
        # XOR with keystream
        ks_start = r * block_size
        for i in range(len(data)):
            data[i] ^= keystream[ks_start + i]
        
        # S-box substitution
        for i in range(len(data)):
            data[i] = sbox[data[i]]
        
        # Tweak addition
        for i in range(len(data)):
            data[i] = (data[i] + tweak_bytes[i % len(tweak_bytes)]) & 0xFF
    
    return bytes(data)


def decrypt_block(block: bytes, pool_bytes: bytes, offset: int, rounds: int = 3) -> bytes:
    """Decrypt single block (reverse of encrypt_block)"""
    block_size = len(block)
    
    # Build cipher resources (same as encryption)
    sbox, inv_sbox = build_sbox(pool_bytes, offset)
    local_perm, inv_local_perm = build_permutation(pool_bytes, offset + 256, block_size)
    
    # Generate keystream and tweaks
    seed = pool_bytes[offset:offset+32]
    keystream = generate_keystream(seed, block_size * rounds)
    tweak_bytes = pool_bytes[offset+300:offset+300+block_size][:block_size]
    
    data = bytearray(block)
    
    # Reverse rounds (last to first)
    for r in range(rounds - 1, -1, -1):
        # Reverse tweak addition
        for i in range(len(data)):
            data[i] = (data[i] - tweak_bytes[i % len(tweak_bytes)]) & 0xFF
        
        # Reverse S-box substitution
        for i in range(len(data)):
            data[i] = inv_sbox[data[i]]
        
        # Reverse XOR with keystream
        ks_start = r * block_size
        for i in range(len(data)):
            data[i] ^= keystream[ks_start + i]
        
        # Reverse local shuffle
        temp = bytearray(len(data))
        for i in range(len(data)):
            temp[i] = data[inv_local_perm[i]]
        data = temp
        
        # Reverse MixColumns diffusion
        if len(data) % 4 == 0:
            data = bytearray(mixcolumns_inverse(bytes(data)))
    
    return bytes(data)


def pad_data(data: bytes, block_size: int) -> Tuple[bytes, int]:
    """Pad data to block size and return pad length"""
    pad_len = block_size - (len(data) % block_size)
    if pad_len == block_size:
        pad_len = 0
        return data, pad_len
    
    padding = bytes([pad_len] * pad_len)
    return data + padding, pad_len


def unpad_data(data: bytes, pad_len: int) -> bytes:
    """Remove padding from data"""
    if pad_len == 0:
        return data
    return data[:-pad_len]


def encrypt_data(data: bytes, pool_bytes: bytes, seed_hash: bytes, block_size: int, rounds: int = 3) -> Tuple[bytes, bytes]:
    """Encrypt data using Lorenz cipher pipeline"""
    # Step 1: Compression
    compressed = zlib.compress(data)
    
    # Step 2: Block partitioning with padding
    padded_data, pad_len = pad_data(compressed, block_size)
    num_blocks = len(padded_data) // block_size
    
    # Get pool offset
    pool_offset = get_pool_offset(seed_hash, len(pool_bytes))
    
    # Step 3-4: Encrypt blocks
    encrypted_blocks = []
    for i in range(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size
        block = padded_data[block_start:block_end]
        
        # Each block gets different pool offset
        block_offset = (pool_offset + i * 1000) % (len(pool_bytes) - 1000)
        encrypted_block = encrypt_block(block, pool_bytes, block_offset, rounds)
        encrypted_blocks.append(encrypted_block)
    
    # Step 5: Global block shuffle
    if num_blocks > 1:
        global_perm, _ = build_permutation(pool_bytes, pool_offset + 500, num_blocks)
        shuffled_blocks = [encrypted_blocks[global_perm[i]] for i in range(num_blocks)]
    else:
        shuffled_blocks = encrypted_blocks
    
    ciphertext = b''.join(shuffled_blocks)
    
    # Pack metadata
    metadata = struct.pack('>III', len(data), pad_len, num_blocks)
    
    return ciphertext, metadata


def decrypt_data(ciphertext: bytes, metadata: bytes, pool_bytes: bytes, seed_hash: bytes, block_size: int, rounds: int = 3) -> bytes:
    """Decrypt data (reverse of encrypt_data)"""
    # Unpack metadata
    orig_len, pad_len, num_blocks = struct.unpack('>III', metadata)
    
    # Get pool offset
    pool_offset = get_pool_offset(seed_hash, len(pool_bytes))
    
    # Split ciphertext into blocks
    encrypted_blocks = []
    for i in range(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size
        encrypted_blocks.append(ciphertext[block_start:block_end])
    
    # Step 1: Reverse global block shuffle
    if num_blocks > 1:
        global_perm, inv_global_perm = build_permutation(pool_bytes, pool_offset + 500, num_blocks)
        unshuffled_blocks = [encrypted_blocks[inv_global_perm[i]] for i in range(num_blocks)]
    else:
        unshuffled_blocks = encrypted_blocks
    
    # Step 2: Decrypt blocks
    decrypted_blocks = []
    for i, block in enumerate(unshuffled_blocks):
        # Each block gets different pool offset
        block_offset = (pool_offset + i * 1000) % (len(pool_bytes) - 1000)
        decrypted_block = decrypt_block(block, pool_bytes, block_offset, rounds)
        decrypted_blocks.append(decrypted_block)
    
    # Reconstruct data
    padded_data = b''.join(decrypted_blocks)
    
    # Step 3: Remove padding
    compressed_data = unpad_data(padded_data, pad_len)
    
    # Step 4: Decompress
    original_data = zlib.decompress(compressed_data)
    
    if len(original_data) != orig_len:
        raise ValueError(f"Decrypted data length mismatch: expected {orig_len}, got {len(original_data)}")
    
    return original_data


def compute_hmac(metadata: bytes, ciphertext: bytes, salt: bytes) -> bytes:
    """Compute HMAC-SHA256 over metadata + ciphertext"""
    data = metadata + ciphertext
    return hmac.new(salt, data, hashlib.sha256).digest()


def verify_hmac(metadata: bytes, ciphertext: bytes, tag: bytes, salt: bytes) -> bool:
    """Verify HMAC tag"""
    expected_tag = compute_hmac(metadata, ciphertext, salt)
    return hmac.compare_digest(tag, expected_tag)
