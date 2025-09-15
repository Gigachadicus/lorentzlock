import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import log2
from encryption import EncryptionSystem

# ---------- Helper Functions ----------

def shannon_entropy(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0,256))
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def local_entropy(image, block_size=16):
    h, w = image.shape
    entropies = np.zeros((h//block_size, w//block_size))
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            entropies[i//block_size, j//block_size] = shannon_entropy(block)
    return entropies

def correlation_coeff(img1, img2):
    return np.corrcoef(img1.flatten(), img2.flatten())[0,1]

def npcr_uaci(img1, img2):
    # Ensure grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = img1 != img2
    npcr = np.sum(diff) / diff.size * 100
    uaci = np.sum(np.abs(img1.astype(np.int32)-img2.astype(np.int32))) / (img1.size*255) * 100
    return npcr, uaci

# ---------- Main Test Functions ----------

def key_space_analysis():
    key_bits = 256
    print(f"Key Space Size = 2^{key_bits} ≈ {2**key_bits:.2e} possible keys")

def histogram_analysis(original, encrypted):
    colors = ('b','g','r')
    plt.figure(figsize=(10,4))
    for i, col in enumerate(colors):
        plt.subplot(2,3,i+1)
        plt.hist(original[:,:,i].flatten(), 256, [0,256], color=col)
        plt.title(f'Orig {col.upper()}')
        plt.subplot(2,3,i+4)
        plt.hist(encrypted[:,:,i].flatten(), 256, [0,256], color=col)
        plt.title(f'Enc {col.upper()}')
    plt.tight_layout()
    plt.show()

def pixel_correlation(original, encrypted):
    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_e = cv2.cvtColor(encrypted, cv2.COLOR_BGR2GRAY)

    # Horizontal pairs
    x = gray_o[:,:-1].flatten()
    y = gray_o[:,1:].flatten()
    corr_o = np.corrcoef(x,y)[0,1]

    x = gray_e[:,:-1].flatten()
    y = gray_e[:,1:].flatten()
    corr_e = np.corrcoef(x,y)[0,1]

    print(f"Correlation Original: {corr_o:.4f}, Encrypted: {corr_e:.4f}")

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.scatter(gray_o[:5000], np.roll(gray_o[:5000],1), s=1)
    plt.title("Original Pixel Correlation")
    plt.subplot(1,2,2)
    plt.scatter(gray_e[:5000], np.roll(gray_e[:5000],1), s=1)
    plt.title("Encrypted Pixel Correlation")
    plt.show()

def entropy_analysis(original, encrypted):
    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_e = cv2.cvtColor(encrypted, cv2.COLOR_BGR2GRAY)
    print(f"Entropy Original: {shannon_entropy(gray_o):.4f}")
    print(f"Entropy Encrypted: {shannon_entropy(gray_e):.4f}")

def local_entropy_analysis(encrypted):
    gray = cv2.cvtColor(encrypted, cv2.COLOR_BGR2GRAY)
    ent_map = local_entropy(gray, block_size=16)
    plt.imshow(ent_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Local Entropy")
    plt.title("Local Entropy Map (Encrypted)")
    plt.show()

def key_sensitivity_test(image):
    enc1 = EncryptionSystem(k_master=b'A'*32).encrypt_frame(cv2.imencode('.jpg', image)[1].tobytes())
    enc2 = EncryptionSystem(k_master=b'B'*32).encrypt_frame(cv2.imencode('.jpg', image)[1].tobytes())

    # Convert ciphertext back to pseudo-images for comparison
    c1 = np.frombuffer(enc1['ciphertext'], dtype=np.uint8)
    c2 = np.frombuffer(enc2['ciphertext'], dtype=np.uint8)

    # Pad to same size and reshape
    size = min(len(c1), len(c2))
    side = int(np.sqrt(size))
    img1 = c1[:side*side].reshape((side,side))
    img2 = c2[:side*side].reshape((side,side))

    npcr, uaci = npcr_uaci(cv2.merge([img1,img1,img1]), cv2.merge([img2,img2,img2]))
    print(f"Key Sensitivity -> NPCR: {npcr:.2f}%, UACI: {uaci:.2f}%")

# ---------- Run All Tests ----------

if __name__ == "__main__":
    # Load a test image and its encrypted version
    original = cv2.imread("master_data/pre_encrypted_frames/pre_encrypted_frame_000000.jpg")
    encrypted_vis = cv2.imread("master_data/encrypted_frames/encrypted_frame_000000.jpg")

    key_space_analysis()
    histogram_analysis(original, encrypted_vis)
    pixel_correlation(original, encrypted_vis)
    entropy_analysis(original, encrypted_vis)
    local_entropy_analysis(encrypted_vis)
    key_sensitivity_test(original)
