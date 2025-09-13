# slave.py
import argparse
import socket
import struct
import threading
import time
import cv2
import numpy as np
import pyaudio
import os
from cipher_system import (
    load_pool_bytes, derive_seed_hash, decrypt_video_data, decrypt_audio_data, verify_hmac
)
from lorenz_system import generate_pool_bytes


class SlaveSystem:
    def __init__(self, args):
        self.args = args
        self.host = args.host
        self.port = args.port
        
        # Load or generate pool bytes
        if args.pool_file:
            if os.path.exists(args.pool_file):
                self.pool_bytes = load_pool_bytes(args.pool_file)
            else:
                print(f"Pool file '{args.pool_file}' not found. Generating new pool...")
                self.pool_bytes = generate_pool_bytes(args.pool_size_mb)
                # Save for future use
                with open(args.pool_file, 'wb') as f:
                    f.write(self.pool_bytes)
                print(f"Pool saved to '{args.pool_file}'")
        else:
            self.pool_bytes = generate_pool_bytes(args.pool_size_mb)
        
        self.salt = args.salt.encode()
        self.block_size = args.block_size
        self.rounds = args.rounds
        self.mode = args.mode
        
        # Networking
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        
        # Video display
        self.video_thread = None
        
        # Audio playback
        self.audio = None
        self.audio_stream = None
        self.audio_thread = None
        self.audio_last_state = None
        
        # Create samples directories if they don't exist
        os.makedirs('samples/audio', exist_ok=True)
        os.makedirs('samples/video', exist_ok=True)
        
        print(f"Slave initialized: {args.mode} mode, block_size={args.block_size}, rounds={args.rounds}")

    def connect_to_master(self):
        """Connect to master server"""
        try:
            # Send discovery message
            self.sock.sendto(b"DISCOVER", (self.host, self.port))
            
            # Wait for acknowledgment
            self.sock.settimeout(5.0)
            data, addr = self.sock.recvfrom(1024)
            
            if data == b"ACK":
                print(f"Connected to master at {addr}")
                self.sock.settimeout(None)
                return True
            else:
                print("Invalid response from master")
                return False
                
        except Exception as e:
            print(f"Failed to connect to master: {e}")
            return False

    def start_audio_playback(self):
        """Initialize audio playback"""
        if self.mode not in ['audio', 'both']:
            return
            
        self.audio = pyaudio.PyAudio()
        
        # Audio parameters (must match master)
        format = pyaudio.paInt16
        channels = 1
        rate = 44100
        
        try:
            self.audio_stream = self.audio.open(
                format=format,
                channels=channels,  
                rate=rate,
                output=True
            )
            print("Audio playback initialized")
        except Exception as e:
            print(f"Failed to initialize audio playback: {e}")

    def save_decrypted_audio_sample(self, audio_data: bytes, frame_no: int):
        """Save decrypted audio sample for debugging"""
        try:
            if frame_no in [1, 3, 5]:  # Save only specific frames
                filename = f'samples/audio/audio_{frame_no}_decrypted.bin'
                with open(filename, 'wb') as f:
                    f.write(audio_data)
                print(f"Saved decrypted audio sample {frame_no}")
        except Exception as e:
            print(f"Error saving decrypted audio sample: {e}")

    def process_video_packet(self, ciphertext: bytes, metadata: bytes, tag: bytes):
        """Process and display video packet"""
        try:
            # Verify HMAC
            if not verify_hmac(metadata, ciphertext, tag, self.salt):
                print("Video packet HMAC verification failed")
                return
            
            # Unpack frame metadata
            width, height, frame_no = struct.unpack('>III', metadata[:12])
            cipher_metadata = metadata[12:]
            
            # Generate seed for this frame
            seed_hash = derive_seed_hash(frame_no, self.salt)
            
            # Decrypt frame data using VIDEO pipeline (2D spatial)
            frame_bytes = decrypt_video_data(
                ciphertext, cipher_metadata, self.pool_bytes, 
                seed_hash, frame_no
            )
            
            # Reconstruct frame
            expected_size = width * height * 3  # RGB
            if len(frame_bytes) != expected_size:
                print(f"Frame size mismatch: expected {expected_size}, got {len(frame_bytes)}")
                return
            
            # Convert RGB to BGR for OpenCV display
            frame_rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Display frame (FIXED DISPLAY)
            cv2.imshow('Slave - Decrypted Video', frame_bgr)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Video processing error: {e}")

    def process_audio_packet(self, ciphertext: bytes, metadata: bytes, tag: bytes):
        """Process and play audio packet"""
        try:
            # Verify HMAC
            if not verify_hmac(metadata, ciphertext, tag, self.salt):
                print("Audio packet HMAC verification failed")
                return
            
            # Unpack audio metadata - first 8 bytes is frame_no
            frame_no = struct.unpack('>Q', metadata[:8])[0]
            
            # Generate seed for this audio packet
            seed_hash = derive_seed_hash(frame_no, self.salt)
            
            # Decrypt audio data using AUDIO pipeline (1D temporal)
            decrypted_frame, self.audio_last_state = decrypt_audio_data(
                ciphertext, metadata, self.pool_bytes, seed_hash
            )
            
            # Save decrypted audio sample
            self.save_decrypted_audio_sample(decrypted_frame.tobytes(), frame_no)
            
            # Play audio directly
            if self.audio_stream:
                self.audio_stream.write(decrypted_frame.tobytes())
                
        except Exception as e:
            print(f"Audio processing error: {e}")

    def packet_receiver(self):
        """Main packet receiving loop"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65536)  # Large buffer for video frames
                
                if len(data) < 13:  # Minimum header size
                    continue
                
                # Parse packet header: [type(1)] [metadata_len(4)] [data_len(4)] [tag_len(4)]
                packet_type = data[0:1]
                metadata_len, data_len, tag_len = struct.unpack('>III', data[1:13])
                
                # Extract packet components
                metadata = data[13:13+metadata_len]
                ciphertext = data[13+metadata_len:13+metadata_len+data_len]
                tag = data[13+metadata_len+data_len:13+metadata_len+data_len+tag_len]
                
                # Process based on packet type
                if packet_type == b'V' and self.mode in ['video', 'both']:
                    self.process_video_packet(ciphertext, metadata, tag)
                elif packet_type == b'A' and self.mode in ['audio', 'both']:
                    self.process_audio_packet(ciphertext, metadata, tag)
                    
            except Exception as e:
                if self.running:
                    print(f"Packet receiving error: {e}")

    def run(self):
        """Start the slave system"""
        try:
            # Connect to master
            if not self.connect_to_master():
                return
            
            self.running = True
            
            # Initialize audio playback
            if self.mode in ['audio', 'both']:
                self.start_audio_playback()
            
            # Start packet receiver thread
            receiver_thread = threading.Thread(target=self.packet_receiver, daemon=True)
            receiver_thread.start()
            
            print("Slave system running. Press Ctrl+C to stop.")
            print("Decrypted video will display in 'Slave - Decrypted Video' window if enabled.")
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping slave system...")
            self.running = False
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        if self.sock:
            self.sock.close()
            
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            
        if self.audio:
            self.audio.terminate()
            
        cv2.destroyAllWindows()
        
        print("Slave system stopped")


def main():
    parser = argparse.ArgumentParser(description='Lorenz Cipher Slave System')
    parser.add_argument('--host', default='127.0.0.1', help='Master host address')
    parser.add_argument('--port', type=int, default=5000, help='Master UDP port')
    parser.add_argument('--pool-file', help='Path to Lorenz pool bytes file (optional)')
    parser.add_argument('--pool-size-mb', type=int, default=1, help='Pool size in MB if generating (default: 1)')
    parser.add_argument('--salt', required=True, help='HMAC salt string (must match master)')
    parser.add_argument('--block-size', type=int, default=768, help='Encryption block size')
    parser.add_argument('--rounds', type=int, default=3, help='Number of encryption rounds')
    parser.add_argument('--mode', choices=['video', 'audio', 'both'], default='both', help='Playback mode')
    
    args = parser.parse_args()
    
    slave = SlaveSystem(args)
    slave.run()


if __name__ == '__main__':
    main()