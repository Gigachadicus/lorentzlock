# slave.py - Video Decryption System
import argparse
import socket
import struct
import threading
import time
import cv2
import numpy as np
import pyaudio
import os
import wave
from cipher_system import (
    VideoChaoticCipher, derive_seed_hash, decrypt_audio_dct_double, verify_hmac
)
from lorenz_system import generate_pool_bytes, load_pool_bytes


class SlaveSystem:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        
        # Load or generate same Lorenz pool as master
        if args.pool_file and os.path.exists(args.pool_file):
            print(f"Loading existing pool from {args.pool_file}")
            self.pool_bytes = load_pool_bytes(args.pool_file)
        else:
            print("Generating new Lorenz chaos pool...")
            self.pool_bytes = generate_pool_bytes(args.pool_size_mb)
            if args.pool_file:
                with open(args.pool_file, 'wb') as f:
                    f.write(self.pool_bytes)
                print(f"Pool saved to {args.pool_file}")
        
        self.salt = args.salt.encode()
        self.final_lorenz_state = (1.0, 1.0, 1.0)  # Must match master
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        
        # Initialize cipher
        self.video_cipher = VideoChaoticCipher(self.pool_bytes)
        
        # Audio setup
        self.audio = None
        self.audio_stream = None
        
        # Create output directories
        os.makedirs('output/video/decrypted', exist_ok=True)
        os.makedirs('output/audio', exist_ok=True)
        
        self.video_frame_count = 0
        self.audio_saved = False

    def connect_to_master(self):
        """Connect to master server"""
        try:
            print(f"Connecting to master at {self.host}:{self.port}")
            self.sock.sendto(b"DISCOVER", (self.host, self.port))
            self.sock.settimeout(5.0)
            data, addr = self.sock.recvfrom(1024)
            
            if data == b"ACK":
                self.sock.settimeout(None)
                print("Successfully connected to master")
                return True
            return False
                
        except Exception as e:
            print(f"Failed to connect to master: {e}")
            return False

    def start_audio_playback(self):
        """Initialize audio playback"""
        try:
            self.audio = pyaudio.PyAudio()
            
            self.audio_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=22050,
                output=True,
                frames_per_buffer=1024
            )
            print("Audio playback initialized")
        except Exception as e:
            print(f"Audio setup error: {e}")

    def process_video_packet(self, encrypted_data: bytes, metadata: bytes, tag: bytes):
        """Decrypt and display video frame"""
        try:
            # Verify HMAC
            if not verify_hmac(metadata, encrypted_data, tag, self.salt):
                print("Video HMAC verification failed")
                return
            
            # Unpack metadata
            frame_no = struct.unpack('>Q', metadata[:8])[0]
            cipher_metadata = metadata[8:]
            
            print(f"Decrypting frame {frame_no} using chaotic pipeline...")
            
            # CHAOTIC VIDEO DECRYPTION PIPELINE
            seed_hash = derive_seed_hash(frame_no, self.salt, self.final_lorenz_state)
            
            decrypted_bytes = self.video_cipher.decrypt_frame(
                encrypted_data, cipher_metadata, seed_hash, frame_no
            )
            
            # Unpack frame dimensions from cipher metadata
            height, width, channels = struct.unpack('>III', cipher_metadata)
            frame_shape = (height, width, channels)
            
            # Convert bytes back to frame
            decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
            decrypted_frame = decrypted_array.reshape(frame_shape)
            
            # Convert RGB to BGR for display
            decrypted_bgr = cv2.cvtColor(decrypted_frame, cv2.COLOR_RGB2BGR)
            
            # Display decrypted frame
            cv2.imshow('Slave - Decrypted Video', decrypted_bgr)
            cv2.waitKey(1)
            
            # Save decrypted frame (first 3 frames only)
            if frame_no < 3:
                decrypted_path = f'output/video/decrypted/frame_{frame_no:06d}_decrypted.png'
                cv2.imwrite(decrypted_path, decrypted_bgr)
                print(f"Saved decrypted frame {frame_no}")
            
            self.video_frame_count += 1
            print(f"Successfully decrypted and displayed frame {frame_no}")
            
        except Exception as e:
            print(f"Video decryption error: {e}")
            import traceback
            traceback.print_exc()

    def process_audio_packet(self, encrypted_data: bytes, metadata: bytes, tag: bytes):
        """Decrypt and play audio segment"""
        if self.audio_saved:
            return
        
        try:
            # Verify HMAC
            if not verify_hmac(metadata, encrypted_data, tag, self.salt):
                print("Audio HMAC verification failed")
                return
            
            print("Decrypting audio using double DCT...")
            
            # Decrypt audio using double DCT
            decrypted_audio = decrypt_audio_dct_double(encrypted_data, metadata)
            
            # Play audio
            if self.audio_stream:
                try:
                    self.audio_stream.write(decrypted_audio.tobytes())
                    print("Audio playback started")
                except Exception as e:
                    print(f"Audio playback error: {e}")
            
            # Save decrypted audio
            decrypted_path = 'output/audio/segment_0000_decrypted.wav'
            with wave.open(decrypted_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(decrypted_audio.tobytes())
            
            print(f"Saved decrypted audio: {decrypted_path}")
            self.audio_saved = True
            
        except Exception as e:
            print(f"Audio decryption error: {e}")
            import traceback
            traceback.print_exc()

    def receive_packets(self):
        """Main packet receiving and processing loop"""
        print("Starting packet receiver...")
        
        while self.running:
            try:
                packet, addr = self.sock.recvfrom(65536)
                
                if len(packet) < 13:  # Minimum header size
                    continue
                
                # Parse packet header: [type(1)] [metadata_len(4)] [data_len(4)] [tag_len(4)]
                packet_type = packet[0:1]
                metadata_len, data_len, tag_len = struct.unpack('>III', packet[1:13])
                
                # Extract packet components
                metadata = packet[13:13+metadata_len]
                encrypted_data = packet[13+metadata_len:13+metadata_len+data_len]
                tag = packet[13+metadata_len+data_len:13+metadata_len+data_len+tag_len]
                
                if packet_type == b'V':
                    print(f"Received video packet (data: {len(encrypted_data)} bytes)")
                    self.process_video_packet(encrypted_data, metadata, tag)
                elif packet_type == b'A':
                    print(f"Received audio packet (data: {len(encrypted_data)} bytes)")
                    self.process_audio_packet(encrypted_data, metadata, tag)
                else:
                    print(f"Unknown packet type: {packet_type}")
                    
            except Exception as e:
                print(f"Packet receive error: {e}")
                time.sleep(0.1)

    def run(self):
        """Main execution function"""
        try:
            print("=== CHAOTIC VIDEO DECRYPTION SLAVE ===")
            print("Pipeline: Reverse Chained Diffusion → Reverse Pixel Permutation → Reverse Block Permutation → Inverse S-box")
            print("Chaos Source: Lorenz Attractor System")
            print("Audio: Double DCT Decryption")
            print()
            
            if not self.connect_to_master():
                print("Failed to connect to master. Exiting...")
                return
            
            self.running = True
            
            # Initialize audio playback
            self.start_audio_playback()
            
            print("Ready to receive encrypted packets...")
            print("Press Ctrl+C to stop")
            
            # Start packet receiver
            self.receive_packets()
            
        except KeyboardInterrupt:
            self.running = False
            print("\nShutting down slave system...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.sock:
            self.sock.close()
        
        if self.audio_stream:
            self.audio_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        cv2.destroyAllWindows()
        print("Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description='Chaotic Video Decryption Slave')
    parser.add_argument('--host', default='127.0.0.1', help='Master host')
    parser.add_argument('--port', type=int, default=5000, help='Master port')
    parser.add_argument('--pool-file', default='lorenz_pool.bin', help='Lorenz pool file (must match master)')
    parser.add_argument('--pool-size-mb', type=int, default=1, help='Pool size in MB if generating')
    parser.add_argument('--salt', required=True, help='Encryption salt (must match master)')
    
    args = parser.parse_args()
    
    slave = SlaveSystem(args)
    slave.run()


if __name__ == '__main__':
    main()