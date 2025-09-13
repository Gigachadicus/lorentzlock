# master.py - Video File Encryption System (no moviepy)
import argparse
import socket
import struct
import threading
import time
import cv2
import numpy as np
import os
import wave
import subprocess
from cipher_system import (
    VideoChaoticCipher, derive_seed_hash, encrypt_audio_dct_double, compute_hmac
)
from lorenz_system import generate_pool_bytes, load_pool_bytes


class MasterSystem:
    def __init__(self, args):
        self.args = args
        self.host = args.host
        self.port = args.port
        self.video_path = args.video_path
        
        # Load or generate Lorenz pool
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
        self.final_lorenz_state = (1.0, 1.0, 1.0)  # placeholder for Lorenz final state
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.clients = set()
        self.running = False
        
        # Initialize cipher
        self.video_cipher = VideoChaoticCipher(self.pool_bytes)
        
        # Create output directories
        os.makedirs('output/video/original', exist_ok=True)
        os.makedirs('output/video/encrypted', exist_ok=True)
        os.makedirs('output/audio', exist_ok=True)
        
        self.video_frame_no = 0
        self.audio_saved = False

    def start_server(self):
        """Start UDP server for clients"""
        self.sock.bind((self.host, self.port))
        self.running = True
        print(f"Master server started on {self.host}:{self.port}")
        
        # Start client discovery thread
        discovery_thread = threading.Thread(target=self.client_discovery, daemon=True)
        discovery_thread.start()

    def client_discovery(self):
        """Handle client discovery"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                if data == b"DISCOVER":
                    self.clients.add(addr)
                    self.sock.sendto(b"ACK", addr)
                    print(f"Client connected: {addr}")
            except:
                pass

    def broadcast_packet(self, packet_type: str, data: bytes, metadata: bytes, tag: bytes):
        """Broadcast encrypted packet to all clients"""
        type_byte = b'V' if packet_type == 'video' else b'A'
        header = type_byte + struct.pack('>III', len(metadata), len(data), len(tag))
        packet = header + metadata + data + tag
        
        for client_addr in list(self.clients):
            try:
                self.sock.sendto(packet, client_addr)
            except:
                self.clients.discard(client_addr)

    def extract_and_encrypt_audio(self):
        """Extract first 3 seconds of audio using ffmpeg, encrypt it"""
        if self.audio_saved:
            return
        
        try:
            print("Extracting audio from video using ffmpeg...")
            temp_audio_path = "temp_audio.wav"
            
            # Use ffmpeg to extract first 3 seconds, mono, 22050 Hz
            cmd = [
                "ffmpeg", "-y",
                "-i", self.video_path,
                "-t", "3",
                "-ac", "1",
                "-ar", "22050",
                temp_audio_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Read the audio file
            with wave.open(temp_audio_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Save original audio
            original_path = 'output/audio/segment_0000_original.wav'
            with wave.open(original_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(audio_data.tobytes())
            
            print(f"Saved original audio: {original_path}")
            
            # Encrypt audio using double DCT
            seed_hash = derive_seed_hash(0, self.salt, self.final_lorenz_state)
            encrypted_audio, audio_metadata = encrypt_audio_dct_double(audio_data, seed_hash)
            
            # Compute HMAC
            tag = compute_hmac(audio_metadata, encrypted_audio, self.salt)
            
            # Broadcast encrypted audio
            self.broadcast_packet('audio', encrypted_audio, audio_metadata, tag)
            print("Encrypted audio segment broadcasted")
            
            # Cleanup
            os.remove(temp_audio_path)
            self.audio_saved = True
            
        except Exception as e:
            print(f"Audio processing error: {e}")

    def process_video(self):
        """Load video file and encrypt frames"""
        if not os.path.exists(self.video_path):
            print(f"Video file not found: {self.video_path}")
            return
        
        print(f"Loading video: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Failed to open video file")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {total_frames} frames at {fps} FPS")
        
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
        
        # Extract and encrypt audio first
        self.extract_and_encrypt_audio()
        
        print("Starting video encryption...")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            # Resize frame to manageable size for encryption
            frame = cv2.resize(frame, (320, 240))
            height, width = frame.shape[:2]
            
            # Save original frame (first 3 frames only)
            if self.video_frame_no < 3:
                original_path = f'output/video/original/frame_{self.video_frame_no:06d}_original.png'
                cv2.imwrite(original_path, frame)
                print(f"Saved original frame {self.video_frame_no}")
            
            # Show original frame
            cv2.imshow('Master - Original Video', frame)
            cv2.waitKey(1)
            
            # Convert BGR to RGB for encryption
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_bytes = frame_rgb.tobytes()
            frame_shape = frame_rgb.shape
            
            # CHAOTIC VIDEO ENCRYPTION PIPELINE
            seed_hash = derive_seed_hash(self.video_frame_no, self.salt, self.final_lorenz_state)
            
            print(f"Encrypting frame {self.video_frame_no} using chaotic pipeline...")
            encrypted_data, cipher_metadata = self.video_cipher.encrypt_frame(
                frame_bytes, seed_hash, self.video_frame_no, frame_shape
            )
            
            # Visualize encrypted frame (first 3 frames only)
            if self.video_frame_no < 3:
                encrypted_array = np.frombuffer(encrypted_data, dtype=np.uint8)
                if len(encrypted_array) == len(frame_bytes):
                    encrypted_frame = encrypted_array.reshape(frame_shape)
                    encrypted_bgr = cv2.cvtColor(encrypted_frame, cv2.COLOR_RGB2BGR)
                    encrypted_path = f'output/video/encrypted/frame_{self.video_frame_no:06d}_encrypted.png'
                    cv2.imwrite(encrypted_path, encrypted_bgr)
                    print(f"Saved encrypted frame {self.video_frame_no}")
            
            # Prepare packet metadata
            packet_metadata = struct.pack('>Q', self.video_frame_no) + cipher_metadata
            
            # Compute HMAC authentication tag
            tag = compute_hmac(packet_metadata, encrypted_data, self.salt)
            
            # Broadcast encrypted frame
            self.broadcast_packet('video', encrypted_data, packet_metadata, tag)
            print(f"Broadcasted encrypted frame {self.video_frame_no}")
            
            self.video_frame_no += 1
            
            # Control playback speed
            time.sleep(frame_delay)
        
        cap.release()
        print("Video processing completed")

    def run(self):
        """Main execution function"""
        try:
            print("=== CHAOTIC VIDEO ENCRYPTION MASTER ===")
            print("Pipeline: S-box → Block Permutation → Pixel Permutation → Chained Diffusion")
            print("Chaos Source: Lorenz Attractor System")
            print("Audio: Double DCT Encryption")
            print()
            
            self.start_server()
            
            # Wait for clients to connect
            print("Waiting for clients to connect... (Press Enter to start encryption)")
            input()
            
            if not self.clients:
                print("No clients connected. Starting anyway...")
            else:
                print(f"Connected clients: {len(self.clients)}")
            
            # Process the video file
            self.process_video()
            
            print("All frames encrypted and sent. Press Ctrl+C to stop server...")
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.running = False
            print("\nShutting down master system...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.sock:
            self.sock.close()
        cv2.destroyAllWindows()
        print("Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description='Chaotic Video Encryption Master')
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--video-path', required=True, help='Path to input video file')
    parser.add_argument('--pool-file', default='lorenz_pool.bin', help='Lorenz pool file')
    parser.add_argument('--pool-size-mb', type=int, default=1, help='Pool size in MB if generating')
    parser.add_argument('--salt', required=True, help='Encryption salt')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    master = MasterSystem(args)
    master.run()


if __name__ == '__main__':
    main()
