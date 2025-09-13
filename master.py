# master.py
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
    load_pool_bytes, derive_seed_hash, encrypt_video_data, encrypt_audio_data, compute_hmac
)
from lorenz_system import generate_pool_bytes


class MasterSystem:
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
        self.frame_rate = args.frame_rate
        
        # Networking
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.clients = set()  # Set of client addresses
        self.running = False
        
        # Video capture
        self.video_cap = None
        self.video_thread = None
        
        # Audio capture  
        self.audio = None
        self.audio_thread = None
        
        # Frame counter for seed generation
        self.video_frame_no = 0
        self.audio_frame_no = 0
        self.audio_last_state = None
        
        # Create samples directories if they don't exist
        os.makedirs('samples/audio', exist_ok=True)
        os.makedirs('samples/video', exist_ok=True)
        
        print(f"Master initialized: {args.mode} mode, block_size={args.block_size}, rounds={args.rounds}")

    def start_server(self):
        """Start UDP server to accept client connections"""
        self.sock.bind((self.host, self.port))
        print(f"Master listening on {self.host}:{self.port}")
        self.running = True
        
        # Start client discovery thread
        discovery_thread = threading.Thread(target=self.client_discovery, daemon=True)
        discovery_thread.start()

    def client_discovery(self):
        """Handle client discovery and registration"""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                if data == b"DISCOVER":
                    self.clients.add(addr)
                    print(f"Client registered: {addr}")
                    # Send acknowledgment
                    self.sock.sendto(b"ACK", addr)
            except Exception as e:
                if self.running:
                    print(f"Client discovery error: {e}")

    def broadcast_packet(self, packet_type: str, data: bytes, metadata: bytes, tag: bytes):
        """Broadcast encrypted packet to all connected clients"""
        # Packet format: [type(1)] [metadata_len(4)] [data_len(4)] [tag_len(4)] [metadata] [data] [tag]
        type_byte = b'V' if packet_type == 'video' else b'A'
        header = (type_byte + 
                 struct.pack('>III', len(metadata), len(data), len(tag)))
        packet = header + metadata + data + tag
        
        for client_addr in self.clients.copy():
            try:
                self.sock.sendto(packet, client_addr)
            except Exception as e:
                print(f"Failed to send to {client_addr}: {e}")
                self.clients.discard(client_addr)

    def save_audio_sample(self, audio_data: bytes, frame_no: int):
        """Save original audio sample for debugging"""
        try:
            if frame_no in [1, 3, 5]:  # Save only specific frames
                filename = f'samples/audio/audio_{frame_no}_original.bin'
                with open(filename, 'wb') as f:
                    f.write(audio_data)
                print(f"Saved original audio sample {frame_no}")
        except Exception as e:
            print(f"Error saving audio sample: {e}")

    def start_video_capture(self):
        """Start video capture and encryption"""
        if self.mode not in ['video', 'both']:
            return
            
        self.video_cap = cv2.VideoCapture(0)
        if not self.video_cap.isOpened():
            print("Failed to open video capture")
            return
            
        # Set video properties
        self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        
        print("Video capture started")
        self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
        self.video_thread.start()

    def video_capture_loop(self):
        """Video capture and encryption loop"""
        frame_interval = 1.0 / self.frame_rate
        
        while self.running:
            ret, frame = self.video_cap.read()
            if not ret:
                print("Failed to read video frame")
                continue
            
            try:
                # Display original frame at master (LOCAL DISPLAY FIX)
                cv2.imshow('Master - Original Video', frame)
                cv2.waitKey(1)
                
                # Convert frame to RGB bytes for encryption
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bytes = frame_rgb.tobytes()
                
                # Generate seed for this frame
                seed_hash = derive_seed_hash(self.video_frame_no, self.salt)
                
                # Encrypt frame data using VIDEO pipeline (2D spatial)
                ciphertext, metadata = encrypt_video_data(
                    frame_bytes, self.pool_bytes, seed_hash, 
                    self.video_frame_no, frame_rgb.shape
                )
                
                # Add frame metadata (dimensions, frame_no)
                frame_metadata = (struct.pack('>III', frame_rgb.shape[1], frame_rgb.shape[0], self.video_frame_no) + 
                                metadata)
                
                # Compute HMAC
                tag = compute_hmac(frame_metadata, ciphertext, self.salt)
                
                # Broadcast to clients
                self.broadcast_packet('video', ciphertext, frame_metadata, tag)
                
                self.video_frame_no += 1
                
                # Rate limiting
                time.sleep(frame_interval)
                
            except Exception as e:
                print(f"Video processing error: {e}")

    def start_audio_capture(self):
        """Start audio capture and encryption"""
        if self.mode not in ['audio', 'both']:
            return
            
        self.audio = pyaudio.PyAudio()
        
        # Audio parameters
        chunk_size = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 44100
        
        try:
            stream = self.audio.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk_size
            )
            
            print("Audio capture started")
            self.audio_thread = threading.Thread(
                target=self.audio_capture_loop, 
                args=(stream, chunk_size), 
                daemon=True
            )
            self.audio_thread.start()
            
        except Exception as e:
            print(f"Failed to start audio capture: {e}")

    def audio_capture_loop(self, stream, chunk_size):
        """Audio capture and encryption loop"""
        while self.running:
            try:
                # Read audio data
                audio_data = stream.read(chunk_size, exception_on_overflow=False)
                audio_frame = np.frombuffer(audio_data, dtype=np.int16)
                
                # Save original audio sample
                self.save_audio_sample(audio_data, self.audio_frame_no)
                
                # Generate seed for this audio packet
                seed_hash = derive_seed_hash(self.audio_frame_no, self.salt)
                
                # Encrypt using AUDIO pipeline (1D temporal)
                ciphertext, audio_metadata = encrypt_audio_data(
                    audio_frame, self.pool_bytes, seed_hash,
                    self.audio_frame_no, self.audio_last_state
                )
                
                # Update audio state
                state_len = struct.unpack('>I', audio_metadata[8:12])[0]
                self.audio_last_state = audio_metadata[12:12+state_len] if state_len > 0 else None
                
                # Save encrypted audio sample
                if self.audio_frame_no in [1, 3, 5]:
                    filename = f'samples/audio/audio_{self.audio_frame_no}_encrypted.bin'
                    with open(filename, 'wb') as f:
                        f.write(ciphertext)
                    print(f"Saved encrypted audio sample {self.audio_frame_no}")
                
                # Compute HMAC
                tag = compute_hmac(audio_metadata, ciphertext, self.salt)
                
                # Broadcast to clients
                self.broadcast_packet('audio', ciphertext, audio_metadata, tag)
                
                self.audio_frame_no += 1
                
            except Exception as e:
                print(f"Audio processing error: {e}")

    def run(self):
        """Start the master system"""
        try:
            self.start_server()
            
            # Start capture based on mode
            if self.mode in ['video', 'both']:
                self.start_video_capture()
                
            if self.mode in ['audio', 'both']:
                self.start_audio_capture()
            
            # Keep main thread alive
            print("Master system running. Press Ctrl+C to stop.")
            print("Original video display in 'Master - Original Video' window.")
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping master system...")
            self.running = False
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        
        if self.sock:
            self.sock.close()
            
        if self.video_cap:
            self.video_cap.release()
            
        if self.audio:
            self.audio.terminate()
            
        cv2.destroyAllWindows()
        
        print("Master system stopped")


def main():
    parser = argparse.ArgumentParser(description='Lorenz Cipher Master System')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='UDP port')
    parser.add_argument('--pool-file', help='Path to Lorenz pool bytes file (optional)')
    parser.add_argument('--pool-size-mb', type=int, default=1, help='Pool size in MB if generating (default: 1)')
    parser.add_argument('--salt', required=True, help='HMAC salt string')
    parser.add_argument('--frame-rate', type=int, default=30, help='Video frame rate')
    parser.add_argument('--block-size', type=int, default=768, help='Encryption block size')
    parser.add_argument('--rounds', type=int, default=3, help='Number of encryption rounds')
    parser.add_argument('--mode', choices=['video', 'audio', 'both'], default='both', help='Capture mode')
    
    args = parser.parse_args()
    
    master = MasterSystem(args)
    master.run()


if __name__ == '__main__':
    main()