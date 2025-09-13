import argparse
import socket
import struct
import threading
import time
import cv2
import numpy as np
import pyaudio
from lorenz_cipher import (
    load_pool_bytes, derive_seed_hash, encrypt_data, compute_hmac
)
from lorenz_system import generate_pool_bytes


class MasterSystem:
    def __init__(self, args):
        self.args = args
        self.host = args.host
        self.port = args.port
        
        # Load or generate pool bytes
        if args.pool_file:
            self.pool_bytes = load_pool_bytes(args.pool_file)
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
                # Convert frame to bytes
                frame_bytes = frame.tobytes()
                
                # Generate seed for this frame
                seed_hash = derive_seed_hash(self.video_frame_no, self.salt)
                
                # Encrypt frame data
                ciphertext, metadata = encrypt_data(
                    frame_bytes, self.pool_bytes, seed_hash, 
                    self.block_size, self.rounds
                )
                
                # Add frame metadata (dimensions, frame_no)
                frame_metadata = (struct.pack('>III', frame.shape[1], frame.shape[0], self.video_frame_no) + 
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
                
                # Save original audio sample
                self.save_audio_sample(audio_data, self.audio_frame_no)
                
                # Generate seed for this audio packet
                seed_hash = derive_seed_hash(self.audio_frame_no, self.salt)
                
                # Encrypt audio data
                ciphertext, metadata = encrypt_data(
                    audio_data, self.pool_bytes, seed_hash,
                    self.block_size, self.rounds  
                )
                
                # Save encrypted audio sample
                if self.audio_frame_no in [1, 3, 5]:
                    filename = f'samples/audio/audio_{self.audio_frame_no}_encrypted.bin'
                    with open(filename, 'wb') as f:
                        f.write(ciphertext)
                    print(f"Saved encrypted audio sample {self.audio_frame_no}")
                
                # Add audio metadata (frame_no)
                audio_metadata = struct.pack('>I', self.audio_frame_no) + metadata
                
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
            
        print("Master system stopped")


def main():
    parser = argparse.ArgumentParser(description='Lorenz Cipher Master System')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='UDP port')
    parser.add_argument('--pool-file', help='Path to Lorenz pool bytes file (optional)')
    parser.add_argument('--pool-size-mb', type=int, default=10, help='Pool size in MB if generating (default: 10)')
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
