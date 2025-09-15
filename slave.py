import cv2
import pyaudio
import socket
import pickle
import threading
import numpy as np
import os
import time
import wave
import subprocess
import queue
from decryption import DecryptionSystem

class SlaveNode:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.decryption = DecryptionSystem()
        self.socket = None
        self.running = False
        
        # Audio playback setup
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 44100
        self.audio_chunk = 1024
        
        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            output=True,
            frames_per_buffer=self.audio_chunk
        )
        
        # Enhanced saving settings for slave
        self.save_data = True
        self.save_interval = 30  # Save every 30th frame/audio block
        self.frame_counter = 0
        self.audio_counter = 0
        
        # Directory structure for organized saving
        self.save_dir = 'slave_data'
        self.decrypted_frames_dir = os.path.join(self.save_dir, 'decrypted_frames')
        self.received_encrypted_frames_dir = os.path.join(self.save_dir, 'received_encrypted_frames')
        self.decrypted_audio_dir = os.path.join(self.save_dir, 'decrypted_audio')
        self.received_encrypted_audio_dir = os.path.join(self.save_dir, 'received_encrypted_audio')
        
        # Create all directories
        directories = [self.save_dir, self.decrypted_frames_dir, self.received_encrypted_frames_dir, 
                      self.decrypted_audio_dir, self.received_encrypted_audio_dir]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Audio saving setup
        self.decrypted_audio_buffer = []  # Buffer for decrypted audio chunks
        
        # Video display queue for thread-safe communication
        self.video_queue = queue.Queue(maxsize=10)
        
        print("Slave node initialized with comprehensive saving")
    
    def connect_to_master(self):
        """Connect to master and start receiving data"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"Connected to master at {self.host}:{self.port}")
            self.running = True
            
            # Start receiving thread
            receive_thread = threading.Thread(target=self.receive_loop, daemon=True)
            receive_thread.start()
            
            # Start video display thread
            display_thread = threading.Thread(target=self.video_display_loop, daemon=True)
            display_thread.start()
            
            # Keep main thread alive and handle keyboard interrupt
            print("Slave running. Press Ctrl+C to quit or focus on video window and press 'q'")
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nShutting down slave...")
                self.running = False
            
            receive_thread.join(timeout=1.0)
            display_thread.join(timeout=1.0)
            
        except ConnectionRefusedError:
            print("Could not connect to master. Make sure master is running.")
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.cleanup()
    
    def video_display_loop(self):
        """Dedicated thread for video display - runs in separate thread but handles OpenCV properly"""
        print("Starting video display thread...")
        cv2.namedWindow('Slave - Decrypted Video', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('Slave - Decrypted Video', 100, 100)
        
        # Show initial blank frame
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, 'Waiting for video...', (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Slave - Decrypted Video', blank_frame)
        cv2.waitKey(1)
        
        while self.running:
            try:
                # Get frame from queue with timeout
                frame = self.video_queue.get(timeout=0.1)
                
                if frame is not None:
                    # Display the frame
                    cv2.imshow('Slave - Decrypted Video', frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit key pressed in video window")
                        self.running = False
                        break
                    elif key == ord('r'):
                        print("Refreshing video window")
                        cv2.destroyWindow('Slave - Decrypted Video')
                        cv2.namedWindow('Slave - Decrypted Video', cv2.WINDOW_AUTOSIZE)
                
            except queue.Empty:
                # No new frame, just process window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed in video window")
                    self.running = False
                    break
                continue
            except Exception as e:
                print(f"Video display error: {e}")
                break
        
        print("Video display thread ended")
    
    def receive_loop(self):
        """Receive and process packets from master"""
        print("Starting packet reception...")
        buffer = b''
        
        while self.running:
            try:
                # Receive data in chunks
                chunk = self.socket.recv(8192)
                if not chunk:
                    print("Connection closed by master")
                    break
                
                buffer += chunk
                
                # Process complete packets
                while len(buffer) >= 4:
                    # Get packet size from first 4 bytes
                    packet_size = int.from_bytes(buffer[:4], 'big')
                    
                    # Check if we have complete packet
                    if len(buffer) < packet_size + 4:
                        break
                    
                    # Extract packet data
                    packet_data = buffer[4:packet_size + 4]
                    buffer = buffer[packet_size + 4:]
                    
                    try:
                        # Deserialize packet
                        packet = pickle.loads(packet_data)
                        
                        # Process based on packet type
                        if packet['type'] == 'video':
                            self.handle_video_packet(packet['data'])
                        elif packet['type'] == 'audio':
                            self.handle_audio_packet(packet['data'])
                        
                    except Exception as e:
                        print(f"Packet processing error: {e}")
                        continue
                
            except ConnectionError:
                print("Connection lost to master")
                break
            except Exception as e:
                print(f"Error in receive loop: {e}")
                break
        
        self.running = False
        print("Receive loop ended")
    
    def handle_video_packet(self, encrypted_packet):
        """Decrypt and queue video frame for display"""
        try:
            # Save encrypted packet as JPG visualization
            if self.save_data and self.frame_counter % self.save_interval == 0:
                self.save_received_encrypted_frame_as_jpg(encrypted_packet, self.frame_counter)
            
            # Decrypt frame
            decrypted_frame_data = self.decryption.decrypt_frame(encrypted_packet)
            
            # Convert bytes to numpy array and decode JPEG
            frame_array = np.frombuffer(decrypted_frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                print(f"Frame {self.frame_counter}: Successfully decrypted, shape: {frame.shape}")
                
                # Save decrypted frame as JPG
                if self.save_data and self.frame_counter % self.save_interval == 0:
                    self.save_decrypted_frame_as_jpg(frame, self.frame_counter)
                
                # Put frame in display queue (non-blocking)
                try:
                    self.video_queue.put_nowait(frame)
                except queue.Full:
                    # If queue is full, remove oldest frame and add new one
                    try:
                        self.video_queue.get_nowait()
                        self.video_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
                
            else:
                print(f"Frame {self.frame_counter}: Failed to decode video frame")
            
            self.frame_counter += 1
                
        except Exception as e:
            print(f"Error handling video packet {self.frame_counter}: {e}")
            self.frame_counter += 1
    
    def handle_audio_packet(self, encrypted_packet):
        """Decrypt and play audio, save both encrypted and decrypted data"""
        try:
            # Save encrypted packet as MP4
            if self.save_data and self.audio_counter % self.save_interval == 0:
                self.save_received_encrypted_audio_as_mp4(encrypted_packet, self.audio_counter)
            
            # Decrypt audio
            decrypted_audio_data = self.decryption.decrypt_audio(encrypted_packet)
            
            # Save decrypted audio data for later MP4 conversion
            if self.save_data:
                self.decrypted_audio_buffer.append(decrypted_audio_data)
            
            # Play audio
            self.audio_stream.write(decrypted_audio_data)
            
            # Save decrypted audio as MP4 periodically (every 100 chunks ~ 2.3 seconds)
            if self.save_data and len(self.decrypted_audio_buffer) >= 100:
                self.save_decrypted_audio_as_mp4()
            
            self.audio_counter += 1
            
        except Exception as e:
            print(f"Error handling audio packet: {e}")
    
    def save_decrypted_frame_as_jpg(self, frame, frame_num):
        """Save decrypted frame as JPG"""
        try:
            filename = os.path.join(self.decrypted_frames_dir, f'decrypted_frame_{frame_num:06d}.jpg')
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"Saved decrypted frame {frame_num} as JPG")
        except Exception as e:
            print(f"Error saving decrypted frame: {e}")
    
    def save_received_encrypted_frame_as_jpg(self, encrypted_packet, frame_num):
        """Save received encrypted frame packet as JPG visualization"""
        try:
            # Save the binary data for later analysis
            filename_bin = os.path.join(self.received_encrypted_frames_dir, f'received_encrypted_frame_{frame_num:06d}.bin')
            
            save_data = {
                'encrypted_packet': encrypted_packet,
                'timestamp': time.time(),
                'frame_number': frame_num,
                'received_at_slave': True
            }
            
            with open(filename_bin, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Create JPG visualization of encrypted data
            ciphertext = encrypted_packet['ciphertext']
            
            # Convert encrypted data to image for visualization
            data_size = len(ciphertext)
            side_length = int(np.ceil(np.sqrt(data_size)))
            padded_size = side_length * side_length
            
            # Pad with zeros if necessary
            padded_data = np.frombuffer(ciphertext, dtype=np.uint8)
            if len(padded_data) < padded_size:
                padding = np.zeros(padded_size - len(padded_data), dtype=np.uint8)
                padded_data = np.concatenate([padded_data, padding])
            
            # Reshape to 2D array (grayscale image)
            encrypted_image = padded_data[:padded_size].reshape(side_length, side_length)
            
            filename_jpg = os.path.join(self.received_encrypted_frames_dir, f'received_encrypted_frame_{frame_num:06d}.jpg')
            cv2.imwrite(filename_jpg, encrypted_image)
            
            print(f"Saved received encrypted frame {frame_num} as binary and JPG visualization")
            
        except Exception as e:
            print(f"Error saving received encrypted frame: {e}")
    
    def save_received_encrypted_audio_as_mp4(self, encrypted_packet, audio_num):
        """Save received encrypted audio packet as MP4"""
        try:
            # Save binary data
            filename_bin = os.path.join(self.received_encrypted_audio_dir, f'received_encrypted_audio_{audio_num:06d}.bin')
            
            save_data = {
                'encrypted_packet': encrypted_packet,
                'timestamp': time.time(),
                'audio_number': audio_num,
                'received_at_slave': True
            }
            
            with open(filename_bin, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Convert encrypted bytes to audio format for MP4
            ciphertext = encrypted_packet['ciphertext']
            
            # Convert encrypted bytes to int16 audio samples for MP4 conversion
            audio_data = np.frombuffer(ciphertext[:len(ciphertext)//2*2], dtype=np.uint8)
            
            if len(audio_data) == 0:
                print(f"No audio data for encrypted audio {audio_num}")
                return
            
            # Convert to int16 format (map 0-255 to -32768 to 32767)
            audio_samples = ((audio_data.astype(np.float32) / 255.0) * 65535 - 32768).astype(np.int16)
            
            # Create temporary WAV file
            wav_filename = os.path.join(self.received_encrypted_audio_dir, f'temp_encrypted_audio_{audio_num:06d}.wav')
            mp4_filename = os.path.join(self.received_encrypted_audio_dir, f'received_encrypted_audio_{audio_num:06d}.mp4')
            
            with wave.open(wav_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.audio_rate)
                wav_file.writeframes(audio_samples.tobytes())
            
            # Convert to MP4
            try:
                subprocess.run(['ffmpeg', '-i', wav_filename, '-c:a', 'aac', '-y', mp4_filename], 
                             check=True, capture_output=True)
                os.remove(wav_filename)
                print(f"Saved received encrypted audio {audio_num} as MP4")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"ffmpeg not found, saved encrypted audio {audio_num} as WAV: {wav_filename}")
            
        except Exception as e:
            print(f"Error saving received encrypted audio: {e}")
    
    def save_decrypted_audio_as_mp4(self):
        """Save accumulated decrypted audio buffer as MP4 file"""
        try:
            # Combine audio chunks
            combined_audio = b''.join(self.decrypted_audio_buffer)
            
            # Create a temporary WAV file
            wav_filename = os.path.join(self.decrypted_audio_dir, f'temp_decrypted_audio_{int(time.time())}.wav')
            mp4_filename = os.path.join(self.decrypted_audio_dir, f'decrypted_audio_{int(time.time())}.mp4')
            
            # Save as WAV first
            with wave.open(wav_filename, 'wb') as wav_file:
                wav_file.setnchannels(self.audio_channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wav_file.setframerate(self.audio_rate)
                wav_file.writeframes(combined_audio)
            
            # Convert WAV to MP4 using ffmpeg (if available)
            try:
                subprocess.run(['ffmpeg', '-i', wav_filename, '-c:a', 'aac', '-y', mp4_filename], 
                             check=True, capture_output=True)
                os.remove(wav_filename)  # Remove temporary WAV file
                print(f"Saved decrypted audio as MP4: {mp4_filename}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If ffmpeg is not available, keep the WAV file
                print(f"ffmpeg not found, saved decrypted audio as WAV: {wav_filename}")
            
            # Clear the buffer
            self.decrypted_audio_buffer = []
            
        except Exception as e:
            print(f"Error saving decrypted audio: {e}")
            self.decrypted_audio_buffer = []  # Clear buffer even if saving failed
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up slave resources...")
        self.running = False
        
        # Save any remaining audio buffer
        if self.decrypted_audio_buffer:
            self.save_decrypted_audio_as_mp4()
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        if self.socket:
            self.socket.close()
        
        cv2.destroyAllWindows()
        print("Slave cleanup complete")

if __name__ == "__main__":
    # Check if Lorenz states exist
    if not os.path.exists('lorenz_states.pkl'):
        print("Error: lorenz_states.pkl not found!")
        print("Please run: python lorenz.py")
        exit(1)
    
    slave = SlaveNode()
    try:
        slave.connect_to_master()
    except KeyboardInterrupt:
        print("\nShutting down slave...")
    finally:
        slave.cleanup()