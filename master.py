import cv2
import pyaudio
import socket
import pickle
import threading
import time
import os
import numpy as np
import wave
import subprocess
from encryption import EncryptionSystem

class MasterNode:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.encryption = EncryptionSystem()
        self.socket = None
        self.client_socket = None
        self.running = False
        
        # Video capture setup
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        # Audio capture setup
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 44100
        self.audio_chunk = 1024
        
        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
            format=self.audio_format,
            channels=self.audio_channels,
            rate=self.audio_rate,
            input=True,
            frames_per_buffer=self.audio_chunk
        )
        
        # Enhanced saving settings for master
        self.save_data = True
        self.save_interval = 30  # Save every 30th frame/audio block
        self.frame_counter = 0
        self.audio_counter = 0
        
        # Directory structure for organized saving
        self.save_dir = 'master_data'
        self.pre_encrypted_frames_dir = os.path.join(self.save_dir, 'pre_encrypted_frames')
        self.encrypted_frames_dir = os.path.join(self.save_dir, 'encrypted_frames')
        self.raw_audio_dir = os.path.join(self.save_dir, 'raw_audio')
        self.encrypted_audio_dir = os.path.join(self.save_dir, 'encrypted_audio')
        
        # Create all directories
        directories = [self.save_dir, self.pre_encrypted_frames_dir, self.encrypted_frames_dir, 
                      self.raw_audio_dir, self.encrypted_audio_dir]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Audio saving setup
        self.raw_audio_buffer = []  # Buffer for raw audio chunks
        
        print("Master node initialized with comprehensive saving")
    
    def start_server(self):
        """Start the master server and wait for slave connection"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print(f"Master listening on {self.host}:{self.port}")
        
        try:
            conn, addr = self.socket.accept()
            print(f"Slave connected from {addr}")
            self.client_socket = conn
            self.running = True
            
            # Start video and audio streaming threads
            video_thread = threading.Thread(target=self.video_loop, daemon=True)
            audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
            
            video_thread.start()
            audio_thread.start()
            
            # Keep main thread alive and handle keyboard interrupt
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nShutting down master...")
                self.running = False
            
            video_thread.join(timeout=1.0)
            audio_thread.join(timeout=1.0)
            
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.cleanup()
    
    def video_loop(self):
        """Capture, encrypt, and send video frames"""
        print("Starting video capture and encryption...")
        
        while self.running:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Display local camera feed
                cv2.imshow('Master - Camera Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                
                # Save pre-encrypted frame as JPG
                if self.save_data and self.frame_counter % self.save_interval == 0:
                    self.save_pre_encrypted_frame(frame, self.frame_counter)
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_data = buffer.tobytes()
                
                # Encrypt frame
                encrypted_frame = self.encryption.encrypt_frame(frame_data)
                
                # Save encrypted frame as JPG (visualization of encrypted data)
                if self.save_data and self.frame_counter % self.save_interval == 0:
                    self.save_encrypted_frame_as_jpg(encrypted_frame, self.frame_counter)
                
                # Create packet
                packet = {
                    'type': 'video',
                    'data': encrypted_frame
                }
                
                # Send packet
                self.send_packet(packet)
                
                self.frame_counter += 1
                
                # Control frame rate
                time.sleep(1/30)  # ~30 FPS
                
            except Exception as e:
                print(f"Video loop error: {e}")
                break
        
        print("Video loop ended")
    
    def audio_loop(self):
        """Capture, encrypt, and send audio"""
        print("Starting audio capture and encryption...")
        
        while self.running:
            try:
                # Capture audio chunk
                audio_data = self.audio_stream.read(self.audio_chunk, exception_on_overflow=False)
                
                # Save raw audio data for later MP4 conversion
                if self.save_data:
                    self.raw_audio_buffer.append(audio_data)
                
                # Encrypt audio
                encrypted_audio = self.encryption.encrypt_audio(audio_data)
                
                # Save encrypted audio as MP4 (every 30th audio chunk)
                if self.save_data and self.audio_counter % self.save_interval == 0:
                    self.save_encrypted_audio_as_mp4(encrypted_audio, self.audio_counter)
                    print(f"Saving encrypted audio {self.audio_counter}")
                
                # Create packet
                packet = {
                    'type': 'audio',
                    'data': encrypted_audio
                }
                
                # Send packet
                self.send_packet(packet)
                
                # Save raw audio as MP4 periodically (every 100 chunks ~ 2.3 seconds)
                if self.save_data and len(self.raw_audio_buffer) >= 100:
                    print(f"Saving raw audio buffer with {len(self.raw_audio_buffer)} chunks")
                    self.save_raw_audio_as_mp4()
                
                self.audio_counter += 1
                
            except Exception as e:
                print(f"Audio loop error: {e}")
                break
        
        print("Audio loop ended")
    
    def send_packet(self, packet):
        """Send packet to slave with size header"""
        try:
            if self.client_socket:
                serialized_packet = pickle.dumps(packet)
                packet_size = len(serialized_packet)
                size_header = packet_size.to_bytes(4, byteorder='big')
                self.client_socket.sendall(size_header + serialized_packet)
        except Exception as e:
            print(f"Send error: {e}")
            self.running = False
    
    def save_pre_encrypted_frame(self, frame, frame_num):
        """Save pre-encrypted frame as JPG"""
        try:
            filename = os.path.join(self.pre_encrypted_frames_dir, f'pre_encrypted_frame_{frame_num:06d}.jpg')
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"Saved pre-encrypted frame {frame_num} as JPG")
        except Exception as e:
            print(f"Error saving pre-encrypted frame: {e}")
    
    def save_encrypted_frame_as_jpg(self, encrypted_packet, frame_num):
        """Save encrypted frame data as JPG (visualization of encrypted data)"""
        try:
            # Convert encrypted data to image for visualization
            ciphertext = encrypted_packet['ciphertext']
            
            # Pad data to make it square-ish
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
            
            filename = os.path.join(self.encrypted_frames_dir, f'encrypted_frame_{frame_num:06d}.jpg')
            cv2.imwrite(filename, encrypted_image)
            print(f"Saved encrypted frame {frame_num} visualization as JPG")
            
        except Exception as e:
            print(f"Error saving encrypted frame as JPG: {e}")
    
    def save_raw_audio_as_mp4(self):
        """Save accumulated raw audio buffer as MP4 file"""
        try:
            # Combine audio chunks
            combined_audio = b''.join(self.raw_audio_buffer)
            
            # Create a temporary WAV file
            wav_filename = os.path.join(self.raw_audio_dir, f'temp_raw_audio_{int(time.time())}.wav')
            mp4_filename = os.path.join(self.raw_audio_dir, f'raw_audio_{int(time.time())}.mp4')
            
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
                print(f"Saved raw audio as MP4: {mp4_filename}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If ffmpeg is not available, keep the WAV file
                print(f"ffmpeg not found, saved raw audio as WAV: {wav_filename}")
            
            # Clear the buffer
            self.raw_audio_buffer = []
            
        except Exception as e:
            print(f"Error saving raw audio: {e}")
            self.raw_audio_buffer = []  # Clear buffer even if saving failed
    
    def save_encrypted_audio_as_mp4(self, encrypted_packet, audio_num):
        """Save encrypted audio as MP4 (convert encrypted bytes to audio format)"""
        try:
            # Extract encrypted data
            ciphertext = encrypted_packet['ciphertext']
            
            # Convert encrypted bytes to int16 audio samples for MP4 conversion
            # This creates a "sound" from the encrypted data for analysis
            audio_data = np.frombuffer(ciphertext[:len(ciphertext)//2*2], dtype=np.uint8)
            
            # Convert to int16 format (map 0-255 to -32768 to 32767)
            audio_samples = ((audio_data.astype(np.float32) / 255.0) * 65535 - 32768).astype(np.int16)
            
            # Ensure we have audio data
            if len(audio_samples) == 0:
                print(f"No audio samples for encrypted audio {audio_num}")
                return
            
            # Create temporary WAV file
            wav_filename = os.path.join(self.encrypted_audio_dir, f'temp_encrypted_audio_{audio_num:06d}.wav')
            mp4_filename = os.path.join(self.encrypted_audio_dir, f'encrypted_audio_{audio_num:06d}.mp4')
            
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
                print(f"Saved encrypted audio {audio_num} as MP4")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"ffmpeg not found, saved encrypted audio {audio_num} as WAV: {wav_filename}")
                
        except Exception as e:
            print(f"Error saving encrypted audio as MP4: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up master resources...")
        self.running = False
        
        # Save any remaining audio buffer
        if hasattr(self, 'raw_audio_buffer') and self.raw_audio_buffer:
            self.save_raw_audio_as_mp4()
        
        if self.video_capture:
            self.video_capture.release()
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        if self.client_socket:
            self.client_socket.close()
        
        if self.socket:
            self.socket.close()
        
        cv2.destroyAllWindows()
        print("Master cleanup complete")

if __name__ == "__main__":
    # Check if Lorenz states exist
    if not os.path.exists('lorenz_states.pkl'):
        print("Error: lorenz_states.pkl not found!")
        print("Please run: python lorenz.py")
        exit(1)
    
    master = MasterNode()
    try:
        master.start_server()
    except KeyboardInterrupt:
        print("\nShutting down master...")
    finally:
        master.cleanup()