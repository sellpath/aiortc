import fractions
import os
import wave
import unittest
import hashlib
import wave 
import numpy as np
from scipy import signal
import librosa
import av
from aiortc.codecs import get_decoder, get_encoder
from aiortc.jitterbuffer import JitterFrame
from aiortc.rtcrtpparameters import RTCRtpCodecParameters

def audio_similarity(file1, file2, sr=22050):
    # Load audio files
    y1, sr = librosa.load(file1, sr=sr)
    y2, sr = librosa.load(file2, sr=sr)
    
    # Extract features
    def extract_features(y):
        # Get MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Get spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Get chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        return np.vstack([mfcc, contrast, chroma])
    
    # Get features
    f1 = extract_features(y1)
    f2 = extract_features(y2)
    
    # Ensure same length
    min_len = min(f1.shape[1], f2.shape[1])
    f1 = f1[:, :min_len]
    f2 = f2[:, :min_len]
    
    # Calculate similarity (cosine distance)
    similarity = np.dot(f1.flatten(), f2.flatten()) / (
        np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-10)
    
    return similarity


def write_wav_file(frames, output_path, target_sample_rate=None, target_number_of_channels=None, target_sample_width=2):
    """
    Write audio frames to a WAV file with the specified properties using PyAV.
    
    Args:
        frames: List of av.AudioFrame objects to write
        output_path: Path to the output WAV file
        target_sample_rate: Target sample rate in Hz (default: use first frame's sample rate)
        target_number_of_channels: Target number of channels (1 for mono, 2 for stereo)
        target_sample_width: Sample width in bytes (default: 2 for 16-bit)
    """
    if not frames:
        print(f"write_wav_file: No frames for {output_path}")
        return

    # Use first frame's properties as defaults
    first_frame = frames[0]
    if target_sample_rate is None:
        target_sample_rate = first_frame.sample_rate
    if target_number_of_channels is None:
        target_number_of_channels = len(first_frame.layout.channels)
    
    # Map sample width to AV format
    format_map = {1: 's16', 2: 's16', 3: 's32', 4: 'flt'}
    output_format = format_map.get(target_sample_width, 's16')
    output_layout = 'mono' if target_number_of_channels == 1 else 'stereo'
    
    # Create output container
    output_container = av.open(output_path, 'w')
    output_stream = output_container.add_stream('pcm_s16le', rate=target_sample_rate)
    output_stream.layout = output_layout
    output_stream.sample_rate = target_sample_rate
    
    # Create resampler if needed
    resampler = None
    if (first_frame.sample_rate != target_sample_rate or 
        len(first_frame.layout.channels) != target_number_of_channels):
        print(f"Resampling frames from {first_frame.sample_rate}Hz to {target_sample_rate}Hz, ")
        resampler = av.AudioResampler(
            format=output_format,
            layout=output_layout,
            rate=target_sample_rate,
            frame_size=960  # 20ms at 48kHz
        )
    
    # Process each frame
    for frame in frames:
        # Apply resampling and channel conversion if needed
        if resampler is not None:
            for resampled_frame in resampler.resample(frame):
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)
        else:
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
    
    # Flush the resampler to get any remaining frames
    if resampler is not None:
        for resampled_frame in resampler.resample(None):
            for packet in output_stream.encode(resampled_frame):
                output_container.mux(packet)
    
    # Flush the encoder
    for packet in output_stream.encode(None):
        output_container.mux(packet)
    
    # Close the container
    output_container.close()

def get_audio_info(file_path):
    with av.open(file_path) as container:
        audio_stream = container.streams.audio[0]
        total_samples = 0
        frame_count = 0
        
        # First pass: count frames and samples
        for frame in container.decode(audio=0):
            if frame.samples > 0:
                total_samples += frame.samples
                frame_count += 1
        
        # Calculate duration from samples and sample rate
        duration_seconds = total_samples / float(audio_stream.rate) if audio_stream.rate > 0 else 0
        
        # Reset file position
        container.seek(0)
        print(f"File: {file_path}")
        print(f"  Duration: {float(container.duration / av.time_base):.2f} seconds")
        print(f"  Format: {container.format.name}")
        print(f"  Streams: {len(container.streams)}")
        print("\nAudio Stream:")
        print(f"  Codec: {audio_stream.codec_context.name}")
        print(f"  Sample Rate: {audio_stream.rate} Hz")
        print(f"  Bit Rate: {getattr(audio_stream, 'bit_rate', 'N/A')} bps")
        print(f"  Frame Size: {getattr(audio_stream, 'frame_size', 'N/A')} samples")
        print(f"  Channels: {audio_stream.channels}")
        print(f"  Layout: {audio_stream.layout.name if hasattr(audio_stream, 'layout') else 'N/A'}")
        print(f"  Sample Format: {audio_stream.format.name if hasattr(audio_stream, 'format') else 'N/A'}")
        print(f"  Frames: {getattr(audio_stream, 'frames', 'N/A')}")
        print(f"  Duration: {float(audio_stream.duration * audio_stream.time_base):.2f} seconds")
        print(f"  Stream Type: {getattr(audio_stream, 'type', 'N/A')}")
        print(f"  Frames: {frame_count}")
        print("\nAudio Stream (decoded frames):")
        print(f"  Total Samples: {total_samples}")
        print(f"  Duration: {duration_seconds:.2f} seconds")
        print(f"  Calculated Duration: {total_samples / audio_stream.rate:.2f} seconds")

def get_audioframe_info(frame: av.AudioFrame, prefix: str = ""):
    print(f"{prefix}AudioFrame:")
    print(f"{prefix}  Format: {frame.format.name}")
    print(f"{prefix}  Layout: {frame.layout.name}")
    print(f"{prefix}  Sample Rate: {frame.sample_rate} Hz")
    print(f"{prefix}  Channels: {frame.layout.channels}")
    print(f"{prefix}  Samples: {frame.samples}")
    print(f"{prefix}  Duration: {frame.duration} samples")
    print(f"{prefix}  PTS: {frame.pts}")
    print(f"{prefix}  Time Base: {frame.time_base}")
    print("\n")
    # print(f"{prefix}  Duration: {frame.duration * frame.time_base:.2f} seconds")

def encode_and_decode(output_pts: int, frame: av.AudioFrame, encoder, decoder)-> list[av.AudioFrame]:
    # Encode to Opus
    # print(f"Processing frame {output_pts}  Input frame: samples={frame.samples}, channels={frame.layout.channels}, "
    #       f"sample_rate={frame.sample_rate}, format={frame.format} layout={frame.layout}")
    
    packages, timestamp = encoder.encode(frame)
    if not packages:
        print("No packages from encoder, might silent?")
        return [], output_pts

    # Depacketize
    data = b''.join(packages)

    # Decode back to PCM
    decoded_frames = decoder.decode(JitterFrame(data=data, timestamp=timestamp))
    result_frames = []
    for df in decoded_frames:
        if isinstance(df, av.AudioFrame):
            # Update the decoded frame's PTS and time_base for 48kHz output
            df.pts = output_pts * frame.samples  # This assumes each input frame starts a new segment
            df.time_base = fractions.Fraction(1, df.sample_rate)
            result_frames.append(df)
            output_pts += df.samples  # Update the PTS for the next frame
        else:
            print(f"Warning: Non-audio frame in decoded output at frame {output_pts}")

    return result_frames, output_pts # should be 48khz wave

def delet_after_test(output_path):
    if os.path.exists(output_path):
        print(f"Output file exists: {output_path}")
        os.remove(output_path)

class OpusAudioTest(unittest.TestCase):
    """Test Opus audio codec with WAV file processing."""

    def test_opus_wav_roundtrip_8k(self):
        """Test encoding a 8kHz WAV file to Opus and back to WAV at 48kHz."""
        self._test_opus_wav_roundtrip('sample-s16le-8k-mono.wav', test_intermediate_resample_rate=8000, test_intermediate_resample_layout="mono")

    def test_opus_wav_roundtrip_48k(self):
        """Test encoding a 8kHz WAV file intermediate 48khz stereo to Opus and back to WAV at 48kHz."""
        self._test_opus_wav_roundtrip('sample-s16le-8k-mono.wav', test_intermediate_resample_rate=48000, test_intermediate_resample_layout="stereo")

    def _test_opus_wav_roundtrip(self, input_filename, test_intermediate_resample_rate=48000, test_intermediate_resample_layout="stereo", 
                                 codec_mime_type="audio/opus", codec_clock_rate=48000, codec_channels=2, codec_payload_type=100):
        """Test encoding a WAV file to Opus and back to WAV at 48kHz."""
        # Input WAV file (8kHz mono 16-bit)
        input_path = os.path.join(os.path.dirname(__file__), input_filename)
        # Output WAV file (48kHz mono 16-bit)
        output_path = os.path.join(os.path.dirname(__file__), f"output_{input_filename}")
         
        input_frames = []

        output_sample_rate = 48000  # Opus always works at 48kHz
        output_chunk_size_for_20ms = int(0.02 * output_sample_rate)  # 20ms at 48kHz
        output_sample_width = 2  # 16-bit
        output_number_of_channels = 2

        try:
            # Read input WAV file
            with wave.open(input_path, 'rb') as wav_in:
                input_number_of_channels = wav_in.getnchannels()
                input_sample_rate = wav_in.getframerate()
                input_frames_count = wav_in.getnframes()
                input_sample_width = wav_in.getsampwidth() 
            
            with av.open(input_path) as input_container:
                input_stream = input_container.streams.audio[0]
                input_container.seek(0)
                print(f"Input Stream layout(will override): {input_stream.layout}") 
                for frame in input_container.decode(audio=0):
                    input_frames.append(frame)
            
            print(f"\n{'='*80}")
            print(f"Processing: {input_filename}")
            print(f"Input: {input_number_of_channels} channel(s), {input_sample_rate} Hz, {len(input_frames)} bytes")
            print(f"Input frames count: {input_frames_count}")

            # resample is important step if input audio layout is not stereo or mono
            resampler_48k = av.AudioResampler(
                format='s16',
                layout=test_intermediate_resample_layout, # rtc codecs expect stereo or mono 
                rate=test_intermediate_resample_rate,
                frame_size=int(0.02 * test_intermediate_resample_rate) # frame_size=960  # 20ms at 48kHz
            )

            # Configure Opus codec (always 48kHz output)
            codec = RTCRtpCodecParameters(
                mimeType=codec_mime_type,
                clockRate=codec_clock_rate,  # Opus always works at 48kHz internally
                channels=codec_channels,
                payloadType=100
            )
            
            # Create encoder and decoder
            encoder = get_encoder(codec)
            decoder = get_decoder(codec)

            # Process audio in chunks (20ms)
            input_chunk_size_for_20ms = int(0.02 * input_sample_rate)  # 20ms chunks at input sample rate
            input_chunk_size_bytes = input_chunk_size_for_20ms * input_sample_width * input_number_of_channels

            resample_input_frames = []
            
            output_frames = []

            output_pts = 0  # Separate PTS counter for output frames
            output_chunk_size_bytes = output_chunk_size_for_20ms * output_sample_width * output_number_of_channels


            _debug_first_input_frame = True
            _debug_first_output_frame = True

            for frame in input_frames:
                resample_frames= None
                if test_intermediate_resample_rate:
                    resample_frames = resampler_48k.resample(frame)
                    resample_input_frames.extend(resample_frames)

                if _debug_first_input_frame and frame:
                    _debug_first_input_frame = False
                    get_audioframe_info(frame, "Input")
                    if test_intermediate_resample_rate:
                        get_audioframe_info(resample_frames[0], "Input Resampled")
                
                decoded_frames = None
                for resample_frame in resample_frames:
                    decoded_frames, output_pts = encode_and_decode(output_pts,resample_frame, encoder, decoder)
                    output_frames.extend(decoded_frames)

                if _debug_first_output_frame and decoded_frames:
                    _debug_first_output_frame = False
                    get_audioframe_info(decoded_frames[0], "Output")

                
        
            output_path_input_itself= output_path.replace('.wav', '_as_input_itself.wav')
            write_wav_file(input_frames, output_path_input_itself)  
            print(f"Input resampled written to {output_path_input_itself}")
            print(f"Processed {len(input_frames)} frames at {input_sample_rate}Hz")

            if test_intermediate_resample_rate and resample_input_frames is not None:
                output_path_input_resampled = output_path.replace('.wav', '_as_input_resampled.wav')
                write_wav_file(resample_input_frames, output_path_input_resampled)  
                print(f"Input resampled written to {output_path_input_resampled}")
                print(f"Processed {len(resample_input_frames)} frames at {output_sample_rate}Hz")

            write_wav_file(output_frames, output_path)
            print(f"Output written to {output_path}")
            print(f"Processed {len(output_frames)} frames at {output_sample_rate}Hz")
            
            output_path_original_prop = output_path.replace('.wav', '_as_original_audio_property.wav')
            write_wav_file(output_frames, output_path_original_prop, input_sample_rate, input_number_of_channels, input_sample_width)
            print(f"Output written to {output_path_original_prop}")


            print(f"Output with original properties written to {output_path_original_prop}")
            print(f"Processed {len(output_frames)} frames at {input_sample_rate}Hz")

            print("\nAudio Stream (input frames):")
            get_audio_info(input_path)
            if resample_input_frames:
                print("\nAudio Stream (input frames resampled):")
                get_audio_info(output_path_input_resampled)
            print("\nAudio Stream (decoded frames):")
            get_audio_info(output_path)
            print("\nAudio Stream (decoded frames) in original properties:")
            get_audio_info(output_path_original_prop)
            
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)
            self.assertTrue(os.path.exists(output_path_original_prop))
            self.assertGreater(os.path.getsize(output_path_original_prop), 0)

            similarity = audio_similarity(input_path, output_path)
            print(f"input to output Similarity: {similarity}")
            self.assertGreater(similarity, 0.99)

            similarity = audio_similarity(input_path, output_path_original_prop)
            print(f"input to output with original properties Similarity: {similarity}")
            self.assertGreater(similarity, 0.99)

            similarity = audio_similarity(input_path, output_path_input_itself)
            print(f"input to input read Similarity: {similarity}")
            self.assertGreater(similarity, 0.99)

            similarity = audio_similarity(input_path, output_path_input_resampled)
            print(f"input to input resampled Similarity: {similarity}")
            self.assertGreater(similarity, 0.99)

        finally:
            # Clean up output file
            delet_after_test(output_path)
            delet_after_test(output_path_input_itself)
            delet_after_test(output_path_original_prop)
            delet_after_test(output_path_input_resampled)


if __name__ == '__main__':
    unittest.main()