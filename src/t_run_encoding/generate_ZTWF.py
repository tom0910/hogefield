import torch
import torchaudio
import os
import math
import random

def generate_wave(sample_rate, duration, frequency, wave_type="sine", amplitude=0.5):
    """Generate a waveform (sine or square)."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    if wave_type == "sine":
        return amplitude * torch.sin(2 * math.pi * frequency * t)
    elif wave_type == "square":
        return amplitude * torch.sign(torch.sin(2 * math.pi * frequency * t))
    else:
        raise ValueError("Unsupported wave type. Choose 'sine' or 'square'.")

def major_scale(root_freq):
    """Generate a major scale from a root frequency."""
    return [root_freq * (2 ** (i / 12)) for i in [0, 2, 4, 5, 7, 9, 11, 12]]  # Major scale intervals

def minor_scale(root_freq):
    """Generate a minor scale from a root frequency."""
    return [root_freq * (2 ** (i / 12)) for i in [0, 2, 3, 5, 7, 8, 10, 12]]  # Minor scale intervals

def chromatic_scale(root_freq):
    """Generate a chromatic scale from a root frequency."""
    return [root_freq * (2 ** (i / 12)) for i in range(13)]  # All semitones in one octave

def generate_scale_wave(sample_rate, note_freqs, duration, amplitude=0.5):
    """Generate a wave for a scale, fitting exactly into the given duration."""
    note_duration = duration / len(note_freqs)
    waves = [generate_wave(sample_rate, note_duration, freq, "sine", amplitude) for freq in note_freqs]
    return torch.cat(waves)

def save_wave(output_dir, label, unique_id, variant_id, waveform, sample_rate):
    """Save a waveform as a .wav file with GSC-like naming conventions."""
    os.makedirs(f"{output_dir}/{label}", exist_ok=True)

    # Naming convention: <unique_id>_nohash_<variant_id>.wav
    # - unique_id: A unique identifier for the file (e.g., an index or hash).
    # - nohash: Placeholder for future variants or metadata (kept for consistency).
    # - variant_id: Indicates a specific variant or repetition of the file.
    file_name = f"{unique_id}_nohash_{variant_id}.wav"
    file_path = f"{output_dir}/{label}/{file_name}"

    torchaudio.save(file_path, waveform.unsqueeze(0), sample_rate)
    print(f"Saved: {file_path}")

def generate_signals(output_dir, sample_rate=16000, duration=1.0, frequencies=None):
    """Generate and save sine, square, and scale wave signals."""
    if frequencies is None:
        frequencies = torch.logspace(math.log10(20), math.log10(20000), steps=256).tolist()

    # Generate and save sine and square waves
    for i, freq in enumerate(frequencies):
        unique_id = str(i).zfill(8)  # Unique identifier for each file
        variant_id = 0

        # Generate and save sine wave
        sine_wave = generate_wave(sample_rate, duration, freq, wave_type="sine")
        save_wave(output_dir, "sine", unique_id, variant_id, sine_wave, sample_rate)

        # Generate and save square wave
        square_wave = generate_wave(sample_rate, duration, freq, wave_type="square")
        save_wave(output_dir, "square", unique_id, variant_id, square_wave, sample_rate)

    # Generate and save scale waveforms
    scale_types = [major_scale, minor_scale, chromatic_scale]
    for i in range(256):  # Create 256 scale examples
        unique_id = str(i + len(frequencies)).zfill(8)  # Offset ID for scales
        variant_id = 0

        # Randomize scale type and root note
        root_freq = random.uniform(27, 4186)  # Random root frequency (A0 to C8)
        scale_type = random.choice(scale_types)
        scale_notes = scale_type(root_freq)

        # Generate scale waveform
        scale_wave = generate_scale_wave(sample_rate, scale_notes, duration)
        save_wave(output_dir, "piano_scale", unique_id, variant_id, scale_wave, sample_rate)

if __name__ == "__main__":
    output_dir = "/project/data/ZTWF" 
    generate_signals(output_dir)

# import torch
# import torchaudio
# import os
# import math

# def generate_wave(sample_rate, duration, frequency, wave_type="sine", amplitude=0.5):
#     """Generate a waveform (sine or square)."""
#     t = torch.linspace(0, duration, int(sample_rate * duration))
#     if wave_type == "sine":
#         return amplitude * torch.sin(2 * math.pi * frequency * t)
#     elif wave_type == "square":
#         return amplitude * torch.sign(torch.sin(2 * math.pi * frequency * t))
#     else:
#         raise ValueError("Unsupported wave type. Choose 'sine' or 'square'.")

# def generate_piano_wave(sample_rate, duration, f_start, f_end, amplitude=0.5):
#     """Generate a piano-like chirp sound going up and down in pitch."""
#     t = torch.linspace(0, duration, int(sample_rate * duration))
#     up_chirp = amplitude * torch.sin(2 * math.pi * ((f_start * (1 - t / duration)) + (f_end * t / duration)) * t)
#     down_chirp = amplitude * torch.sin(2 * math.pi * ((f_end * (1 - t / duration)) + (f_start * t / duration)) * t)
#     return torch.cat((up_chirp, down_chirp))

# def save_wave(output_dir, label, unique_id, variant_id, waveform, sample_rate):
#     """Save a waveform as a .wav file with GSC-like naming conventions."""
#     os.makedirs(f"{output_dir}/{label}", exist_ok=True)

#     # Naming convention: <unique_id>_nohash_<variant_id>.wav
#     # - unique_id: A unique identifier for the file (e.g., an index or hash).
#     # - nohash: Placeholder for future variants or metadata (kept for consistency).
#     # - variant_id: Indicates a specific variant or repetition of the file.
#     file_name = f"{unique_id}_nohash_{variant_id}.wav"
#     file_path = f"{output_dir}/{label}/{file_name}"

#     torchaudio.save(file_path, waveform.unsqueeze(0), sample_rate)
#     print(f"Saved: {file_path}")

# def generate_signals(output_dir, sample_rate=16000, duration=1.0, frequencies=None):
#     """Generate and save sine, square, and piano wave signals."""
#     if frequencies is None:
#         frequencies = torch.logspace(math.log10(20), math.log10(20000), steps=256).tolist()

#     for i, freq in enumerate(frequencies):
#         unique_id = str(i).zfill(8)  # Generate a unique identifier (zero-padded index)
#         variant_id = 0  # Variant ID starts at 0 for each unique_id

#         # Generate and save sine wave
#         sine_wave = generate_wave(sample_rate, duration, freq, wave_type="sine")
#         save_wave(output_dir, "sine", unique_id, variant_id, sine_wave, sample_rate)

#         # Generate and save square wave
#         square_wave = generate_wave(sample_rate, duration, freq, wave_type="square")
#         save_wave(output_dir, "square", unique_id, variant_id, square_wave, sample_rate)

#     # Generate and save piano-like wave
#     for i in range(256):  # Create 256 piano examples
#         unique_id = str(i + len(frequencies)).zfill(8)  # Offset ID for piano class
#         variant_id = 0
#         f_start, f_end = 220, 880  # Example range: A3 to A5
#         piano_wave = generate_piano_wave(sample_rate, duration, f_start, f_end)
#         save_wave(output_dir, "piano", unique_id, variant_id, piano_wave, sample_rate)

# if __name__ == "__main__":
#     output_dir = "/project/data/ZTWF"  # Change to your desired output path
#     generate_signals(output_dir)
