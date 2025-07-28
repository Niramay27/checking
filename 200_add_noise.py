import json
import random
import os
import torchaudio
import torch

input_jsonl = "./jsonl files/200_hours.jsonl"   # Replace with your JSONL file
output_jsonl = "./jsonl files/noisy_ds.jsonl"
output_audio_dir = "./noisy_audio"  # Where noisy files will be saved

noise_mean = 0
noise_var = 0.01
noise_std = noise_var ** 0.5
noise_ratio = 0.2  # 20% of dataset

seed = 42
os.makedirs(output_audio_dir, exist_ok=True)

# Load dataset
with open(input_jsonl, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Randomly select 20% for noise injection
random.seed(seed)
indices_with_noise = set(random.sample(range(len(data)), int(noise_ratio * len(data))))

updated_data = []

for idx, item in enumerate(data):
    audio_path = item["source"]["audio_local_path"]
    sampling_rate = item["source"]["sampling_rate"]

    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != sampling_rate:
            print(f"Warning: Sampling rate mismatch for {audio_path} (expected {sampling_rate}, got {sr})")

        if idx in indices_with_noise:
            # Add Gaussian noise
            noise = torch.randn_like(waveform) * noise_std
            noisy_waveform = waveform + noise
            noisy_waveform = torch.clamp(noisy_waveform, -1.0, 1.0)  # Clamp to valid audio range

            # Save new noisy audio file
            filename = os.path.basename(audio_path)
            new_path = os.path.join(output_audio_dir, f"noisy_{filename}")
            torchaudio.save(new_path, noisy_waveform, sr)

            # Update JSON entry
            item["source"]["audio_local_path"] = new_path

    except Exception as e:
        print(f"Warning: Failed to process {audio_path}: {e}")

    updated_data.append(item)

# Save updated JSONL
with open(output_jsonl, "w", encoding="utf-8") as f:
    for item in updated_data:
        f.write(json.dumps(item) + "\n")

print(f"Processed {len(data)} entries.")
print(f"Noisy audio files saved to {output_audio_dir}")
print(f"Updated JSONL saved to {output_jsonl}")
