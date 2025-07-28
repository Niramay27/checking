# code to sample audio files 
import json
import random
import torchaudio
import os

input_path = "./jsonl files/train_manifest.jsonl"  # Replace with your path
output_200h = "./jsonl files/200_hours.jsonl"
output_2h = "./jsonl files/2_hours.jsonl"
seed = 42

target_200h = 200 * 3600  # seconds
target_2h = 2 * 3600      # seconds

# Load data
with open(input_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.seed(seed)
random.shuffle(data)

# Helper to get duration
def get_duration(item):
    audio_path = item["source"]["audio_local_path"]
    sampling_rate = item["source"]["sampling_rate"]

    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        return 0

    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != sampling_rate:
            print(f"Warning: Sampling rate mismatch for {audio_path} (expected {sampling_rate}, got {sr})")
        num_samples = waveform.shape[1]
        return num_samples / sr
    except Exception as e:
        print(f"Warning: Failed to load {audio_path}: {e}")
        return 0

# Collect 200h
selected_200h = []
total_duration_200h = 0

for item in data:
    dur = get_duration(item)
    if dur <= 0:
        continue

    selected_200h.append(item)
    total_duration_200h += dur

    if total_duration_200h >= target_200h:
        break

# Remove selected 200h
remaining_data = [item for item in data if item not in selected_200h]

# Collect 2h
selected_2h = []
total_duration_2h = 0

for item in remaining_data:
    dur = get_duration(item)
    if dur <= 0:
        continue

    selected_2h.append(item)
    total_duration_2h += dur

    if total_duration_2h >= target_2h:
        break

# Save files
with open(output_200h, "w", encoding="utf-8") as f:
    for item in selected_200h:
        f.write(json.dumps(item) + "\n")

with open(output_2h, "w", encoding="utf-8") as f:
    for item in selected_2h:
        f.write(json.dumps(item) + "\n")

print(f"Saved {len(selected_200h)} samples to {output_200h} ({total_duration_200h/3600:.2f} hours)")
print(f"Saved {len(selected_2h)} samples to {output_2h} ({total_duration_2h/3600:.2f} hours)")