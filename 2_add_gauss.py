import json
import random
import os
import torchaudio
import torch

# Config
input_jsonl = "./jsonl files/2_hours.jsonl"
noise_levels = [0.001, 0.005, 0.01, 0.05]
noise_mean = 0
noise_ratio = 1  # 100% of files get noise
seed = 42

# Load dataset once
with open(input_jsonl, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

for noise_var in noise_levels:
    noise_std = noise_var ** 0.5
    suffix = f"{noise_var}_audio"
    output_jsonl = f"./jsonl files/{suffix}.jsonl"
    output_audio_dir = f"./{suffix}"
    os.makedirs(output_audio_dir, exist_ok=True)

    print(f"\n--- Injecting noise (var={noise_var}) ---")
    print(f"Saving to: {output_audio_dir}, {output_jsonl}")

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
                noisy_waveform = torch.clamp(noisy_waveform, -1.0, 1.0)

                # Save noisy audio
                filename = os.path.basename(audio_path)
                new_path = os.path.join(output_audio_dir, f"noisy_{filename}")
                torchaudio.save(new_path, noisy_waveform, sr)

                # Update path in JSON
                item = item.copy()  # Important to not overwrite original
                item["source"]["audio_local_path"] = new_path

        except Exception as e:
            print(f"Warning: Failed to process {audio_path}: {e}")

        updated_data.append(item)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in updated_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Done: {len(data)} entries processed. Noisy audio saved to {output_audio_dir}.")