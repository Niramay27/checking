import json
import os
import torch
import torchaudio
from tqdm import tqdm

# === Config ===
input_jsonl = "./jsonl files/2_hours.jsonl"
snr_values = [10, 15, 25]

def add_noise_with_target_snr(waveform, target_snr_db):
    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return waveform + noise

# Load data once
with open(input_jsonl, "r", encoding="utf-8") as f:
    lines = f.readlines()

for snr in snr_values:
    output_jsonl = f"./jsonl files/snr{snr}_noisy_output.jsonl"
    output_audio_dir = f"./noisy_audio_{snr}dB"
    os.makedirs(output_audio_dir, exist_ok=True)
    
    print(f"\n--- Processing for SNR={snr}dB ---")
    updated_entries = []

    for line in tqdm(lines, desc=f"SNR {snr}dB"):
        entry = json.loads(line)
        audio_path = entry["source"]["audio_local_path"]

        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

            noisy_waveform = add_noise_with_target_snr(waveform, snr)
            noisy_waveform = torch.clamp(noisy_waveform, -1.0, 1.0)

            filename = os.path.basename(audio_path)
            new_path = os.path.join(output_audio_dir, f"noisy{snr}_{filename}")
            torchaudio.save(new_path, noisy_waveform, sr)

            # Update JSON entry
            entry = entry.copy()
            entry["source"]["audio_local_path"] = new_path
            entry["source"]["added_noise_snr_db"] = snr

        except Exception as e:
            print(f"⚠️ Failed: {audio_path} — {e}")
            entry["source"]["audio_local_path"] = None

        updated_entries.append(entry)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Done for SNR={snr}dB. Saved to folder: {output_audio_dir}")
    print(f"✅ JSONL saved to: {output_jsonl}")
