import json
import random
import os
import argparse
from tqdm import tqdm
import torchaudio
import torch

def add_gaussian_noise(waveform, noise_std):
    noise = torch.randn_like(waveform) * noise_std
    return torch.clamp(waveform + noise, -1.0, 1.0)

def add_noise_with_target_snr(waveform, target_snr_db):
    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return torch.clamp(waveform + noise, -1.0, 1.0)

def augment_train_200hr(input_jsonl, output_jsonl, output_audio_dir, noise_var, noise_ratio=0.2, seed=42):
    noise_std = noise_var ** 0.5
    os.makedirs(output_audio_dir, exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.seed(seed)
    noisy_indices = set(random.sample(range(len(data)), int(noise_ratio * len(data))))

    updated_data = []
    for idx, item in enumerate(data):
        audio_path = item["source"]["audio_local_path"]
        sampling_rate = item["source"]["sampling_rate"]

        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != sampling_rate:
                print(f"⚠ Sampling rate mismatch: {audio_path}")

            if idx in noisy_indices:
                noisy_waveform = add_gaussian_noise(waveform, noise_std)
                filename = os.path.basename(audio_path)
                new_path = os.path.join(output_audio_dir, f"noisy_{filename}")
                torchaudio.save(new_path, noisy_waveform, sr)
                item["source"]["audio_local_path"] = new_path

        except Exception as e:
            print(f"⚠ Failed: {audio_path}: {e}")

        updated_data.append(item)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in updated_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ 200hr train set: {len(data)} samples processed with {noise_var} variance noise on {noise_ratio*100}%.")

def augment_2hr_gauss(input_jsonl, noise_vars, seed=42):
    with open(input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for noise_var in noise_vars:
        noise_std = noise_var ** 0.5
        suffix = f"{noise_var}_audio"
        output_dir = f"./{suffix}"
        output_jsonl = f"./jsonl files/{suffix}.jsonl"
        os.makedirs(output_dir, exist_ok=True)

        updated_data = []
        for item in data:
            audio_path = item["source"]["audio_local_path"]
            sampling_rate = item["source"]["sampling_rate"]

            try:
                waveform, sr = torchaudio.load(audio_path)
                noisy_waveform = add_gaussian_noise(waveform, noise_std)
                filename = os.path.basename(audio_path)
                new_path = os.path.join(output_dir, f"noisy_{filename}")
                torchaudio.save(new_path, noisy_waveform, sr)

                item = item.copy()
                item["source"]["audio_local_path"] = new_path

            except Exception as e:
                print(f"⚠ Failed: {audio_path}: {e}")

            updated_data.append(item)

        with open(output_jsonl, "w", encoding="utf-8") as f:
            for item in updated_data:
                f.write(json.dumps(item) + "\n")

        print(f"✅ 2hr GAUSS: {noise_var} variance noise applied to {len(data)} files.")

def augment_2hr_snr(input_jsonl, snr_values):
    with open(input_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for snr in snr_values:
        output_dir = f"./noisy_audio_{snr}dB"
        output_jsonl = f"./jsonl files/snr{snr}_noisy_output.jsonl"
        os.makedirs(output_dir, exist_ok=True)

        updated_entries = []
        for line in tqdm(lines, desc=f"SNR {snr}dB"):
            entry = json.loads(line)
            audio_path = entry["source"]["audio_local_path"]

            try:
                waveform, sr = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                noisy_waveform = add_noise_with_target_snr(waveform, snr)
                filename = os.path.basename(audio_path)
                new_path = os.path.join(output_dir, f"noisy{snr}_{filename}")
                torchaudio.save(new_path, noisy_waveform, sr)

                entry = entry.copy()
                entry["source"]["audio_local_path"] = new_path
                entry["source"]["added_noise_snr_db"] = snr

            except Exception as e:
                print(f"⚠ Failed: {audio_path} — {e}")
                entry["source"]["audio_local_path"] = None

            updated_entries.append(entry)

        with open(output_jsonl, "w", encoding="utf-8") as f:
            for entry in updated_entries:
                f.write(json.dumps(entry) + "\n")

        print(f"✅ 2hr SNR: {snr}dB noise applied to {len(updated_entries)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Augmentation Script")
    parser.add_argument("--gauss_var", type=float, default=0.01, help="Variance for Gaussian noise in 200hr train set (default=0.01)")
    args = parser.parse_args()

    # Paths
    train_200_path = "./jsonl files/200_hours.jsonl"
    output_200_jsonl = "./jsonl files/noisy_ds.jsonl"
    output_200_audio = "./noisy_audio"

    two_hr_path = "./jsonl files/2_hours.jsonl"

    # Augmentations
    augment_train_200hr(train_200_path, output_200_jsonl, output_200_audio, args.gauss_var)
    augment_2hr_gauss(two_hr_path, noise_vars=[0.001, 0.005, 0.01, 0.05])
    augment_2hr_snr(two_hr_path, snr_values=[10, 15, 25])