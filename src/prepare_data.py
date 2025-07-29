import os
import json
import uuid
import yaml
import random
import logging
from pathlib import Path
from typing import List, Dict
from pydub import AudioSegment
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(message)s")
logger = logging.getLogger("prepare_dataset")

# -----------------------------------------------------------------------------------
# SECTION 1: JSONL Generation from YAML and Text Files
# -----------------------------------------------------------------------------------

def split_audio(input_audio_path, offset, duration, output_dir, filename_prefix):
    audio = AudioSegment.from_wav(input_audio_path)
    start_ms = int(offset * 1000)
    end_ms = int((offset + duration) * 1000)
    sliced_audio = audio[start_ms:end_ms]
    output_filename = f"{filename_prefix}_{start_ms}_{end_ms}.wav"
    output_file_path = os.path.join(output_dir, output_filename)
    sliced_audio.export(output_file_path, format="wav")
    return output_file_path

def read_audio_paths_from_yaml(yaml_file_path: str) -> Dict[str, str]:
    with open(yaml_file_path, 'r', encoding='utf-8') as yaml_file:
        dev_data = yaml.safe_load(yaml_file)
    return {item['speaker_id']: item['wav'] for item in dev_data}

def create_jsonl_from_text_and_yaml(offset_duration_file: str, en_file: str, hi_file: str,
                                    yaml_file_path: str, output_dir: str, jsonl_output_file: str):
    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    eng_file = os.path.join(BASE_DIR, en_file)
    hin_file = os.path.join(BASE_DIR, hi_file)
    os.makedirs(os.path.dirname(jsonl_output_file), exist_ok=True)

    with open(offset_duration_file, 'r') as f:
        offset_duration_data = yaml.safe_load(f)
    with open(eng_file, 'r', encoding='utf-8') as f:
        english_sentences = f.readlines()
    with open(hin_file, 'r', encoding='utf-8') as f:
        hindi_translations = f.readlines()

    audio_paths = read_audio_paths_from_yaml(yaml_file_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(jsonl_output_file, 'w', encoding='utf-8') as jsonl_file:
        for i, entry in enumerate(offset_duration_data):
            duration = entry['duration']
            offset = entry['offset']
            speaker_id = entry['speaker_id']
            english_sentence = english_sentences[i].strip()
            hindi_translation = hindi_translations[i].strip()
            audio_file_path = audio_paths.get(speaker_id)

            if audio_file_path:
                full_path = os.path.join(BASE_DIR , "ds","train","wav", audio_file_path)
                split_audio_path = split_audio(full_path, offset, duration, output_dir, f"{speaker_id}_{i}")
                jsonl_entry = {
                    "sentence": english_sentence,
                    "audio": {"path": split_audio_path},
                    "translation": hindi_translation
                }
                jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
            else:
                logger.warning(f"Audio path for speaker {speaker_id} not found.")

    logger.info(f"Saved JSONL to {jsonl_output_file}")

    cleaned_entries = []
    with open(jsonl_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry["sentence"].strip() and entry["translation"].strip():
                    cleaned_entries.append(entry)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line during JSONL cleanup")

    with open(jsonl_output_file, 'w', encoding='utf-8') as f:
        for entry in cleaned_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Removed empty entries. Final JSONL has {len(cleaned_entries)} valid samples.")

# -----------------------------------------------------------------------------------
# SECTION 2: Convert JSONL to Manifest Format
# -----------------------------------------------------------------------------------

def process_file(file_path: str, source_lang: str, target_lang: str, sampling_rate: int) -> list:
    samples = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                raw_sample = json.loads(line.strip())
                sample_id = str(uuid.uuid4())
                samples.append({
                    "source": {
                        "id": sample_id,
                        "text": raw_sample.get("sentence", ""),
                        "lang": source_lang,
                        "audio_local_path": raw_sample["audio"]["path"],
                        "sampling_rate": sampling_rate
                    },
                    "target": {
                        "id": sample_id,
                        "text": raw_sample["translation"],
                        "lang": target_lang
                    }
                })
            except Exception as e:
                logger.error("Skipping line due to error: %s", e)
    return samples

def prepare_manifest(input_file: str, output_manifest: str, source_lang: str, target_lang: str, sampling_rate: int):
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found!")
        return
    samples = process_file(input_file, source_lang, target_lang, sampling_rate)
    with open(output_manifest, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    logger.info(f"Saved {len(samples)} samples to manifest at {output_manifest}")

# -----------------------------------------------------------------------------------
# SECTION 3: Sampling Audio from Manifest
# -----------------------------------------------------------------------------------

def get_duration(item):
    audio_path = item["source"]["audio_local_path"]
    sampling_rate = item["source"]["sampling_rate"]
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != sampling_rate:
            logger.warning(f"Sampling rate mismatch for {audio_path} (expected {sampling_rate}, got {sr})")
        return waveform.shape[1] / sr
    except Exception as e:
        logger.warning(f"Failed to load {audio_path}: {e}")
        return 0

def sample_duration(input_manifest: str, output_200h: str, output_2h: str, seed=42):
    with open(input_manifest, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.seed(seed)
    random.shuffle(data)

    def collect_samples(target_duration):
        total, samples = 0, []
        for item in data:
            dur = get_duration(item)
            if dur <= 0:
                continue
            total += dur
            samples.append(item)
            if total >= target_duration:
                break
        return samples, total

    selected_200h, dur_200h = collect_samples(200 * 3600)
    remaining_data = [item for item in data if item not in selected_200h]
    selected_2h, dur_2h = collect_samples(2 * 3600)

    with open(output_200h, "w", encoding="utf-8") as f:
        for item in selected_200h:
            f.write(json.dumps(item) + "\n")
    with open(output_2h, "w", encoding="utf-8") as f:
        for item in selected_2h:
            f.write(json.dumps(item) + "\n")

    logger.info(f"Saved {len(selected_200h)} samples to {output_200h} ({dur_200h/3600:.2f} hours)")
    logger.info(f"Saved {len(selected_2h)} samples to {output_2h} ({dur_2h/3600:.2f} hours)")


# -----------------------------------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    # STEP 1: Create JSONL
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    create_jsonl_from_text_and_yaml(
        offset_duration_file = os.path.join(BASE_DIR, "ds","train","txt","train.yaml"),
        en_file= os.path.join(BASE_DIR,"ds","train","txt","train.en"),
        hi_file= os.path.join(BASE_DIR,"ds","train","txt","train.hi"),
        yaml_file_path= os.path.join(BASE_DIR,"ds","train","txt","train.yaml"),
        output_dir= os.path.join(BASE_DIR,'chunked_audio'),
        jsonl_output_file=os.path.join(BASE_DIR, "jsonl files", "train.jsonl")
    )

    # STEP 2: Create Manifest
    prepare_manifest(
        input_file="./jsonl files/train.jsonl",
        output_manifest="./jsonl files/train_manifest.jsonl",
        source_lang="eng",
        target_lang="hin",
        sampling_rate=16000
    )

    # STEP 3: Sample Subsets
    sample_duration(
        input_manifest="./jsonl files/train_manifest.jsonl",
        output_200h="./jsonl files/200_hours.jsonl",
        output_2h="./jsonl files/2_hours.jsonl",
        seed=42
    )
