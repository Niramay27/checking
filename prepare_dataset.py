import json
import uuid
import os
import logging
from pathlib import Path

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s -- %(message)s")
logger = logging.getLogger("prepare_custom_dataset")

def process_file(file_path: str, source_lang: str, target_lang: str, sampling_rate: int) -> list:
    samples = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_sample = json.loads(line)
                    sample_id = str(uuid.uuid4())
                    manifest_sample = {
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
                            "lang": target_lang,
                        }
                    }
                    samples.append(manifest_sample)
                except Exception as e:
                    logger.error("Error processing line in file %s: %s", file_path, e)
    except Exception as e:
        logger.error("Error reading file %s: %s", file_path, e)
    return samples

def prepare_manifest(input_file: str, output_manifest: str, source_lang: str, target_lang: str, sampling_rate: int) -> None:
    if os.path.exists(input_file):
        logger.info(f"Found input file: {input_file}")
        try:
            samples = process_file(input_file, source_lang, target_lang, sampling_rate)
            with open(output_manifest, "w") as fp_out:
                for sample in samples:
                    fp_out.write(json.dumps(sample) + "\n")
            logger.info(f"Saved {len(samples)} samples to manifest: {output_manifest}")
        except Exception as e:
            logger.error(f"Error processing input file: {e}")
    else:
        logger.error(f"Input file {input_file} not found!")

def main() -> None:
    # Default values
    input_file = "./jsonl files/train.jsonl"
    output_manifest = "./jsonl files/train_manifest.jsonl"
    source_lang = "eng"
    target_lang = "hin"
    sampling_rate = 16000

    prepare_manifest(
        input_file=input_file,
        output_manifest=output_manifest,
        source_lang=source_lang,
        target_lang=target_lang,
        sampling_rate=sampling_rate
    )

if __name__ == "__main__":
    main()