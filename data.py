# code tocreate a JSONL file from audio transcriptions and translations
import os
import json
import yaml
from pydub import AudioSegment
from typing import List, Dict

os.mkdir('jsonl files')

# Function to split audio based on offset and duration
def split_audio(input_audio_path, offset, duration, output_dir, filename_prefix):
    """
    Splits the input audio file at a given offset and duration.
    """
    # Load the original audio file
    audio = AudioSegment.from_wav(input_audio_path)
    
    # Convert offset and duration from seconds to milliseconds
    start_ms = int(offset * 1000)
    end_ms = int((offset + duration) * 1000)
    
    # Slice the audio
    sliced_audio = audio[start_ms:end_ms]
    
    # Define the output file path
    output_filename = f"{filename_prefix}_{start_ms}_{end_ms}.wav"
    output_file_path = os.path.join(output_dir, output_filename)
    
    # Export the sliced audio
    sliced_audio.export(output_file_path, format="wav")
    
    return output_file_path

# Function to read dev.yaml and extract audio file paths
def read_audio_paths_from_yaml(yaml_file_path: str) -> Dict[str, str]:
    """
    Reads the dev.yaml file and returns a dictionary with audio paths.
    """
    with open(yaml_file_path, 'r', encoding='utf-8') as yaml_file:
        dev_data = yaml.safe_load(yaml_file)
        
    # Extract the 'wav' paths for audio files
    audio_paths = {item['speaker_id']: item['wav'] for item in dev_data}
    
    return audio_paths

# Function to process the transcription and translation files
def process_transcriptions_and_create_jsonl(offset_duration_file: str, en_file: str, hi_file: str, 
                                             yaml_file_path: str, output_dir: str, jsonl_output_file: str):
    """
    Processes the transcription and translation files, splits the audio, and creates a JSONL file.
    """
    # Read the offset-duration data (from YAML format)
    with open(offset_duration_file, 'r') as f:
        offset_duration_data = yaml.safe_load(f)  # Read YAML file directly
    
    # Read the sentence-level transcriptions and translations
    with open(en_file, 'r', encoding='utf-8') as f:
        english_sentences = f.readlines()
    
    with open(hi_file, 'r', encoding='utf-8') as f:
        hindi_translations = f.readlines()
    
    # Read audio paths from dev.yaml
    audio_paths = read_audio_paths_from_yaml(yaml_file_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the JSONL output file
    with open(jsonl_output_file, 'w', encoding='utf-8') as jsonl_file:
        # Iterate over the offset-duration data and corresponding sentences
        for i, entry in enumerate(offset_duration_data):
            duration = entry['duration']
            offset = entry['offset']
            speaker_id = entry['speaker_id']
            wav_file = entry['wav']
            
            # Extract the corresponding sentence and translation
            english_sentence = english_sentences[i].strip()
            hindi_translation = hindi_translations[i].strip()
            
            # Get the audio file path from the YAML data
            audio_file_path = audio_paths.get(speaker_id, None)
            
            if audio_file_path:
                # Create full path to the audio file based on the directory
                full_audio_file_path = os.path.join("./ds/train/wav/", audio_file_path)
                
                # Split the audio for this entry
                split_audio_path = split_audio(full_audio_file_path, offset, duration, output_dir, f"{speaker_id}_{i}")
                
                # Create the entry for the JSONL file
                jsonl_entry = {
                    "sentence": english_sentence,
                    "audio": {
                        "path": split_audio_path
                    },
                    "translation": hindi_translation
                }
                
                # Write the entry to the JSONL file
                jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
                
                print(f"Processed sentence {i+1}/{len(offset_duration_data)}: {english_sentence}")
            else:
                print(f"Audio file for speaker {speaker_id} not found.")
    
    print(f"Finished processing. JSONL file saved to {jsonl_output_file}")

# Example usage
if __name__ == "__main__":
    # Paths to input files
    offset_duration_file = './ds/train/txt/train.yaml'
    en_file = './ds/train/txt/train.en'
    hi_file = './ds/train/txt/train.hi'
    yaml_file_path = './ds/train/txt/train.yaml'  # Path to the dev.yaml file containing wav paths
    
    # Output directory for split audio and JSONL file
    output_dir = './chunked_audio'
    jsonl_output_file = './jsonl files/train.jsonl'
    
    # Process the data and generate the JSONL file
    process_transcriptions_and_create_jsonl(offset_duration_file, en_file, hi_file, yaml_file_path, output_dir, jsonl_output_file)