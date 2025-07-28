import os
import torch
import json
import numpy as np
import torchaudio
from datasets import Dataset, Audio
from transformers import (
    SeamlessM4TTokenizer,
    SeamlessM4TProcessor,
    SeamlessM4TModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import evaluate
import logging
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from tqdm import tqdm
# Configure logging to output to both console and a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results.txt', mode='w')
    ]
)

logger = logging.getLogger(__name__)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)


# Constants - Optimized for 350 hours dataset
MODEL_ID = "facebook/hf-seamless-m4t-medium"
TRAIN_DATA_PATH = './jsonl files/200_hours.jsonl'  # Path to training data
EVAL_DATA_PATH = './jsonl files/2_hours.jsonl'  # Path to evaluation data
OUTPUT_DIR = "./seamless_m4t_finetuned"
BATCH_SIZE = 8  # Adjusted based on your previous results
LEARNING_RATE = 3e-5  # Slightly lower initial learning rate
NUM_EPOCHS = 10  # Increased epochs with better early stopping
GRADIENT_ACCUMULATION_STEPS = 4  
MAX_OUTPUT_LENGTH = 1024
SRC_LANG = "eng"
TGT_LANG = "hin"
SAMPLING_RATE = 16000
WARMUP_RATIO = 0.1  # Using ratio instead of fixed steps (10% of training will be warmup)
WEIGHT_DECAY = 0.03  # Slightly increased weight decay
MAX_GRAD_NORM = 0.5  # Reduced to control high gradient norms seen in previous run

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
# Not running the function is the first trial. We can check audio files with noise in second experiment 

def augment_audio(audio_array):
    """Apply random audio augmentations to prevent overfitting."""
    # Skip augmentation ~20% of the time
    if np.random.random() < 0.2:
        return audio_array
    
    # Convert to torch tensor for torchaudio operations
    audio_tensor = torch.tensor(audio_array).float().unsqueeze(0)
    
    # Apply one of several possible augmentations
    aug_type = np.random.choice(['speed', 'volume', 'pitch', 'none'], p=[0.4, 0.3, 0.2, 0.1])
    
    if aug_type == 'speed':
        # Speed perturbation
        speed_factor = np.random.uniform(0.85, 1.15)
        effects = [["tempo", f"{speed_factor}"]]
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio_tensor, SAMPLING_RATE, effects)
            return augmented.squeeze(0).numpy()
        except Exception:
            return audio_array
            
    elif aug_type == 'volume':
        # Volume perturbation
        volume_factor = np.random.uniform(0.75, 1.25)
        return (audio_array * volume_factor).astype(np.float32)
    
    elif aug_type == 'pitch':
        # Pitch shift
        try:
            pitch_shift = np.random.uniform(-300, 300)  # Shift by up to 300 cents (3 semitones)
            effects = [["pitch", f"{pitch_shift}"], ["rate", str(SAMPLING_RATE)]]
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio_tensor, SAMPLING_RATE, effects)
            return augmented.squeeze(0).numpy()
        except Exception:
            return audio_array
    
    return audio_array
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text batches with augmentation."""
    processor: Any
    apply_augmentation: bool = False
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio and text features
        audio_arrays = [feature["audio"]["array"] for feature in features]
        
        # Apply augmentation during training
        if self.apply_augmentation:
            audio_arrays = [augment_audio(array) for array in audio_arrays]
            
        labels = [feature["hindi_text"] for feature in features]
        
        # Process inputs - passing audio arrays directly
        batch = self.processor(
            audios=audio_arrays,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
            src_lang=SRC_LANG
        )
        
        # Process labels
        labels_batch = self.processor(
            text=labels,
            return_tensors="pt",
            padding=True,
            tgt_lang=TGT_LANG
        )
        
        # Replace padding token id with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        return batch

# Load dataset from a JSONL file
def load_dataset_from_file(jsonl_file):
    """Load the dataset from a JSONL file without splitting."""
    logger.info(f"Loading dataset from {jsonl_file}")
    
    # Check if file exists
    if not os.path.exists(jsonl_file):
        logger.error(f"File not found: {jsonl_file}")
        raise FileNotFoundError(f"Could not find {jsonl_file}")
    
    # Load JSONL data
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        # data = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(data)} examples from {jsonl_file}")
    
    # Convert to format expected by datasets library
    dataset_dict = {
        "audio": [item["source"]["audio_local_path"] for item in data],
        "hindi_text": [item["target"]["text"] for item in data],
        "english_text": [item["source"]["text"] for item in data],
        "id": [item["source"]["id"] for item in data]
    }
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Load audio files
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    return dataset

# Load all datasets
from datasets import DatasetDict

def load_datasets():
    train_dataset = load_dataset_from_file(TRAIN_DATA_PATH)
    
    # Load evaluation dataset
    eval_dataset = load_dataset_from_file(EVAL_DATA_PATH)
    
    # Split train dataset into train/test (90/10 split)
    logger.info("Splitting training dataset into train/test sets (90/10 split)")
    train_test_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    
    logger.info(f"Train set size: {len(train_test_split['train'])}")
    logger.info(f"Validation set size: {len(eval_dataset)}")
    logger.info(f"Test set size: {len(train_test_split['test'])}")
    
    return train_test_split['train'], eval_dataset, train_test_split['test']

# Set up decoder parameters for the model
def setup_decoder_params(model, tokenizer):
    """Set up decoder parameters for the model."""
    # Identify the target language token ID
    target_lang_token = f"__{TGT_LANG}__"
    target_lang_token_id = tokenizer.convert_tokens_to_ids(target_lang_token)
    
    # Create forced decoder ids
    forced_decoder_ids = [[0, target_lang_token_id]]
    
    # Configure model with dropout for regularization
    model.config.attention_dropout = 0.15
    model.config.hidden_dropout_prob = 0.15
    
    return model, forced_decoder_ids
# Custom training class
class CustomSeq2SeqTrainer(Trainer):
    def __init__(self, forced_decoder_ids=None, processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_decoder_ids = forced_decoder_ids
        self.processor = processor
        self.best_metrics = {"eval_bleu": 0, "eval_wer": float('inf')}
        self.metrics_history = {"eval_bleu": [], "eval_wer": [], "eval_loss": []}
        self.files_sent = 0

    def evaluate(self, *args, **kwargs):
        # Run regular evaluation
        metrics = super().evaluate(*args, **kwargs)
        
        # Add BLEU and WER score calculation
        eval_dataloader = self.get_eval_dataloader()
        all_preds = []
        all_labels = []
        
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Computing metrics"):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            
            with torch.no_grad():
                # Generate predictions
                generated_tokens = self.model.generate(
                    input_features=batch["input_features"],
                    tgt_lang=TGT_LANG,
                    max_new_tokens=MAX_OUTPUT_LENGTH,
                    num_beams=4,  # Using beam search for better quality
                    decoder_input_ids=self.forced_decoder_ids,
                    generate_speech=False,
                    return_dict_in_generate=False
                )
                
                # Get labels
                labels = batch["labels"]

            # Add predictions and labels to our lists
            all_preds.extend(generated_tokens.sequences.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Replace -100 in labels with pad token id
        all_labels_processed = [
            np.where(seq != -100, seq, self.processor.tokenizer.pad_token_id).astype(np.int64) 
            for seq in all_labels
        ]

        # Decode using processor
        decoded_preds = self.processor.batch_decode(all_preds, skip_special_tokens=True)
        decoded_labels = self.processor.batch_decode(all_labels_processed, skip_special_tokens=True)

        # Compute BLEU score
        bleu_metric = evaluate.load("sacrebleu")
        bleu_result = bleu_metric.compute(
            predictions=decoded_preds, 
            references=[[label] for label in decoded_labels]
        )
        
        # Compute WER
        wer_metric = evaluate.load("wer")
        wer_result = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # Add metrics to results
        metrics.update({
            "eval_bleu": bleu_result["score"],
            "eval_wer": wer_result
        })
        
        # Log all metrics
        logger.info(f"BLEU score: {bleu_result['score']:.2f}")
        logger.info(f"WER: {wer_result:.4f}")
        logger.info(f"Eval Loss: {metrics['eval_loss']:.4f}")
        
        # Track metrics history
        self.metrics_history["eval_bleu"].append(bleu_result["score"])
        self.metrics_history["eval_wer"].append(wer_result)
        self.metrics_history["eval_loss"].append(metrics["eval_loss"])
        
        # Track best metrics
        if bleu_result["score"] > self.best_metrics["eval_bleu"]:
            self.best_metrics["eval_bleu"] = bleu_result["score"]
            logger.info(f"New best BLEU score: {self.best_metrics['eval_bleu']:.2f}")
            
        if wer_result < self.best_metrics["eval_wer"]:
            self.best_metrics["eval_wer"] = wer_result
            logger.info(f"New best WER: {self.best_metrics['eval_wer']:.4f}")
        
        # Log trend information for early stopping insights
        if len(self.metrics_history["eval_bleu"]) >= 3:
            last_three_bleu = self.metrics_history["eval_bleu"][-3:]
            bleu_trend = [b - a for a, b in zip(last_three_bleu[:-1], last_three_bleu[1:])]
            logger.info(f"BLEU score trend (last 3 evals): {bleu_trend}")
            
            last_three_loss = self.metrics_history["eval_loss"][-3:]
            loss_trend = [a - b for a, b in zip(last_three_loss[:-1], last_three_loss[1:])]
            logger.info(f"Loss trend (last 3 evals): {loss_trend}")
        
        return metrics
# Main function
import sys
def main():
    """Main function for fine-tuning the model."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "full_training_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    try:
        # Load model and processor
        logger.info(f"Loading model and processor from {MODEL_ID}")
        tokenizer = SeamlessM4TTokenizer.from_pretrained(MODEL_ID)
        processor = SeamlessM4TProcessor.from_pretrained(
            MODEL_ID,
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG
        )
        model = SeamlessM4TModel.from_pretrained(MODEL_ID)
        
        # Set up decoder parameters
        model, forced_decoder_ids = setup_decoder_params(model, tokenizer)
        
        # Load datasets from separate files
        train_datasets, eval_dataset, test_dataset = load_datasets()
        logger.info(f"Datasets loaded: {len(train_datasets)} training samples, "
    f"{len(eval_dataset)} evaluation samples")
        
        # Create data collators
        train_data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor, 
            apply_augmentation=False  # Apply augmentation during training
        )
        
        eval_data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor, 
            apply_augmentation=False  # No augmentation for evaluation
        )
        
        # Calculate training steps for scheduler
        train_steps = (
            len(train_datasets) 
            // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) 
            * NUM_EPOCHS
        )
        
        # Configure training arguments
        logger.info("Setting up training arguments")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=2,  
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_EPOCHS,
            fp16=True if device == "cuda" else False,
            eval_strategy="steps",
            eval_steps=0.05,  # Evaluate every 5% of training
            save_strategy="steps",
            save_steps=0.05,  # Save every 5% of training
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=50,
            save_total_limit=3,  # Keep only the 3 best checkpoints
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_bleu",
            greater_is_better=True,
            report_to="tensorboard",
            gradient_checkpointing=True,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            fp16_opt_level="O1",
            dataloader_num_workers=4  # Adjust based on your CPU
        )
        
        # Create custom trainer
        logger.info("Creating trainer")
        trainer = CustomSeq2SeqTrainer(
            forced_decoder_ids=forced_decoder_ids,
            model=model,
            args=training_args,
            train_dataset=train_datasets,
            eval_dataset=eval_dataset,
            data_collator=train_data_collator,
            processor=processor,
            # Enhanced early stopping configuration
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.005
            )]
        )
        
        # Train model
        logger.info("Starting training")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        
        # Log complete metrics history for analysis
        logger.info("Complete metrics history:")
        for metric_name, values in trainer.metrics_history.items():
            logger.info(f"{metric_name}: {values}")
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    finally:
        log_file.close()

if __name__ == "__main__":
    main()
