import json
import os
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4TModel
import evaluate
import jiwer
from tqdm import tqdm

# === CONFIG ===
checkpoint_dir = "./seamless_m4t_finetuned"
checkpoint_path_lst = sorted([
    os.path.join(checkpoint_dir, d) 
    for d in os.listdir(checkpoint_dir) 
    if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint-")
])

jsonl_files = [
    "2_hours.jsonl", "0.001_audio.jsonl", "0.01_audio.jsonl", "0.005_audio.jsonl",
    "0.05_audio.jsonl", "snr10_noisy_output.jsonl", "snr15_noisy_output.jsonl", "snr25_noisy_output.jsonl"
]

base_jsonl_path = "./jsonl files"
output_dir = "./evaluation_results"

# Loop through each input JSONL file
for jsonl_file in jsonl_files:
    jsonl_path = os.path.join(base_jsonl_path, jsonl_file)
    print(f"\n=== Running evaluation for: {jsonl_file} ===\n")

    os.makedirs(output_dir, exist_ok=True)

    # === Load Metrics ===
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    bertscore = evaluate.load("bertscore")
    comet = evaluate.load("comet", model="Unbabel/wmt22-comet-da")

    # === Loop over checkpoints ===
    for idx, ckp in enumerate(checkpoint_path_lst, start=1):
        print(f"\n=== Evaluating checkpoint {idx}: {ckp} ===")
        device = "cuda"  # force GPU
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
            f"Total Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

        # Load model & processor
        processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
        model = SeamlessM4TModel.from_pretrained(
            ckp,
            torch_dtype=torch.float32
        )
        model.to(device)  # move entire model to CUDA
        model.eval()

        # Prepare accumulators
        predictions, references = [], []
        wer_scores = []
        evaluation_results = []

        # Read data
        with open(jsonl_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # Try to extract sources for COMET (if available)
        if "text" in data[0].get("source", {}):
            sources = [entry["source"]["text"].strip() for entry in data]
        else:
            sources = None

        # Inference loop
        for i, entry in enumerate(tqdm(data, desc="Samples")):
            path = entry["source"]["audio_local_path"]
            ref = entry["target"]["text"].strip()

            try:
                wav, sr = torchaudio.load(path)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
                # ensure mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.squeeze().numpy()

                # Preprocess & move ALL tensors to device
                inputs = processor(audios=wav, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                with torch.no_grad():
                    generated = model.generate(
                        input_features=inputs["input_features"],
                        max_new_tokens=444,
                        num_beams=1,
                        tgt_lang="hin",        # check your target lang code!
                        generate_speech=False
                    )

                # Decode output
                if hasattr(generated, "sequences"):
                    output_tokens = generated.sequences
                elif isinstance(generated, (list, torch.Tensor)):
                    output_tokens = generated[0] if isinstance(generated, list) else generated
                else:
                    output_tokens = generated

                pred = processor.decode(output_tokens[0], skip_special_tokens=True).strip()

                # metrics
                wer = jiwer.wer(ref, pred)
                predictions.append(pred)
                references.append([ref])
                wer_scores.append(wer)

                evaluation_results.append({
                    "sample_id": i,
                    "audio_path": path,
                    "reference": ref,
                    "prediction": pred,
                    "wer": wer
                })

                # free memory occasionally
                if i % 50 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {path}: {e}")
                evaluation_results.append({
                    "sample_id": i,
                    "audio_path": path,
                    "reference": ref,
                    "prediction": "",
                    "wer": 1.0,
                    "error": str(e)
                })

        # Compute overall metrics
        if predictions:
            bleu_score = bleu.compute(predictions=predictions, references=references)["score"]
            chrf_score = chrf.compute(predictions=predictions, references=references)["score"]
            flat_refs = [r[0] for r in references]
            bert_res = bertscore.compute(predictions=predictions, references=flat_refs, lang="en")
            bert_f1 = sum(bert_res["f1"]) / len(bert_res["f1"])

            if sources:
                comet_res = comet.compute(predictions=predictions, references=flat_refs, sources=sources)
            else:
                try:
                    comet_res = comet.compute(predictions=predictions, references=flat_refs)
                except Exception as e:
                    print(f"COMET metric failed: {e}")
                    comet_res = {"mean_score": 0.0}
            comet_score = comet_res["mean_score"]

            wer_score = sum(wer_scores) / len(wer_scores)
        else:
            bleu_score = chrf_score = bert_f1 = comet_score = 0.0
            wer_score = 1.0

        # Print summary
        print(f"\nSummary for checkpoint {idx}:")
        print(f"  samples: {len(data)}, succeeded: {len(predictions)}, failed: {len(data)-len(predictions)}")
        print(f"  BLEU   : {bleu_score:.2f}")
        print(f"  CHRF   : {chrf_score:.2f}")
        print(f"  BERT-F1: {bert_f1:.4f}")
        print(f"  COMET  : {comet_score:.4f}")
        print(f"  WER    : {wer_score:.4f}")

        # Save JSONL
        out_path = os.path.join(output_dir, f"eval_{os.path.splitext(jsonl_file)[0]}_ckpt{idx}.jsonl")
        with open(out_path, "w", encoding="utf-8") as out_f:
            # summary first
            summary = {
                "evaluation_summary": True,
                "total_samples": len(data),
                "successful_samples": len(predictions),
                "failed_samples": len(data) - len(predictions),
                "overall_bleu_score": bleu_score,
                "overall_chrf_score": chrf_score,
                "overall_bertscore_f1": bert_f1,
                "overall_comet_score": comet_score,
                "overall_wer_score": wer_score
            }
            out_f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            # then per-sample
            for res in evaluation_results:
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")

        torch.cuda.empty_cache()


