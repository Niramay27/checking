#!/bin/bash

set -e  # Exit immediately if a command fails

echo "==========================================="
echo "Step 1: Preparing clean training + eval data"
echo "==========================================="
python3 src/prepare_data.py

echo ""
echo "==========================================="
echo "Step 2: Augmenting audio (Gaussian + SNR)"
echo "==========================================="
# You can change --gauss_var as needed, e.g., 0.001, 0.005, etc.
python3 src/augment_audio.py --gauss_var 0.01

echo ""
echo "==========================================="
echo "Step 3: Training the speech translation model"
echo "==========================================="
python3 src/model.py

echo ""
echo "==========================================="
echo "Step 4: Evaluating model performance (BLEU, WER, CHRF)"
echo "==========================================="
python3 src/model_eval.py

echo ""
echo "All training and evaluation completed successfully!"
