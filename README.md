# NOPE-HYPE

This project evaluates the robustness of speech translation models under noisy conditions using audio augmentation (Gaussian and SNR-based).

---

## Dataset

- **Dataset paper & download:**  
  <https://dl.acm.org/doi/pdf/10.1145/3736720>  
- **Local storage:** all raw audio & transcripts live under the `ds/` folder.

---

## How to Run

### 1. Environment Setup

### Prerequisites

- Python 3.11 recommended
- For best compatibility, use [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Recommended tools:
  - `conda` for managing the environment 
  - `pip` if you're using a virtualenv-based setup
  - CUDA 12.x installed if using GPU acceleration (for torch + cu128 build)

---

### Option 1: Using Conda

> Best for GPU compatibility, `mkl-service`, and system-wide dependencies

```bash
# Create and activate environment with a compatible Python version
conda create -n myenv python=3.11
conda activate myenv

# Install dependencies (edit this path if needed)
pip install -r requirements.txt
```

### Option 2: Using Virtualenv (pip only)

```bash
python3.10 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```


### 2. Run the Full Pipeline

Make the script executable and run it:

```bash
chmod +x run.sh
./run.sh
```

This will sequentially run:

1. `prepare_data.py` – Prepares JSONL metadata.
2. `augment_audio.py` – Adds noise to audio.
3. `model.py` – Trains the SeamlessM4t model.
4. `model_eval.py` – Evaluates model performance.

---

### 3. Audio Augmentation

- **200-hour set**: Only 20% is augmented with Gaussian noise (variance provided via CLI).
- **2-hour set**: Fully augmented using:
  - Gaussian noise with var : 0.001, 0.005, 0.01, 0.05 
  - SNR levels: 10 dB, 15 dB, 25 dB

Noisy audio is saved in folders like:

```
0.005_audio/       # Gaussian noise with var=0.005
10dB_audio/        # SNR-based noisy files
```

---

### 4. Evaluation

Metrics computed:

- **BLEU** (`sacrebleu`)
- **WER** and **CHRF** (`evaluate`, `jiwer`)
- **BERTSCORE** and **COMET** (optional, if installed)

Evaluation results and trained models are saved automatically.

---

## 5. Example: Robustness Calculation

We demonstrate the t-test and RA workflow for **two training variances**: `0.0` (clean) vs. `0.05`.

### 5.1 Logged Evaluation Metrics

| train_var | noise_rate | BLEU  | WER    | chrF   | BERTScore | COMET  |
|-----------|------------|-------|--------|--------|-----------|--------|
| **0.0**   | No noise   | 63.66 | 0.3300 | 77.51  | 0.9302    | 0.7880 |
|           | 10         | 52.47 | 0.4700 | 69.38  | 0.9032    | 0.7473 |
|           | 15         | 54.43 | 0.5300 | 74.68  | 0.9196    | 0.7724 |
|           | 25         | 64.15 | 0.3300 | 78.74  | 0.9319    | 0.7855 |
| **0.05**  | No noise   | 51.16 | 0.4492 | 70.56  | 0.9673    | 0.7841 |
|           | 10         | 44.24 | 0.5890 | 67.00  | 0.9639    | 0.7619 |
|           | 15         | 47.52 | 0.4859 | 68.67  | 0.9656    | 0.7715 |
|           | 25         | 51.14 | 0.4414 | 70.18  | 0.9671    | 0.7808 |

### 5.2 Pairwise t-Test on BLEU

```python
import numpy as np
from scipy.stats import t
from math import sqrt

# BLEU scores for Var = 0.05 vs. Var = 0.0
bleu_var_005 = np.array([51.16, 44.24, 47.52, 51.14])  # Var = 0.05
bleu_var_000 = np.array([63.66, 52.47, 54.43, 64.15])  # Var = 0.0

n1, n2 = len(bleu_var_005), len(bleu_var_000)
mean1, mean2 = bleu_var_005.mean(), bleu_var_000.mean()
var1, var2   = bleu_var_005.var(ddof=1), bleu_var_000.var(ddof=1)

t_stat = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)

# Degrees of freedom (Welch–Satterthwaite)
df_numer = (var1/n1 + var2/n2)**2
df_denom = ((var1/n1)**2/(n1-1)) + ((var2/n2)**2/(n2-1))
df = df_numer / df_denom

p_value = 2 * t.sf(abs(t_stat), df)

print(f"t-statistic = {t_stat:.3f}")
print(f"degrees of freedom = {df:.2f}")
print(f"p-value = {p_value:.3f}")
```

*Result:*  
- **t-statistic:** -2.929  
- **p-value:** 0.036  

Run the same code for scores across wer and chrF. This will give us a table for p value. 

*Continue with RA and robustness scoring as described in Section 5.*  

### 5.3 Metric Summary: Mean and Standard Deviation

To compare model performance across different training noise levels, we compute the **mean** and **standard deviation (std)** for five evaluation metrics—**BLEU**, **WER**, **CHRF**, **BERTScore**, and **COMET**—over four test-time noise settings: `No Noise`, `SNR=10`, `SNR=15`, and `SNR=25`.

These statistics help quantify both average performance and consistency across varying noise conditions.

| Train Gaussian Var | BLEU (Mean) | BLEU (Std) | WER (Mean) | WER (Std) | CHRF (Mean) | CHRF (Std) | BERTScore (Mean) | BERTScore (Std) | COMET (Mean) | COMET (Std) |
|--------------------|-------------|------------|------------|-----------|--------------|-------------|-------------------|------------------|---------------|--------------|
| 0.0                | 58.6775     | 6.0923     | 0.4150     | 0.1012    | 75.0775      | 4.1613      | 0.9212            | 0.0132           | 0.7733        | 0.0186       |
| 0.05               | 48.5150     | 3.3243     | 0.4914     | 0.0679    | 69.1025      | 1.6220      | 0.9660            | 0.0016           | 0.7746        | 0.0100       |

###  Note
- These summary statistics form the basis for further robustness analysis using **Risk-Adjusted (RA) Scores** and **Significance Penalty** methods.

### 5.4 Risk–Adjusted (RA) Scores under SNR Stress

To robustly evaluate model performance under noisy test conditions, we compute the **Risk–Adjusted (RA) Score**, which combines the average metric value with its variability across noise levels.

#### RA Score Formula

For a metric \( m \) under training noise variance \( v \), the RA score is:

<pre> Risk-adjusted Score (RA): RAₘ(v) = μₘ(v) / [1 + CVₘ(v)] where CVₘ(v) = σₘ(v) / μₘ(v) </pre>

This penalizes metrics with high variability, emphasizing **both performance and stability** under noise.

#### Final RA Scores (SNR-based Evaluation)

| Training Gaussian Var | BLEU RA | WER RA  | CHRF RA |
|-----------------------|---------|---------|----------|
| 0.0                   | 0.9059  | 0.8040  | 0.9475   |
| 0.05                  | 0.7738  | 0.7420  | 0.8993   |

- **Higher RA score** implies better performance and robustness across noise levels.

### 5.5 Significance Penalty and Rejection Rate

To quantify the statistical reliability of model performance under noisy conditions, we incorporate a **Significance Penalty** based on hypothesis testing.

#### Rejection Rate

The **rejection rate** \( r_s(v) \) for each stress type \( s \in \{\text{SNR}, \text{GVAR}\} \) and training variance \( v \) is computed as:

<pre> rₛ(v) = # {p < 0.05} / # comparisons </pre>

This reflects the proportion of pairwise t-tests that reject the null hypothesis, i.e., identify statistically significant differences in performance under noise. A high rejection rate indicates instability.

#### Significance Penalty Score

The **significance penalty score** \( P(v) \) combines the rejection rates from both stress suites:

<pre> P(v) = 1 - 0.5 · r_SNR(v) - 0.5 · r_GVAR(v) </pre>

| Training Gaussian Var | Penalty |
|-----------------------|---------|
| 0.0                   | 0.95    | 
| 0.05                  | 0.97  |

- A value of **1** means no statistically significant differences were detected — the model is stable under noise.
- A value **closer to 0** indicates frequent significant variations — the model is less robust.

> **Note:** The full significance penalty could not yet be computed as we currently lack complete t-test statistics across all metrics. Once available, this section will be updated with exact r_s(v) and P(v) values for each configuration.

### 5.6 Robustness Score

To holistically evaluate the model's stability and performance under noisy conditions, we compute the **Robustness Score** for each metric. This combines the **risk-adjusted score** and the **significance penalty**.

#### Formula

For each metric \( m \in \{\text{BLEU}, \text{WER}, \text{CHRF}\} \), the robustness score under a given training noise variance \( v \) is computed as:

<pre> Rₘ(v) = RAₘ,avg(v) · P(v) </pre> 

Where:

<pre> RAₘ,avg(v) = (1/2) · [RAₘ,SNR(v) + RAₘ,GVAR(v)] </pre>

- <pre> RAₘ,SNR(v), RAₘ,GVAR(v) </pre>: Risk-adjusted scores for SNR and GVAR stress types.
- \( P(v) \): Significance penalty.

---

#### Robustness Score Results

| Training Variance | BLEU Robustness | WER Robustness | CHRF Robustness |
|-------------------|------------------|----------------|------------------|
| 0.00              | 0.61297          | 0.58601        | 0.70260          |
| 0.05              | 0.66342          | 0.66731        | 0.80190          |

These values indicate that adding a small amount of training noise \( v = 0.05 \) improves the model’s robustness across all three metrics — BLEU, WER, and CHRF.
