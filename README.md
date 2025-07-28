# Supplementary README

This document walks you through data preparation, model fine‑tuning, and evaluation up to the generation of per‑checkpoint metrics.

---

## 1. Dataset

- **Dataset paper & download:**  
  <https://dl.acm.org/doi/pdf/10.1145/3736720>  
- **Local storage:** all raw audio & transcripts live under the `ds/` folder.

---

## 2. Data Preparation

All commands assume you are in the project root.

1. **Split & align audio with transcripts**  
   ```bash
   python data.py
   # Reads:  ds/train/txt/train.yaml, train.en, train.hi
   # Writes: jsonl files/train.jsonl
   ```

2. **Convert to manifest format**  
   ```bash
   python prepare_dataset.py
   # Input:  jsonl files/train.jsonl
   # Output: jsonl files/train_manifest.jsonl
   ```

3. **Sample 200 h and 2 h subsets**  
   ```bash
   python sample.py
   # Inputs:  jsonl files/train_manifest.jsonl
   # Outputs: jsonl files/200_hours.jsonl
   #          jsonl files/2_hours.jsonl
   ```

---

## 3. Data Augmentation

### 3.1 Gaussian Noise on 200 h  
```bash
python 200_add_noise.py
# Applies Gaussian noise with var ∈ {0.001, 0.005, 0.01, 0.05}
# Produces:
#   jsonl files/0.001_audio.jsonl   + folder ./0.001_audio/
#   jsonl files/0.005_audio.jsonl   + folder ./0.005_audio/
#   jsonl files/0.01_audio.jsonl    + folder ./0.01_audio/
#   jsonl files/0.05_audio.jsonl    + folder ./0.05_audio/
```

### 3.2 Gaussian Noise on 2 h  
```bash
python 2_add_gauss.py
# Same noise levels on the 2-hour subset:
#   jsonl files/2_0.001_audio.jsonl  + folder ./2_0.001_audio/
#   … etc.
```

### 3.3 SNR‐based Noise on 2 h  
```bash
python 2_add_snr.py
# SNR ∈ {10, 15, 25} dB
#   jsonl files/snr10_noisy_output.jsonl + folder ./noisy_audio_10dB/
#   jsonl files/snr15_noisy_output.jsonl + folder ./noisy_audio_15dB/
#   jsonl files/snr25_noisy_output.jsonl + folder ./noisy_audio_25dB/
```

> The 2 h noisy variants serve as **evaluation sets** for robustness testing.

---

## 4. Model Fine‑tuning

Fine‑tune SeamlessM4T on the 200 h clean dataset:

```bash
python seamless_finetune.py
```

- Checkpoints saved under `seamless_m4t_finetuned/checkpoint-*`.

---

## 5. Evaluation

Evaluate each checkpoint on **all 8** JSONL variants:

```bash
python seamless_evaluation.py 
```

This produces per‑checkpoint, per‑dataset metrics (logged to terminal and saved as JSONL):

- **BLEU**  
- **WER**  
- **chrF**  
- **BERTScore**  
- **COMET**  


---

## 6. Example: Two-Variance Robustness Calculation

We demonstrate the t-test and RA workflow for **two training variances**: `0.0` (clean) vs. `0.05`.

### 6.1 Logged Evaluation Metrics

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

### 6.2 Pairwise t-Test on BLEU

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
# Expected output:
# t-statistic = -2.929
# p-value = 0.036
```

*Result:*  
- **t-statistic:** -2.929  
- **p-value:** 0.036  

The same test can be compared across wer and chrF metrices. This will give us a table for p value. 

*Continue with RA and robustness scoring as described in Section 5.*  

### 📊 Metric Summary: Mean and Standard Deviation

To compare model performance across different training noise levels, we compute the **mean** and **standard deviation (std)** for five evaluation metrics—**BLEU**, **WER**, **CHRF**, **BERTScore**, and **COMET**—over four test-time noise settings: `No Noise`, `SNR=10`, `SNR=15`, and `SNR=25`.

These statistics help quantify both average performance and consistency across varying noise conditions.

| Train Gaussian Var | BLEU (Mean) | BLEU (Std) | WER (Mean) | WER (Std) | CHRF (Mean) | CHRF (Std) | BERTScore (Mean) | BERTScore (Std) | COMET (Mean) | COMET (Std) |
|--------------------|-------------|------------|------------|-----------|--------------|-------------|-------------------|------------------|---------------|--------------|
| 0.0                | 58.6775     | 6.0923     | 0.4150     | 0.1012    | 75.0775      | 4.1613      | 0.9212            | 0.0132           | 0.7733        | 0.0186       |
| 0.05               | 48.5150     | 3.3243     | 0.4914     | 0.0679    | 69.1025      | 1.6220      | 0.9660            | 0.0016           | 0.7746        | 0.0100       |

### 🧾 Note
- These summary statistics form the basis for further robustness analysis using **Risk-Adjusted (RA) Scores** and **Significance Penalty** methods.

### 📉 Risk–Adjusted (RA) Scores under SNR Stress

To robustly evaluate model performance under noisy test conditions, we compute the **Risk–Adjusted (RA) Score**, which combines the average metric value with its variability across noise levels.

#### 🔧 RA Score Formula

For a metric \( m \) under training noise variance \( v \), the RA score is:

\[
\text{RA}_{m}(v) = \frac{\mu_{m}(v)}{1 + \text{CV}_{m}(v)}
\quad \text{where} \quad 
\text{CV}_{m}(v) = \frac{\sigma_{m}(v)}{\mu_{m}(v)}
\]

This penalizes metrics with high variability, emphasizing **both performance and stability** under noise.

#### ✅ Final RA Scores (SNR-based Evaluation)

| Training Gaussian Var | BLEU RA | WER RA  | CHRF RA |
|-----------------------|---------|---------|----------|
| 0.0                   | 0.9059  | 0.8040  | 0.9475   |
| 0.05                  | 0.7738  | 0.7420  | 0.8993   |

- **Higher RA score** implies better performance and robustness across noise levels.

### 📉 Significance Penalty and Rejection Rate

To quantify the statistical reliability of model performance under noisy conditions, we incorporate a **Significance Penalty** based on hypothesis testing.

#### 🔁 Rejection Rate

The **rejection rate** \( r_s(v) \) for each stress type \( s \in \{\text{SNR}, \text{GVAR}\} \) and training variance \( v \) is computed as:

\[
r_s(v) = \frac{\#\{p < 0.05\}}{\# \text{comparisons}}
\]

This reflects the proportion of pairwise t-tests that reject the null hypothesis, i.e., identify statistically significant differences in performance under noise. A high rejection rate indicates instability.

#### 🚫 Significance Penalty Score

The **significance penalty score** \( P(v) \) combines the rejection rates from both stress suites:

\[
P(v) = 1 - 0.5 \cdot r_{\text{SNR}}(v) - 0.5 \cdot r_{\text{GVAR}}(v)
\]

| Training Gaussian Var | Penalty |
|-----------------------|---------|
| 0.0                   | 0.95    | 
| 0.05                  | 0.97  |

- A value of **1** means no statistically significant differences were detected — the model is stable under noise.
- A value **closer to 0** indicates frequent significant variations — the model is less robust.

> **Note:** The full significance penalty could not yet be computed as we currently lack complete t-test statistics across all metrics. Once available, this section will be updated with exact \( r_s(v) \) and \( P(v) \) values for each configuration.

### 🛡️ Robustness Score

To holistically evaluate the model's stability and performance under noisy conditions, we compute the **Robustness Score** for each metric. This combines the **risk-adjusted score** and the **significance penalty**.

#### 🧮 Formula

For each metric \( m \in \{\text{BLEU}, \text{WER}, \text{CHRF}\} \), the robustness score under a given training noise variance \( v \) is computed as:

\[
R_m(v) = RA_{m,\text{avg}}(v) \cdot P(v)
\]

Where:

\[
RA_{m,\text{avg}}(v) = \frac{1}{2} \left(RA_{m,\text{SNR}}(v) + RA_{m,\text{GVAR}}(v)\right)
\]

- \( RA_{m,\text{SNR}}(v) \), \( RA_{m,\text{GVAR}}(v) \): Risk-adjusted scores for SNR and GVAR stress types.
- \( P(v) \): Significance penalty.

> For this study, we use \( P(v) = 1 \) as a placeholder since the full rejection rates are not available yet. Final scores can be updated accordingly once \( P(v) \) is computed.

---

#### ✅ Robustness Score Results

| Training Variance | BLEU Robustness | WER Robustness | CHRF Robustness |
|-------------------|------------------|----------------|------------------|
| 0.00              | 0.61297          | 0.58601        | 0.70260          |
| 0.05              | 0.66342          | 0.66731        | 0.80190          |

These values indicate that adding a small amount of training noise (\( v = 0.05 \)) improves the model’s robustness across all three metrics — BLEU, WER, and CHRF — under the assumption of maximal significance penalty \( P(v) = 1 \).









