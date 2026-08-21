<div align="center">

# Auditing and Mitigating Privacy Leakage in Cloud-Edge Collaborative Decoding

[![EMNLP 2026 Findings](https://img.shields.io/badge/EMNLP%202026-Findings-7B1FA2?style=flat-square)](https://2026.emnlp.org/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-2E7D32?style=flat-square)](LICENSE)
[![Datasets](https://img.shields.io/badge/Data-MedPriv%20%7C%20CommPriv-D97706?style=flat-square)](#data)

<img src="assets/collaborative_decoding.png" alt="Cloud-side and edge-side collaborative decoding" width="68%">

</div>

| **DecodeLeak** | **CoVeil** |
| :---: | :---: |
| Private-evidence recall · context inversion | Training-free privacy-aware fusion |

<p align="center">
  <img src="assets/decodeleak_audit.png" alt="DecodeLeak privacy audit" width="100%">
</p>

## Run

```bash
export LARGE_MODEL=/path/to/large-model
export SMALL_MODEL=/path/to/small-model
export DATASET=dataset/MEDPRIV.jsonl

bash sh/LM_fusion_defense.sh   # cloud-side fusion
bash sh/SM_fusion_defense.sh   # edge-side fusion
```

## Evaluate

```bash
RESULTS=outputs/cloud_side/results.jsonl \
TOKENIZER="$LARGE_MODEL" \
bash sh/evaluate.sh
```

`Accuracy` · `Token-ER@K` · `ROUGE1-ER@K` · `Span-ER@K` · `AUC@K` · `ROUGE-1 Recall/F1`

## Data

[**MedPriv**](dataset/MEDPRIV.jsonl) · [**CommPriv**](dataset/COMMPRIV.jsonl)

<p align="center"><sub>Apache-2.0 licensed code · Dataset terms follow the original sources.</sub></p>
