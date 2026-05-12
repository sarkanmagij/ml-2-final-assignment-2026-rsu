# Yelp Restaurant Intelligence
### Advanced Machine Learning — Final Group Project
**Rīgas Stradiņš University · Spring 2026**

> CNN photo classifier + BiLSTM sentiment analyzer, integrated into a business decision layer for the Yelp hospitality domain.

| Resource | Link |
|----------|------|
| Open in Colab | [notebook.ipynb](https://colab.research.google.com/github/sarkanmagij/ml-2-final-assignment-2026-rsu/blob/main/notebook.ipynb) |
| Live Demo (HuggingFace Space) | [tusama/yelp-restaurant-intelligence](https://huggingface.co/spaces/tusama/yelp-restaurant-intelligence) |
| GitHub Repo | [sarkanmagij/ml-2-final-assignment-2026-rsu](https://github.com/sarkanmagij/ml-2-final-assignment-2026-rsu) |

---

## 1. Business Problem & Motivation

Yelp hosts millions of restaurant listings, each generating two independent quality signals:

- **Visual signal** — user-uploaded photos reveal whether a restaurant leads with food photography (high-intent, conversion-positive) or only has interior/exterior shots.
- **Textual signal** — written reviews carry explicit sentiment that predicts customer retention and star trajectory.

Neither signal alone is sufficient. A restaurant can have stunning food photos but a collapsing reputation in recent reviews (an emerging service problem), or can have overwhelmingly positive reviews but visually unengaging photos (a missed marketing opportunity). **By analysing both signals simultaneously**, Yelp's Quality Team can triage thousands of listings automatically — escalating urgent cases and surfacing coaching opportunities — without manual review.

**End users:** Yelp Quality Team, account managers, restaurant partners.

---

## 2. Dataset & EDA

### Datasets

| Dataset | Source | Size | Task |
|---------|--------|------|------|
| Yelp Photos | [yelp.com/dataset](https://www.yelp.com/dataset) | 200,098 images + `photos.json` | CNN training |
| Yelp Reviews | [yelp.com/dataset](https://www.yelp.com/dataset) | 6.9M reviews + star ratings | RNN training |

Both datasets come from the same open dataset release — every photo and review is tied to a real Yelp business, making the business integration natural and meaningful.

### Subsets used

| Split | Images (CNN) | Reviews (RNN) |
|-------|-------------|---------------|
| Training (70%) | 3,150 | 10,500 |
| Validation (15%) | 675 | 2,250 |
| Test (15%) | 675 | 2,250 |
| **Total** | **4,500** (1,500/class) | **15,000** (5,000/class) |

Stratified split, random seed = 42.

### Label mappings

**CNN — Photo category (3 classes):**
| Label | Class | Meaning |
|-------|-------|---------|
| `food` | 0 | High-intent food photography |
| `inside` | 1 | Interior / ambience |
| `outside` | 2 | Exterior / facade |

**RNN — Review sentiment (3 classes):**
| Stars | Class | Label |
|-------|-------|-------|
| 1–2 | 0 | Negative |
| 3 | 1 | Neutral |
| 4–5 | 2 | Positive |

### EDA highlights

- **Images:** 224×224 px input, balanced 3-class distribution (500 test samples each). Mean pixel statistics follow ImageNet distribution. Augmentation visualisations confirm diversity of crops and colour jitter. *(see `figures/01_image_grid.png`, `figures/02_image_class_dist.png`)*
- **Text:** Median review length ≈ 90 tokens; 95th percentile ≈ 256 tokens → max sequence length set to 256 (< 5% truncated). Word clouds show strong class-discriminative vocabulary: "amazing", "delicious" dominate positive; "terrible", "waited", "cold" dominate negative. *(see `figures/03_text_eda.png`, `figures/04_wordclouds.png`)*

---

## 3. Architecture Design

### CNN Branch

```
Input 224×224×3
        │
┌───────┴───────────────────────────────┐
│  Approach 1: Custom CNN (scratch)     │
│  Conv(32)-BN-ReLU-Pool                │
│  Conv(64)-BN-ReLU-Pool                │
│  Conv(128)-BN-ReLU-Pool               │
│  Conv(256)-BN-ReLU-Pool               │
│  GlobalAvgPool → FC(256) → FC(3)      │
│  ~3.5M trainable parameters           │
└───────────────────────────────────────┘
        │
┌───────┴───────────────────────────────┐
│  Approach 2: ResNet-18 Transfer Learn │
│  Phase 1: Frozen backbone + new head  │
│           ~1.5K trainable params      │
│  Phase 2: Unfreeze layer4 + fine-tune │
│           ~2.7M trainable params      │
└───────────────────────────────────────┘
```

Training: Adam (lr=1e-3, wd=1e-4), ReduceLROnPlateau (patience=2, factor=0.5), early stopping (patience=3).
Augmentation: RandomCrop, RandomHorizontalFlip, RandomRotation(15°), ColorJitter. ImageNet normalisation.

### RNN Branch

```
Input: tokenised review (max 256 tokens)
        │
Embedding(vocab=20000, dim=128)
        │
┌───────┴───────────────────────────────┐
│  Variation 1: Single LSTM             │
│  LSTM(hidden=256, layers=2, drop=0.3) │
│  → FC(64) → FC(3)                     │
└───────────────────────────────────────┘
        │
┌───────┴───────────────────────────────┐
│  Variation 2: BiLSTM                  │
│  BiLSTM(hidden=128, layers=2, drop=0.3│
│  → FC(64) → FC(3)                     │
└───────────────────────────────────────┘
```

Tokenisation: word-level regex `\b[a-z']+\b` — chosen for simplicity and interpretability over subword; OOV handled via `<UNK>` token. Vocabulary: top 20,000 tokens from training set.

Training: Adam (lr=1e-3, wd=1e-5), early stopping (patience=3 on val loss).

### Business Integration Layer

```
CNN prediction          RNN prediction
(food / inside / outside)   (negative / neutral / positive)
         │                          │
         └──────────┬───────────────┘
                    ▼
          Decision rule engine
                    ▼
┌────────────────────────┬──────────────────────────────────────┐
│ food  + positive       │ FEATURED — Highlight on Yelp          │
│ food  + neutral        │ MONITOR — Watch sentiment trajectory  │
│ food  + negative       │ REVIEW ALERT — Investigate quality    │
│ other + positive       │ PHOTO COACHING — Improve visuals      │
│ other + neutral        │ IMPROVEMENT NEEDED — Both weak        │
│ other + negative       │ URGENT FLAG — Escalate to manager     │
└────────────────────────┴──────────────────────────────────────┘
```

---

## 4. Results: CNN

| Model | Test Accuracy | Test Macro F1 | Trainable Params |
|-------|-------------|--------------|-----------------|
| Custom CNN (scratch) | 81.2% | 0.812 | ~3.5M |
| ResNet-18 Phase 1 only | 89.4% | 0.893 | ~1.5K |
| **ResNet-18 TL (Phase 1+2)** | **94.7%** | **0.947** | ~2.7M |

**Why transfer learning wins:** ResNet-18 pre-trained on ImageNet already encodes low-level edge detectors and mid-level texture detectors in its frozen layers. Restaurant photos share significant visual structure with ImageNet (objects, textures, lighting). Fine-tuning `layer4` adapts the high-level features to food/inside/outside discrimination without overfitting on our 4,500-sample subset.

Training curves and confusion matrices: `figures/07_cnn_scratch_curves.png`, `figures/08_cnn_comparison_curves.png`, `figures/09_cnn_confusion.png`

---

## 5. Results: RNN

| Model | Test Accuracy | Test Macro F1 |
|-------|-------------|--------------|
| Single LSTM (2 layers) | 35.5% | 0.229 |
| **BiLSTM (2 layers)** | **72.0%** | **0.720** |

**Why BiLSTM wins:** Bidirectional processing allows the model to use future context when encoding each token — e.g., the word "not" is better understood as a negation when the model also sees "bad" coming after it. The single LSTM collapsed into predicting "positive" for most inputs (reflected in its poor macro F1 = 0.229 vs. accuracy of 35.5%), a known failure mode when the training set has any subtle class imbalance and the model lacks expressive capacity.

**Neutral class is hardest:** Both models struggle with 3-star "neutral" reviews — they are linguistically ambiguous by design (mixed positive and negative statements). BiLSTM achieves 68% on neutral vs. 79% on negative and 83% on positive.

Training curves and confusion matrices: `figures/10_lstm_single_curves.png`, `figures/11_rnn_comparison_curves.png`, `figures/12_rnn_confusion.png`

---

## 6. Business Integration

The integration layer takes the best model from each branch (ResNet-18 TL + BiLSTM) and applies the 6-action decision table above to every restaurant listing.

**Why combined output beats either model alone:**

| Scenario | CNN only | RNN only | Combined |
|----------|---------|---------|---------|
| Good food photos, hidden service failure | No signal | ✅ Catches it | ✅ REVIEW ALERT |
| Terrible photos, loyal regulars | ✅ Flags it | No signal | ✅ PHOTO COACHING |
| Both signals bad | ✅ Flags it | ✅ Flags it | ✅ URGENT FLAG (escalated) |
| Both signals good | No action needed | No action needed | ✅ FEATURED |

A 20-row example table showing both model predictions side-by-side with the combined recommendation is generated in notebook Section 14 (`figures/14_business_actions.png`). The comparison chart (`figures/13_model_comparison.png`) shows CNN accuracy (94.7%), RNN accuracy (72.0%), and the business-action agreement rate of the integrated system.

---

## 7. Interpretability

- **CNN (Grad-CAM):** Gradient-weighted class activation maps highlight which regions of a photo drove the classification. Food photos: activation concentrates on the dish. Inside photos: activation on furniture/lighting. This confirms the model is attending to semantically meaningful regions, not background artefacts. *(notebook Section 10)*
- **RNN (token-level salience):** Gradient norms on the embedding layer reveal which tokens most influenced the sentiment prediction. Positive: "amazing", "perfect", "loved". Negative: "terrible", "cold", "never". Neutral: hedging words ("okay", "decent", "but") dominate, consistent with mixed-signal reviews. *(notebook Section 13)*

---

## 8. Deployment Demo

An interactive Streamlit app is live at **[huggingface.co/spaces/tusama/yelp-restaurant-intelligence](https://huggingface.co/spaces/tusama/yelp-restaurant-intelligence)**.

The user can:
1. **Upload a restaurant photo** → ResNet-18 predicts the category with a confidence score.
2. **Paste a customer review** → BiLSTM predicts sentiment with a confidence score.
3. **See the combined business recommendation** with colour-coded action.

Pre-loaded demo examples are included for quick evaluation without any uploads.

To run locally:
```bash
pip install -r requirements.txt
# model weights are downloaded automatically from HuggingFace on first run
streamlit run app.py
```

---

## 9. Business Impact & Ethics

### Business impact

| Question | Answer |
|----------|--------|
| Who uses this? | Yelp Quality Team, account managers |
| What decision does it support? | Triage and prioritisation of restaurant listings for human review or coaching |
| How does it integrate? | Runs nightly on new photo+review batches; outputs a ranked queue for account managers |
| False positive cost (CNN) | A food-photo restaurant incorrectly flagged wastes an account manager's time |
| False negative cost (CNN) | A restaurant with no food photos goes uncoached — missed revenue opportunity |
| False positive cost (RNN) | A happy restaurant flagged as negative damages the partner relationship |
| False negative cost (RNN) | A declining restaurant missed — customer churn goes undetected |

At 94.7% CNN accuracy and 72.0% RNN accuracy, the combined system reliably handles the easy cases (clear food photos, clearly positive/negative reviews). Human review is still recommended for borderline predictions (confidence < 60%).

### Ethical considerations

- **Demographic bias in photos:** ResNet-18 pre-trained on ImageNet may perform differently across cuisines or restaurant styles under-represented in ImageNet (e.g., non-Western food presentations). Mitigation: audit confusion matrices per cuisine type; consider domain-specific fine-tuning data.
- **Language bias in reviews:** The BiLSTM vocabulary was built from English reviews only. Non-English reviews are tokenised to `<UNK>` and their sentiment will be misclassified. Mitigation: language detection pre-filter; multilingual embeddings for future iterations.
- **Star-rating as ground truth:** Mapping 3 stars to "neutral" is a heuristic. A 3-star review for a cheap lunch spot may signal satisfaction; for a fine-dining restaurant it signals disappointment. Mitigation: stratify by price tier in future.
- **Feedback loop risk:** If Yelp demotes listings flagged by this system, those businesses receive fewer visitors and fewer reviews, reinforcing the flag — regardless of whether it was correct. Any deployment must include an appeal mechanism.

---

## 10. Conclusions & Lessons Learned

**Key findings:**
- Transfer learning is dramatically more sample-efficient than training from scratch: ResNet-18 TL achieves 94.7% vs. 81.2% with 13× fewer effective gradient steps on task-specific features.
- BiLSTM's 72.0% accuracy on 3-class review sentiment is reasonable but reveals that 3-star "neutral" reviews are genuinely ambiguous — not a model failure, but a data labelling challenge.
- Business integration creates value beyond individual model performance: the combined layer correctly differentiates "photo problem" from "service problem" from "both", enabling targeted interventions rather than a binary flag.

**What we would explore next:**
- Replace BiLSTM with a fine-tuned DistilBERT encoder (expected +10–15% accuracy on neutral class).
- Expand CNN classes to include `drink`, `menu`, `staff` from the full Yelp photo taxonomy.
- Build a joint model (CNN feature vector ⊕ LSTM hidden state → shared FC head) on matched photo+review pairs.
- Add time-series dimension: track sentiment trajectory per restaurant over rolling 90-day windows.

---

## Team Contributions

| Member | Responsibility |
|--------|---------------|
| Igors Uhaņs | EDA & Data Loading (notebook Sections 1–3) |
| Auseklis Sarkans | Image Preprocessing & CNN (Sections 4, 8–10) |
| Dāvis Okmanis | Text Preprocessing & RNN/LSTM (Sections 5, 11–13) |
| Monta Vilumsone | Business Integration & Evaluation (Sections 6, 14–15) |
| Ruslans Muhitovs | Deployment Prototype — `app.py`, Streamlit app (Section 16) |

---

## Project Structure

```
final-assignment/
├── notebook.ipynb              ← Main Jupyter Notebook (65 cells, run top-to-bottom)
├── app.py                      ← Streamlit deployment prototype
├── requirements.txt            ← Python dependencies
├── Dockerfile                  ← Container definition for HuggingFace Space
├── README.md                   ← This file (presentation outline)
├── demo_photos/                ← Sample images for the Streamlit demo
├── figures/                    ← All 14 plots generated during notebook execution
└── content/data/               ← Yelp dataset (not in repo — download below)
    ├── Yelp JSON/
    └── Yelp Photos/
```

Model weights (`.pth`, `vocab.pkl`) are hosted on HuggingFace and downloaded automatically by the notebook in Colab, or by `app.py` on first run.

### Getting the Yelp dataset

The dataset is free for academic use:
1. Request access at [yelp.com/dataset](https://www.yelp.com/dataset)
2. Download and extract into `content/data/Yelp JSON/` and `content/data/Yelp Photos/`

**Training time estimates (Apple Silicon MPS):**
- CNN from scratch: ~20 min (15 epochs)
- CNN transfer learning: ~25 min (20 epochs, 2 phases)
- Single LSTM: ~10 min (10 epochs)
- BiLSTM: ~12 min (10 epochs)

---

## References

1. Asghar, N. (2016). *Yelp Dataset Challenge: Review Rating Prediction*. arXiv:1605.05362.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
4. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*. ICCV 2017.
5. Yelp Open Dataset. https://www.yelp.com/dataset
6. PyTorch Documentation. https://pytorch.org/docs/stable/
7. Hugging Face Spaces. https://huggingface.co/docs/hub/spaces
