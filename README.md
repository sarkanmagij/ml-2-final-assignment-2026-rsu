# Advanced Machine Learning — Final Group Project
## Yelp Restaurant Intelligence: CNN Photo Classifier + LSTM Sentiment Analyzer

---

## Project Summary

This project builds a dual deep-learning pipeline applied to the **Yelp Open Dataset** in the hospitality domain:

| Model | Data | Task | Classes |
|-------|------|------|---------|
| **CNN** | Yelp Photos (200k images) | Photo category classification | `food` · `inside` · `outside` |
| **RNN/LSTM** | Yelp Reviews (6M+ reviews) | Sentiment analysis | `negative` · `neutral` · `positive` |

**Business Integration:** Automatically flag restaurants with poor photo content AND negative review sentiment for follow-up by the Yelp Quality Team.

---

## Team Members

| Member | Responsibility |
|--------|---------------|
| Igors Uhaņs | EDA & Data Loading |
| Auseklis Sarkans | Image Preprocessing & CNN |
| Dāvis Okmanis | Text Preprocessing & RNN/LSTM |
| Monta Vilumsone | Business Integration & Evaluation |
| Ruslans Muhitovs | Deployment Prototype (Streamlit) |

---

## Project Structure

```
final-assignment/
│
├── notebook.ipynb              ← Main Jupyter Notebook (run this)
├── app.py                      ← Streamlit deployment prototype
├── requirements.txt            ← Python dependencies
├── create_notebook.py          ← Script that generated notebook.ipynb
├── README.md                   ← This file
│
├── data/
│   ├── Yelp Photos/
│   │   ├── photos/             ← 200,000 JPG images
│   │   └── photos.json         ← Photo metadata (photo_id, label, business_id)
│   └── Yelp JSON/
│       ├── yelp_academic_dataset_review.json   ← 6M+ reviews
│       ├── yelp_academic_dataset_business.json
│       └── ...
│
├── models/                     ← Saved model weights (created during training)
│   ├── cnn_scratch_best.pth    ← Best Custom CNN checkpoint
│   ├── cnn_tl_best.pth         ← Best ResNet-18 Transfer Learning checkpoint
│   ├── rnn_single_best.pth     ← Best Single LSTM checkpoint
│   ├── rnn_bi_best.pth         ← Best BiLSTM checkpoint
│   └── vocab.pkl               ← Vocabulary dictionary for text tokenisation
│
└── figures/                    ← All plots saved during notebook execution
    ├── 01_image_grid.png
    ├── 02_image_class_dist.png
    ├── 03_text_eda.png
    ├── 04_wordclouds.png
    ├── 05_augmentation.png
    ├── 06_token_lengths.png
    ├── 07_cnn_scratch_curves.png
    ├── 08_cnn_comparison_curves.png
    ├── 09_cnn_confusion.png
    ├── 10_lstm_single_curves.png
    ├── 11_rnn_comparison_curves.png
    ├── 12_rnn_confusion.png
    ├── 13_model_comparison.png
    └── 14_business_actions.png
```

---

## Setup & Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify data is extracted
The following directories must exist:
```
data/Yelp Photos/photos/        (200,098 .jpg files)
data/Yelp Photos/photos.json
data/Yelp JSON/yelp_academic_dataset_review.json
```

If the `.tar` files are still compressed, extract them:
```bash
tar -xf "data/Yelp Photos/yelp_photos.tar" -C "data/Yelp Photos/"
tar -xf "data/Yelp JSON/yelp_dataset.tar"   -C "data/Yelp JSON/"
```

### 3. Run the notebook
Open `notebook.ipynb` in Jupyter Lab or VS Code and run all cells top-to-bottom.

> **Estimated training time on Apple Silicon (MPS):**
> - CNN from scratch: ~20 min (15 epochs)
> - CNN transfer learning: ~25 min (20 epochs total across both phases)
> - Single LSTM: ~10 min (10 epochs)
> - BiLSTM: ~12 min (10 epochs)
>
> Reduce `MAX_IMGS_PER_CLASS` and `MAX_REVIEWS` in cell 3 for faster iteration.

### 4. Run the deployment prototype
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

> **Note:** The Streamlit app requires trained model weights in `models/`. Run the full notebook before launching the app.

---

## Dataset

**Source:** [Yelp Open Dataset](https://www.yelp.com/dataset)

| Dataset | Size | Used for |
|---------|------|----------|
| Yelp Photos | 200,098 images + metadata | CNN training |
| Yelp Reviews | 6M+ reviews with star ratings | RNN training |

**CNN subset used:** 1,500 images per class × 3 classes = 4,500 images  
**RNN subset used:** 5,000 reviews per sentiment class × 3 = 15,000 reviews

**Photo label mapping (CNN):**
- `food` → Class 0
- `inside` → Class 1
- `outside` → Class 2

**Stars → Sentiment mapping (RNN):**
- 1–2 stars → `negative` (Class 0)
- 3 stars → `neutral` (Class 1)
- 4–5 stars → `positive` (Class 2)

---

## Technical Architecture

### CNN Branch

| Approach | Architecture | Trainable Params |
|----------|-------------|-----------------|
| From Scratch | 4× Conv blocks (Conv-BN-ReLU-Pool) + GAP + 2× FC | ~3.5M |
| Transfer Learning | ResNet-18 (frozen) + new FC head | ~1.5K (phase 1) / ~2.7M (phase 2) |

**Training details:**
- Input size: 224×224
- Batch size: 32
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Early stopping: patience=3 on validation loss
- Augmentation: RandomCrop, RandomHorizontalFlip, RandomRotation(15°), ColorJitter
- Normalisation: ImageNet mean/std

### RNN Branch

| Variation | Architecture | Bidirectional |
|-----------|-------------|--------------|
| Single LSTM | Embedding(128) → LSTM(256, 2 layers) → FC(64) → FC(3) | No |
| BiLSTM | Embedding(128) → BiLSTM(128, 2 layers) → FC(64) → FC(3) | Yes |

**Training details:**
- Tokenisation: word-level regex (`\b[a-z']+\b`)
- Vocabulary: top 20,000 tokens from training set
- Max sequence length: 256 tokens (padding/truncation applied)
- Batch size: 32
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Early stopping: patience=3 on validation loss

### Business Integration Layer

```
CNN prediction (food | inside | outside)
          +
RNN prediction (negative | neutral | positive)
          ↓
Business recommendation engine
          ↓
┌─────────────────────────────────┬──────────────────────────────────────────────┐
│ CNN: food + RNN: positive       │ ✅ FEATURED — Highlight on Yelp              │
│ CNN: food + RNN: neutral        │ 🟡 MONITOR — Watch for sentiment changes      │
│ CNN: food + RNN: negative       │ ⚠️ REVIEW ALERT — Investigate service/quality │
│ CNN: other + RNN: positive      │ 🟡 PHOTO COACHING — Improve visual content    │
│ CNN: other + RNN: neutral       │ 🟠 IMPROVEMENT NEEDED — Both signals weak     │
│ CNN: other + RNN: negative      │ 🚨 URGENT FLAG — Escalate to account manager  │
└─────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## Notebook Structure

| Section | Content |
|---------|---------|
| 0 · Setup | Imports, paths, device, constants |
| 1 · Business Problem | Domain framing, end users, motivation |
| 2 · Image EDA | Grid display, class distribution, channel statistics |
| 3 · Text EDA | Length histogram, word clouds, sample texts |
| 4 · Image Preprocessing | Transforms, PhotoDataset class, augmentation visualisation |
| 5 · Text Preprocessing | Tokeniser, vocabulary building, ReviewDataset class |
| 6 · Train/Val/Test Split | Stratified 70/15/15 split for both datasets |
| 7 · Training Utilities | Generic train/evaluate/fit/plot functions |
| 8 · CNN from Scratch | Custom 4-block CNN, training, curves |
| 9 · CNN Transfer Learning | ResNet-18 phase 1 (head) + phase 2 (fine-tune layer4) |
| 10 · CNN Evaluation | Test metrics, confusion matrices, comparison |
| 11 · RNN Single LSTM | Single unidirectional LSTM, training, curves |
| 12 · RNN BiLSTM | Bidirectional LSTM, training, curves |
| 13 · RNN Evaluation | Test metrics, confusion matrices, comparison |
| 14 · Business Integration | Decision logic, 20-row example table, comparison chart |
| 15 · Business Framing | Decision context, error costs, workflow, ethics |
| 16 · Deployment | Streamlit app description and usage |
| 17 · References | Citations for datasets, models, papers |

---

## Presentation Structure (15–20 min)

1. **Title Slide** — Project name, team, date
2. **Business Problem** (2 min) — Why both image + text analysis?
3. **Dataset & EDA** (2 min) — Key stats, sample images, word clouds, class balance
4. **CNN Architecture & Results** (4 min) — Scratch vs TL, training curves, test metrics
5. **RNN Architecture & Results** (3 min) — Single vs BiLSTM, training curves, test metrics
6. **Business Integration** (3 min) — Decision logic, 20-example table, comparison chart
7. **Deployment Demo** (2 min) — Live Streamlit demo (or screen recording)
8. **Ethics & Real-World Use** (1 min) — Bias, fairness, workflow integration
9. **Conclusions** (1 min) — Key findings, what we'd explore next
10. **Team Contributions & Q&A**

---

## Key Results (update after training)

| Model | Test Accuracy | Test Macro F1 |
|-------|-------------|--------------|
| Custom CNN (scratch) | 81.2% | 0.812 |
| ResNet-18 (TL) | 94.7% | 0.947 |
| Single LSTM | 35.5% | 0.229 |
| BiLSTM | 72.0% | 0.720 |

---

## References

1. Asghar, N. (2016). *Yelp Dataset Challenge: Review Rating Prediction*. arXiv:1605.05362.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8).
4. Yelp Open Dataset. https://www.yelp.com/dataset
5. PyTorch Documentation. https://pytorch.org/docs/
