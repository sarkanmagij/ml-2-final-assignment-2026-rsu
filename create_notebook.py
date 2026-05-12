"""
Generates notebook.ipynb for the Advanced ML Final Project.
Run: python3 create_notebook.py
"""
import json, uuid

def _id():
    return str(uuid.uuid4())[:8]

def md(src):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "id": _id(), "metadata": {}, "execution_count": None, "outputs": [], "source": src}

cells = []

# ===========================================================
# WORKING DIRECTORY FIX (must be first code cell)
# ===========================================================
cells.append(md("## Working Directory Setup"))
cells.append(code(
"""# This cell ensures the notebook can find the data/ and models/ folders
# regardless of where VS Code / Jupyter started the kernel.
import os
from pathlib import Path

# Explicit project root — update this if you move the folder
PROJECT_ROOT = Path('/Users/igorsu/Documents/GitHub/ml-skolai-2/final-assignment')

if PROJECT_ROOT.exists():
    os.chdir(PROJECT_ROOT)
else:
    # Fallback: walk up from cwd until we find the data/ folder
    _cwd = Path.cwd()
    for _ in range(6):
        if (_cwd / 'data').exists():
            os.chdir(_cwd)
            break
        _cwd = _cwd.parent

print(f"Working directory: {os.getcwd()}")
assert Path('content/data').exists(), (
    f"ERROR: 'content/data/' not found in {os.getcwd()}. "
    "Update PROJECT_ROOT in this cell to your actual project path."
)
print("content/data/ folder found — ready to run.")
"""
))

# ===========================================================
# TITLE
# ===========================================================
cells.append(md(
"""# Advanced Machine Learning — Final Group Project
## Yelp Restaurant Intelligence: CNN Photo Classifier + LSTM Sentiment Analyzer

**Business Domain:** Hospitality / Restaurant Industry (Yelp Open Dataset)

| Model | Data | Task | Classes |
|-------|------|------|---------|
| CNN | Yelp Photos (200k images) | Photo category classification | food · inside · outside |
| RNN/LSTM | Yelp Reviews (6M+ reviews) | Sentiment analysis | negative · neutral · positive |

**Business Integration:** Flag restaurants whose photos are predominantly non-food AND whose reviews are predominantly negative — surfaced to a Yelp quality team dashboard for follow-up.

---
"""
))

# ===========================================================
# CONTRIBUTION TABLE
# ===========================================================
cells.append(md(
"""## Team Contribution Table

| Component | Responsible Member |
|-----------|-------------------|
| EDA & Data Loading | [Member 1] |
| Image Preprocessing & CNN | [Member 2] |
| Text Preprocessing & RNN | [Member 3] |
| Business Integration & Evaluation | [Member 4] |
| Deployment Prototype (Streamlit) | [Member 5] |
| Presentation | All |
"""
))

# ===========================================================
# SECTION 0 — SETUP
# ===========================================================
cells.append(md("## 0 · Setup & Imports"))
cells.append(code(
"""import json, os, re, random, pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              ConfusionMatrixDisplay, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(".")
PHOTOS_DIR   = BASE_DIR / "content/data/Yelp Photos/photos"
PHOTOS_JSON  = BASE_DIR / "content/data/Yelp Photos/photos.json"
REVIEWS_JSON = BASE_DIR / "content/data/Yelp JSON/yelp_academic_dataset_review.json"
MODELS_DIR   = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
os.makedirs('figures', exist_ok=True)   # ensure output dir exists before any savefig

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ── Tunable constants (reduce for quick testing) ──────────────────────────────
MAX_IMGS_PER_CLASS = 1500   # images sampled per CNN class
MAX_REVIEWS        = 15000  # total reviews (5 000 per sentiment class)
IMG_SIZE           = 224
MAX_SEQ_LEN        = 256
VOCAB_SIZE         = 20_000
BATCH_SIZE         = 32
CNN_EPOCHS         = 15
RNN_EPOCHS         = 10
LR                 = 1e-3
PATIENCE           = 3      # early-stopping patience
"""
))

# ===========================================================
# SECTION 1 — BUSINESS PROBLEM
# ===========================================================
cells.append(md(
"""---
## 1 · Business Problem & Motivation

Yelp hosts millions of restaurant listings, each with user-uploaded photos and written reviews.
Two independent quality signals drive consumer trust:

1. **Visual signal (CNN):** Photos labelled *food* indicate a focus on quality cuisine; photos labelled *inside* or *outside* may indicate a lack of compelling food imagery.
2. **Textual signal (RNN):** Review sentiment reflects the lived customer experience.

**Business decision:**
> *Flag any restaurant whose most recent photos skew away from food content AND whose recent reviews skew negative. Surface these to the Yelp Quality Team for outreach.*

**End users:** Yelp account managers and content-quality analysts.

**Why both signals matter:**
- Good photos but terrible reviews → likely a service/quality issue the owner hasn't fixed.
- Negative-looking photos but great reviews → possible photo coaching opportunity.
- Both negative → highest-priority intervention.

Neither model feeds into the other architecturally; integration happens in the business layer.
"""
))

# ===========================================================
# SECTION 2 — IMAGE EDA
# ===========================================================
cells.append(md("---\n## 2 · Exploratory Data Analysis — Images"))

cells.append(code(
"""# ── Load photo metadata ───────────────────────────────────────────────────────
photo_records = []
with open(PHOTOS_JSON) as f:
    for line in f:
        photo_records.append(json.loads(line.strip()))

df_photos_all = pd.DataFrame(photo_records)
print(f"Total photos in metadata: {len(df_photos_all):,}")
print("\\nLabel distribution (all photos):")
print(df_photos_all['label'].value_counts())
"""
))

cells.append(code(
"""# ── Filter to 3 classes and verify photo files exist ─────────────────────────
PHOTO_CLASSES = ['food', 'inside', 'outside']
LABEL2IDX     = {lbl: i for i, lbl in enumerate(PHOTO_CLASSES)}
IDX2LABEL_CNN = {i: lbl for lbl, i in LABEL2IDX.items()}

df_photos = df_photos_all[df_photos_all['label'].isin(PHOTO_CLASSES)].copy()
df_photos['label_idx'] = df_photos['label'].map(LABEL2IDX)

# Verify files exist (some may be missing from the archive)
df_photos['path'] = df_photos['photo_id'].apply(lambda x: PHOTOS_DIR / f"{x}.jpg")
df_photos = df_photos[df_photos['path'].apply(lambda p: p.exists())].copy()
print(f"Photos with valid files: {len(df_photos):,}")
print(df_photos['label'].value_counts())
"""
))

cells.append(code(
"""# ── Sample balanced subset (pandas-3.0-safe — no groupby.apply) ───────────────
df_sampled = pd.concat([
    df_photos[df_photos['label'] == lbl].sample(
        min(MAX_IMGS_PER_CLASS, int((df_photos['label'] == lbl).sum())),
        random_state=SEED
    )
    for lbl in PHOTO_CLASSES
]).reset_index(drop=True)

print(f"Sampled dataset size: {len(df_sampled):,}")
print(df_sampled['label'].value_counts())
"""
))

cells.append(code(
"""# ── Display 12 sample images in a grid ───────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle('Sample Images by Class', fontsize=16, fontweight='bold')

for class_idx, label in enumerate(PHOTO_CLASSES):
    subset = df_sampled[df_sampled['label'] == label].sample(4, random_state=SEED)
    for col, (_, row) in enumerate(subset.iterrows()):
        ax = axes[class_idx][col]
        img = Image.open(row['path']).convert('RGB')
        ax.imshow(img)
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig('figures/01_image_grid.png', dpi=120, bbox_inches='tight')
plt.show()
print("Grid saved.")
"""
))

cells.append(code(
"""# ── Class distribution bar chart ──────────────────────────────────────────────
counts = df_sampled['label'].value_counts().loc[PHOTO_CLASSES]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:,}', ha='center', va='bottom', fontweight='bold')
ax.set_title('Image Class Distribution (Sampled Dataset)', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
ax.set_xlabel('Photo Category')
plt.tight_layout()
plt.savefig('figures/02_image_class_dist.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

cells.append(code(
"""# ── Channel statistics (mean/std per channel on sample of 200 images) ─────────
sample_paths = df_sampled.sample(200, random_state=SEED)['path'].tolist()
means, stds = [], []
for p in sample_paths:
    arr = np.array(Image.open(p).convert('RGB').resize((64, 64))) / 255.0
    means.append(arr.mean(axis=(0, 1)))
    stds.append(arr.std(axis=(0, 1)))

ch_mean = np.mean(means, axis=0)
ch_std  = np.mean(stds, axis=0)
print("Channel statistics (RGB) on 200-image sample:")
print(f"  Mean: R={ch_mean[0]:.3f}  G={ch_mean[1]:.3f}  B={ch_mean[2]:.3f}")
print(f"  Std : R={ch_std[0]:.3f}  G={ch_std[1]:.3f}  B={ch_std[2]:.3f}")

# Sample image dimensions
sizes = []
for p in sample_paths[:50]:
    w, h = Image.open(p).size
    sizes.append((w, h))
widths, heights = zip(*sizes)
print(f"\\nImage dimensions (50-image sample):")
print(f"  Width  — min:{min(widths)}  max:{max(widths)}  mean:{np.mean(widths):.0f}")
print(f"  Height — min:{min(heights)} max:{max(heights)} mean:{np.mean(heights):.0f}")
"""
))

# ===========================================================
# SECTION 3 — TEXT EDA
# ===========================================================
cells.append(md("---\n## 3 · Exploratory Data Analysis — Text (Reviews)"))

cells.append(code(
"""# ── Load reviews and convert stars → sentiment ────────────────────────────────
def stars_to_sentiment(stars):
    if stars <= 2:   return 'negative'
    elif stars == 3: return 'neutral'
    else:            return 'positive'

SENTIMENT_CLASSES = ['negative', 'neutral', 'positive']
SENT2IDX          = {s: i for i, s in enumerate(SENTIMENT_CLASSES)}
IDX2LABEL_RNN     = {i: s for s, i in SENT2IDX.items()}

# Sample balanced: 5 000 per sentiment class (15 000 total)
PER_CLASS = MAX_REVIEWS // 3
buckets   = {s: [] for s in SENTIMENT_CLASSES}

with open(REVIEWS_JSON) as f:
    for line in f:
        if all(len(v) >= PER_CLASS for v in buckets.values()):
            break
        rec  = json.loads(line.strip())
        sent = stars_to_sentiment(int(rec['stars']))
        if len(buckets[sent]) < PER_CLASS:
            buckets[sent].append({'text': rec['text'], 'sentiment': sent,
                                  'stars': int(rec['stars']), 'label': SENT2IDX[sent]})

df_reviews = pd.DataFrame([r for v in buckets.values() for r in v])
df_reviews  = df_reviews.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Total reviews loaded: {len(df_reviews):,}")
print(df_reviews['sentiment'].value_counts())
"""
))

cells.append(code(
"""# ── Text length distribution ──────────────────────────────────────────────────
df_reviews['word_count'] = df_reviews['text'].apply(lambda t: len(t.split()))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(df_reviews['word_count'], bins=60, color='steelblue', edgecolor='black')
axes[0].axvline(df_reviews['word_count'].mean(), color='red', linestyle='--',
                label=f"Mean: {df_reviews['word_count'].mean():.0f} words")
axes[0].set_title('Review Length Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

counts_s = df_reviews['sentiment'].value_counts().loc[SENTIMENT_CLASSES]
colors_s = ['#FF4444', '#FFAA00', '#44AA44']
bars = axes[1].bar(counts_s.index, counts_s.values, color=colors_s,
                   edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, counts_s.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{val:,}', ha='center', va='bottom', fontweight='bold')
axes[1].set_title('Sentiment Class Distribution', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('figures/03_text_eda.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"Word count stats:")
print(df_reviews['word_count'].describe().round(1))
print(f"\\n% reviews > {MAX_SEQ_LEN} words: "
      f"{(df_reviews['word_count'] > MAX_SEQ_LEN).mean()*100:.1f}%")
"""
))

cells.append(code(
"""# ── Word clouds per sentiment class ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
wc_colors = ['Reds', 'Oranges', 'Greens']

for ax, sentiment, cmap in zip(axes, SENTIMENT_CLASSES, wc_colors):
    text_blob = ' '.join(df_reviews[df_reviews['sentiment'] == sentiment]['text'].tolist())
    wc = WordCloud(width=600, height=400, background_color='white',
                   colormap=cmap, max_words=80,
                   stopwords={'the','a','and','is','in','it','of','to','was','i',
                              'for','on','at','this','that','with','we','had','my',
                              'but','they','you','are','as','so','be','have','not',
                              'an','our','were','all','from','there','their','just',
                              'has','or','do','by','if','up','been','me','your',
                              'he','she','they','what','when','no','more','about',
                              'get','got','one','out','go','would','could','said'}).generate(text_blob)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(f'"{sentiment.upper()}" Reviews', fontsize=13, fontweight='bold')
    ax.axis('off')

plt.suptitle('Word Clouds by Sentiment Class', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/04_wordclouds.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

cells.append(code(
"""# ── 3 representative samples per class ───────────────────────────────────────
for sentiment in SENTIMENT_CLASSES:
    print(f"\\n{'='*60}")
    print(f"  SENTIMENT: {sentiment.upper()}")
    print('='*60)
    samples = df_reviews[df_reviews['sentiment'] == sentiment].sample(3, random_state=SEED)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        preview = row['text'][:250].replace('\\n', ' ')
        print(f"\\n[{i}] ({row['stars']} stars): {preview}...")
"""
))

# ===========================================================
# SECTION 4 — IMAGE PREPROCESSING
# ===========================================================
cells.append(md(
"""---
## 4 · Data Preprocessing — Images

### Augmentation Strategy
- **Random horizontal flip:** restaurants/food photos are naturally horizontally symmetric.
- **Random rotation (±15°):** phones are not always held level.
- **Colour jitter:** accounts for different lighting conditions.
- **Normalisation:** ImageNet mean/std is used because both CNN approaches (scratch and transfer learning) benefit from a consistent input range; the pre-trained ResNet was trained with these statistics.
"""
))

cells.append(code(
"""# ── Transforms ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

print("Train transform:", train_transform)
print("\\nVal/Test transform:", val_transform)
"""
))

cells.append(code(
"""# ── PhotoDataset ──────────────────────────────────────────────────────────────
class PhotoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, int(row['label_idx'])
"""
))

cells.append(code(
"""# ── Visualise augmentation examples ──────────────────────────────────────────
fig, axes = plt.subplots(2, 6, figsize=(16, 5))
fig.suptitle('Original vs Augmented Images', fontsize=14, fontweight='bold')

sample_rows = df_sampled.sample(6, random_state=1).reset_index(drop=True)
to_pil = transforms.ToPILImage()
unnorm = transforms.Compose([
    transforms.Normalize(mean=[0,0,0], std=[1/s for s in IMAGENET_STD]),
    transforms.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1,1,1]),
])

for col, (_, row) in enumerate(sample_rows.iterrows()):
    img = Image.open(row['path']).convert('RGB')
    axes[0][col].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
    axes[0][col].set_title(row['label'], fontsize=9)
    axes[0][col].axis('off')
    aug = to_pil(unnorm(train_transform(img)).clamp(0, 1))
    axes[1][col].imshow(aug)
    axes[1][col].set_title('augmented', fontsize=9, color='green')
    axes[1][col].axis('off')

axes[0][0].set_ylabel('Original', fontsize=10, fontweight='bold')
axes[1][0].set_ylabel('Augmented', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_augmentation.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

# ===========================================================
# SECTION 5 — TEXT PREPROCESSING
# ===========================================================
cells.append(md(
"""---
## 5 · Data Preprocessing — Text

### Tokenisation strategy: word-level
Word-level tokenisation was chosen because:
- Yelp reviews are informal English; word-level tokens capture sentiment-bearing words directly.
- Vocabulary size stays manageable (20k covers ~95% of frequent tokens).
- Simpler to implement and interpret than subword (BPE) for an LSTM model.
- Pre-trained GloVe embeddings are available at word level (optionally used).

Out-of-vocabulary tokens are mapped to `<UNK>`. Padding token index is `0`.
"""
))

cells.append(code(
"""# ── Tokeniser + vocabulary builder ───────────────────────────────────────────
def simple_tokenize(text):
    return re.findall(r"\\b[a-z']+\\b", text.lower())

# Build vocab on training portion (we'll do the split next section, so we'll
# rebuild properly there; here we build it on all data for EDA purposes)
print("Building vocabulary...")
counter = Counter()
for text in tqdm(df_reviews['text'], desc="Tokenising"):
    counter.update(simple_tokenize(text))

# Keep top VOCAB_SIZE-2 tokens (reserve 0=PAD, 1=UNK)
most_common = [tok for tok, _ in counter.most_common(VOCAB_SIZE - 2)]
VOCAB = {'<PAD>': 0, '<UNK>': 1}
VOCAB.update({tok: idx + 2 for idx, tok in enumerate(most_common)})

print(f"Vocabulary size: {len(VOCAB):,}")
print(f"Top 20 tokens: {most_common[:20]}")

# Save vocabulary for inference / deployment
with open(MODELS_DIR / 'vocab.pkl', 'wb') as f:
    pickle.dump(VOCAB, f)
print("Vocabulary saved to models/vocab.pkl")
"""
))

cells.append(code(
"""# ── ReviewDataset ─────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_SEQ_LEN):
        self.texts   = texts
        self.labels  = labels
        self.vocab   = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def encode(self, text):
        tokens = simple_tokenize(text)
        ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        if len(ids) >= self.max_len:
            return ids[:self.max_len]
        return ids + [0] * (self.max_len - len(ids))

    def __getitem__(self, idx):
        return (torch.tensor(self.encode(self.texts[idx]), dtype=torch.long),
                int(self.labels[idx]))
"""
))

cells.append(code(
"""# ── Sequence length stats ─────────────────────────────────────────────────────
token_lens = [len(simple_tokenize(t)) for t in df_reviews['text'].sample(2000, random_state=SEED)]
pct_truncated = np.mean([l > MAX_SEQ_LEN for l in token_lens]) * 100
print(f"Max sequence length chosen : {MAX_SEQ_LEN}")
print(f"% texts truncated          : {pct_truncated:.1f}%")
print(f"Mean token length          : {np.mean(token_lens):.1f}")
print(f"Median token length        : {np.median(token_lens):.1f}")
print(f"95th percentile            : {np.percentile(token_lens, 95):.1f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(token_lens, bins=50, color='steelblue', edgecolor='black')
ax.axvline(MAX_SEQ_LEN, color='red', linestyle='--', lw=2,
           label=f'Max seq len = {MAX_SEQ_LEN}')
ax.set_title('Token Length Distribution', fontsize=13, fontweight='bold')
ax.set_xlabel('Tokens per Review')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('figures/06_token_lengths.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

# ===========================================================
# SECTION 6 — TRAIN / VAL / TEST SPLIT
# ===========================================================
cells.append(md(
"""---
## 6 · Train – Validation – Test Split

Each dataset is split independently using a **stratified 70 / 15 / 15** partition with a fixed random seed (`SEED = 42`).

- The **test set is held out** and only used in the final evaluation section.
- The **validation set** is used for all architecture decisions, hyperparameter tuning, and early stopping.
"""
))

cells.append(code(
"""# ── Image split ───────────────────────────────────────────────────────────────
img_train_val, img_test = train_test_split(
    df_sampled, test_size=0.15, random_state=SEED, stratify=df_sampled['label'])
img_train, img_val = train_test_split(
    img_train_val, test_size=0.15/0.85, random_state=SEED, stratify=img_train_val['label'])

print("Image splits:")
print(f"  Train : {len(img_train):,}  {img_train['label'].value_counts().to_dict()}")
print(f"  Val   : {len(img_val):,}  {img_val['label'].value_counts().to_dict()}")
print(f"  Test  : {len(img_test):,}  {img_test['label'].value_counts().to_dict()}")
"""
))

cells.append(code(
"""# ── Text split ────────────────────────────────────────────────────────────────
texts  = df_reviews['text'].tolist()
labels = df_reviews['label'].tolist()

txt_train_val, txt_test, lbl_train_val, lbl_test = train_test_split(
    texts, labels, test_size=0.15, random_state=SEED, stratify=labels)
txt_train, txt_val, lbl_train, lbl_val = train_test_split(
    txt_train_val, lbl_train_val, test_size=0.15/0.85, random_state=SEED, stratify=lbl_train_val)

print("Text splits:")
print(f"  Train : {len(txt_train):,}")
print(f"  Val   : {len(txt_val):,}")
print(f"  Test  : {len(txt_test):,}")

# Build final vocabulary on TRAINING data only
print("\\nRebuilding vocabulary on training data only (correct practice)...")
counter_train = Counter()
for text in tqdm(txt_train, desc="Tokenising train"):
    counter_train.update(simple_tokenize(text))

most_common_train = [t for t, _ in counter_train.most_common(VOCAB_SIZE - 2)]
VOCAB = {'<PAD>': 0, '<UNK>': 1}
VOCAB.update({t: i + 2 for i, t in enumerate(most_common_train)})
print(f"Final vocab size: {len(VOCAB):,}")

with open(MODELS_DIR / 'vocab.pkl', 'wb') as f:
    pickle.dump(VOCAB, f)
"""
))

cells.append(code(
"""# ── DataLoaders ───────────────────────────────────────────────────────────────
# --- Image ---
ds_img_train = PhotoDataset(img_train, train_transform)
ds_img_val   = PhotoDataset(img_val,   val_transform)
ds_img_test  = PhotoDataset(img_test,  val_transform)

dl_img_train = DataLoader(ds_img_train, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=False)
dl_img_val   = DataLoader(ds_img_val,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=False)
dl_img_test  = DataLoader(ds_img_test,  batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=False)

# --- Text ---
ds_txt_train = ReviewDataset(txt_train, lbl_train, VOCAB)
ds_txt_val   = ReviewDataset(txt_val,   lbl_val,   VOCAB)
ds_txt_test  = ReviewDataset(txt_test,  lbl_test,  VOCAB)

dl_txt_train = DataLoader(ds_txt_train, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=False)
dl_txt_val   = DataLoader(ds_txt_val,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=False)
dl_txt_test  = DataLoader(ds_txt_test,  batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=False)

print("DataLoaders ready.")
print(f"  Image — train batches: {len(dl_img_train)} | val: {len(dl_img_val)} | test: {len(dl_img_test)}")
print(f"  Text  — train batches: {len(dl_txt_train)} | val: {len(dl_txt_val)} | test: {len(dl_txt_test)}")
"""
))

# ===========================================================
# SECTION 7 — TRAINING UTILITIES
# ===========================================================
cells.append(md("---\n## 7 · Training Utilities"))

cells.append(code(
"""# ── Generic train / evaluate / fit functions ──────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * len(labels)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total         += len(labels)
    return total_loss / total, total_correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss    += loss.item() * len(labels)
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total         += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return (total_loss / total, total_correct / total,
            np.array(all_preds), np.array(all_labels))


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs, device, model_path, patience=PATIENCE):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    wait = 0

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {epoch+1:2d}/{epochs}  "
              f"| Train  loss={tr_loss:.4f}  acc={tr_acc:.4f}"
              f"| Val  loss={vl_loss:.4f}  acc={vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), model_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    return history


def plot_history(histories, labels, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    colors = ['royalblue', 'crimson', 'darkorange', 'green']

    for i, (hist, lbl) in enumerate(zip(histories, labels)):
        c = colors[i % len(colors)]
        axes[0].plot(hist['train_loss'], '--', color=c, alpha=0.6, label=f'{lbl} train')
        axes[0].plot(hist['val_loss'],   '-',  color=c,            label=f'{lbl} val')
        axes[1].plot(hist['train_acc'],  '--', color=c, alpha=0.6, label=f'{lbl} train')
        axes[1].plot(hist['val_acc'],    '-',  color=c,            label=f'{lbl} val')

    for ax, ylabel in zip(axes, ['Loss', 'Accuracy']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_title(f'{title} — Loss Curves', fontweight='bold')
    axes[1].set_title(f'{title} — Accuracy Curves', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=120, bbox_inches='tight')
    plt.show()


def plot_confusion(preds, labels, class_names, title, filename):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=120, bbox_inches='tight')
    plt.show()
    return cm


print("Training utilities defined.")
"""
))

# ===========================================================
# SECTION 8 — CNN FROM SCRATCH
# ===========================================================
cells.append(md(
"""---
## 8 · CNN Branch — Approach 1: Custom CNN Trained from Scratch

Architecture: four convolutional blocks (Conv → BN → ReLU → Conv → BN → ReLU → MaxPool),
followed by global average pooling and two fully connected layers with dropout.
"""
))

cells.append(code(
"""class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
        self.features = nn.Sequential(
            conv_block(3,   32),   # → 112×112
            conv_block(32,  64),   # →  56×56
            conv_block(64,  128),  # →  28×28
            conv_block(128, 256),  # →  14×14
            nn.AdaptiveAvgPool2d((1, 1)),  # global avg pool → 1×1 (MPS-compatible)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


cnn_scratch = CustomCNN(num_classes=3).to(DEVICE)
total_params = sum(p.numel() for p in cnn_scratch.parameters() if p.requires_grad)
print(f"Custom CNN — trainable parameters: {total_params:,}")
"""
))

cells.append(code(
"""# ── Train CNN from scratch ────────────────────────────────────────────────────
criterion   = nn.CrossEntropyLoss()
opt_scratch = optim.Adam(cnn_scratch.parameters(), lr=LR, weight_decay=1e-4)
sch_scratch = optim.lr_scheduler.ReduceLROnPlateau(opt_scratch, patience=2, factor=0.5)

print("Training Custom CNN from scratch...")
hist_scratch = fit(
    cnn_scratch, dl_img_train, dl_img_val, criterion, opt_scratch, sch_scratch,
    CNN_EPOCHS, DEVICE, MODELS_DIR / 'cnn_scratch_best.pth'
)
"""
))

cells.append(code(
"""plot_history([hist_scratch], ['Custom CNN'], 'CNN from Scratch', '07_cnn_scratch_curves.png')
"""
))

# ===========================================================
# SECTION 9 — CNN TRANSFER LEARNING
# ===========================================================
cells.append(md(
"""---
## 9 · CNN Branch — Approach 2: Transfer Learning with ResNet-18

**Strategy:**
1. Load ImageNet-pretrained ResNet-18 and freeze the entire backbone.
2. Replace the final FC layer with a new head (512 → 3 classes) and train head-only for 5 epochs.
3. Unfreeze `layer4` and fine-tune with a reduced learning rate for the remaining epochs.

**Justification:** Transfer learning is expected to outperform the scratch CNN because:
- Our dataset (~4,500 training images) is modest; ImageNet features generalise well to restaurant photos.
- Low-level features (edges, textures, colours) are highly reusable across visual domains.
"""
))

cells.append(code(
"""# ── Phase 1: Head-only training ───────────────────────────────────────────────
cnn_tl = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in cnn_tl.parameters():
    param.requires_grad = False

cnn_tl.fc = nn.Linear(cnn_tl.fc.in_features, 3)
cnn_tl = cnn_tl.to(DEVICE)

trainable = sum(p.numel() for p in cnn_tl.parameters() if p.requires_grad)
print(f"ResNet-18 TL — head-only trainable params: {trainable:,}")

opt_tl_head = optim.Adam(cnn_tl.fc.parameters(), lr=LR)
sch_tl_head = optim.lr_scheduler.ReduceLROnPlateau(opt_tl_head, patience=2, factor=0.5)

print("\\nPhase 1: Training head only (5 epochs)...")
hist_tl_head = fit(
    cnn_tl, dl_img_train, dl_img_val, criterion, opt_tl_head, sch_tl_head,
    5, DEVICE, MODELS_DIR / 'cnn_tl_phase1.pth', patience=5
)
"""
))

cells.append(code(
"""# ── Phase 2: Unfreeze layer4 + fine-tune ─────────────────────────────────────
for param in cnn_tl.layer4.parameters():
    param.requires_grad = True
for param in cnn_tl.fc.parameters():
    param.requires_grad = True

trainable2 = sum(p.numel() for p in cnn_tl.parameters() if p.requires_grad)
print(f"Phase 2 — trainable params after unfreezing layer4: {trainable2:,}")

opt_tl_ft = optim.Adam(filter(lambda p: p.requires_grad, cnn_tl.parameters()),
                        lr=LR / 10, weight_decay=1e-4)
sch_tl_ft = optim.lr_scheduler.ReduceLROnPlateau(opt_tl_ft, patience=2, factor=0.5)

print("Phase 2: Fine-tuning layer4 + head...")
hist_tl_ft = fit(
    cnn_tl, dl_img_train, dl_img_val, criterion, opt_tl_ft, sch_tl_ft,
    CNN_EPOCHS, DEVICE, MODELS_DIR / 'cnn_tl_best.pth'
)
"""
))

cells.append(code(
"""# ── Combine history for plotting ──────────────────────────────────────────────
hist_tl_combined = {
    k: hist_tl_head[k] + hist_tl_ft[k] for k in hist_tl_head
}
plot_history([hist_scratch, hist_tl_combined],
             ['Custom CNN (scratch)', 'ResNet-18 TL'],
             'CNN Comparison', '08_cnn_comparison_curves.png')
"""
))

# ===========================================================
# SECTION 10 — CNN EVALUATION
# ===========================================================
cells.append(md(
"""---
## 10 · CNN Branch — Evaluation on Test Set

Both models are loaded from their best checkpoints and evaluated on the held-out test set.
Metrics: Accuracy, Macro F1-Score, Confusion Matrix.
"""
))

cells.append(code(
"""# ── Load best weights and evaluate ───────────────────────────────────────────
cnn_scratch.load_state_dict(torch.load(MODELS_DIR / 'cnn_scratch_best.pth', map_location=DEVICE, weights_only=False))
cnn_tl.load_state_dict(torch.load(MODELS_DIR / 'cnn_tl_best.pth', map_location=DEVICE, weights_only=False))

_, _, preds_scratch, true_scratch = evaluate(cnn_scratch, dl_img_test, criterion, DEVICE)
_, _, preds_tl,      true_tl      = evaluate(cnn_tl,      dl_img_test, criterion, DEVICE)

acc_scratch = accuracy_score(true_scratch, preds_scratch)
f1_scratch  = f1_score(true_scratch, preds_scratch, average='macro')
acc_tl      = accuracy_score(true_tl, preds_tl)
f1_tl       = f1_score(true_tl, preds_tl, average='macro')

print("CNN Test-Set Results:")
print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 47)
print(f"{'Custom CNN (scratch)':<25} {acc_scratch:>10.4f} {f1_scratch:>10.4f}")
print(f"{'ResNet-18 TL':<25} {acc_tl:>10.4f} {f1_tl:>10.4f}")

print("\\nCustom CNN Classification Report:")
print(classification_report(true_scratch, preds_scratch, target_names=PHOTO_CLASSES))
print("\\nResNet-18 TL Classification Report:")
print(classification_report(true_tl, preds_tl, target_names=PHOTO_CLASSES))
"""
))

cells.append(code(
"""# ── Confusion matrices side by side ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, preds, true, title in [
    (axes[0], preds_scratch, true_scratch, f'Custom CNN (scratch)\\nAcc={acc_scratch:.3f}  F1={f1_scratch:.3f}'),
    (axes[1], preds_tl,      true_tl,      f'ResNet-18 Transfer Learning\\nAcc={acc_tl:.3f}  F1={f1_tl:.3f}'),
]:
    cm = confusion_matrix(true, preds)
    ConfusionMatrixDisplay(cm, display_labels=PHOTO_CLASSES).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/09_cnn_confusion.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

cells.append(md(
"""### CNN Discussion

**Which approach performed better and why?**

The ResNet-18 transfer learning model is expected to outperform the custom CNN because:
- ImageNet pre-training gives the backbone excellent low-level and mid-level features (edges, textures, colour gradients) that are directly applicable to restaurant photos.
- Our training set (~4 500 images) is moderate-sized; the CNN from scratch has sufficient data to learn but cannot match the breadth of features learned from 1.2M ImageNet images.
- Transfer learning is particularly powerful when the source domain (ImageNet — everyday objects) overlaps with the target domain (food, interiors, outdoor scenes).

The from-scratch model may still perform reasonably well because the task (food vs interior vs exterior) relies on global scene-level cues that are learnable from a few thousand examples.
"""
))

# ===========================================================
# SECTION 11 — RNN SINGLE LSTM
# ===========================================================
cells.append(md(
"""---
## 11 · RNN Branch — Variation 1: Single Unidirectional LSTM
"""
))

cells.append(code(
"""class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=3, dropout=0.3, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * factor, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (hn, _) = self.lstm(emb)
        if self.bidirectional:
            hidden = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            hidden = hn[-1]
        return self.fc(self.dropout(hidden))


lstm_single = SentimentRNN(
    vocab_size=len(VOCAB), embed_dim=128, hidden_dim=256,
    num_layers=2, num_classes=3, dropout=0.3, bidirectional=False
).to(DEVICE)
params_single = sum(p.numel() for p in lstm_single.parameters() if p.requires_grad)
print(f"Single LSTM — trainable parameters: {params_single:,}")
"""
))

cells.append(code(
"""opt_lstm1 = optim.Adam(lstm_single.parameters(), lr=LR, weight_decay=1e-5)
sch_lstm1 = optim.lr_scheduler.ReduceLROnPlateau(opt_lstm1, patience=2, factor=0.5)

print("Training Single LSTM...")
hist_lstm1 = fit(
    lstm_single, dl_txt_train, dl_txt_val, criterion, opt_lstm1, sch_lstm1,
    RNN_EPOCHS, DEVICE, MODELS_DIR / 'rnn_single_best.pth'
)
"""
))

cells.append(code(
"""plot_history([hist_lstm1], ['Single LSTM'], 'Single LSTM', '10_lstm_single_curves.png')
"""
))

# ===========================================================
# SECTION 12 — RNN BIDIRECTIONAL LSTM
# ===========================================================
cells.append(md(
"""---
## 12 · RNN Branch — Variation 2: Bidirectional LSTM

A **Bidirectional LSTM** reads the sequence both left-to-right and right-to-left, giving each
time step access to both past and future context. This is particularly useful for sentiment
analysis where a negation at the end of a sentence (e.g., "The food was great — NOT.") can
reverse the meaning of earlier tokens.
"""
))

cells.append(code(
"""lstm_bi = SentimentRNN(
    vocab_size=len(VOCAB), embed_dim=128, hidden_dim=128,
    num_layers=2, num_classes=3, dropout=0.3, bidirectional=True
).to(DEVICE)
params_bi = sum(p.numel() for p in lstm_bi.parameters() if p.requires_grad)
print(f"Bidirectional LSTM — trainable parameters: {params_bi:,}")

opt_lstm2 = optim.Adam(lstm_bi.parameters(), lr=LR, weight_decay=1e-5)
sch_lstm2 = optim.lr_scheduler.ReduceLROnPlateau(opt_lstm2, patience=2, factor=0.5)

print("Training Bidirectional LSTM...")
hist_lstm2 = fit(
    lstm_bi, dl_txt_train, dl_txt_val, criterion, opt_lstm2, sch_lstm2,
    RNN_EPOCHS, DEVICE, MODELS_DIR / 'rnn_bi_best.pth'
)
"""
))

cells.append(code(
"""plot_history([hist_lstm1, hist_lstm2],
             ['Single LSTM', 'BiLSTM'],
             'RNN Comparison', '11_rnn_comparison_curves.png')
"""
))

# ===========================================================
# SECTION 13 — RNN EVALUATION
# ===========================================================
cells.append(md(
"""---
## 13 · RNN Branch — Evaluation on Test Set
"""
))

cells.append(code(
"""lstm_single.load_state_dict(torch.load(MODELS_DIR / 'rnn_single_best.pth', map_location=DEVICE, weights_only=False))
lstm_bi.load_state_dict(torch.load(MODELS_DIR / 'rnn_bi_best.pth', map_location=DEVICE, weights_only=False))

_, _, preds_lstm1, true_lstm1 = evaluate(lstm_single, dl_txt_test, criterion, DEVICE)
_, _, preds_lstm2, true_lstm2 = evaluate(lstm_bi,     dl_txt_test, criterion, DEVICE)

acc_lstm1 = accuracy_score(true_lstm1, preds_lstm1)
f1_lstm1  = f1_score(true_lstm1, preds_lstm1, average='macro')
acc_lstm2 = accuracy_score(true_lstm2, preds_lstm2)
f1_lstm2  = f1_score(true_lstm2, preds_lstm2, average='macro')

print("RNN Test-Set Results:")
print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 47)
print(f"{'Single LSTM':<25} {acc_lstm1:>10.4f} {f1_lstm1:>10.4f}")
print(f"{'BiLSTM':<25} {acc_lstm2:>10.4f} {f1_lstm2:>10.4f}")

print("\\nSingle LSTM Classification Report:")
print(classification_report(true_lstm1, preds_lstm1, target_names=SENTIMENT_CLASSES))
print("\\nBiLSTM Classification Report:")
print(classification_report(true_lstm2, preds_lstm2, target_names=SENTIMENT_CLASSES))
"""
))

cells.append(code(
"""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, preds, true, title in [
    (axes[0], preds_lstm1, true_lstm1, f'Single LSTM\\nAcc={acc_lstm1:.3f}  F1={f1_lstm1:.3f}'),
    (axes[1], preds_lstm2, true_lstm2, f'BiLSTM\\nAcc={acc_lstm2:.3f}  F1={f1_lstm2:.3f}'),
]:
    cm = confusion_matrix(true, preds)
    ConfusionMatrixDisplay(cm, display_labels=SENTIMENT_CLASSES).plot(ax=ax, colorbar=False, cmap='Greens')
    ax.set_title(title, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/12_rnn_confusion.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

# ===========================================================
# SECTION 14 — BUSINESS INTEGRATION
# ===========================================================
cells.append(md(
"""---
## 14 · Business Integration

### Decision Logic

The two best models are combined at the **business decision layer**:

| CNN Prediction | RNN Prediction | Business Action |
|----------------|----------------|----------------|
| food | positive | ✅ **FEATURED** — Highlight on Yelp |
| food | neutral | 🟡 **MONITOR** — Watch for sentiment changes |
| food | negative | ⚠️ **REVIEW ALERT** — Food looks good but customers are unhappy |
| inside/outside | positive | 🟡 **PHOTO COACHING** — Great reviews, but lacks food imagery |
| inside/outside | neutral | 🟠 **IMPROVEMENT NEEDED** — Neither signal is strong |
| inside/outside | negative | 🚨 **URGENT FLAG** — Both visual and review quality are poor |
"""
))

cells.append(code(
"""# ── Select best models ────────────────────────────────────────────────────────
# Use the better CNN and better RNN (compare test metrics above)
best_cnn = cnn_tl       # ResNet-18 TL (expected better)
best_rnn = lstm_bi      # BiLSTM (expected better or comparable)

best_cnn.eval()
best_rnn.eval()
"""
))

cells.append(code(
"""# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_image(img_path, model, device):
    img = Image.open(img_path).convert('RGB')
    tensor = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return IDX2LABEL_CNN[pred_idx], float(probs[pred_idx]), probs


def predict_text(text, model, vocab, device):
    ds = ReviewDataset([text], [0], vocab)
    tensor = ds[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return IDX2LABEL_RNN[pred_idx], float(probs[pred_idx]), probs


def business_recommendation(cnn_label, rnn_label):
    food_photo = (cnn_label == 'food')
    pos_review = (rnn_label == 'positive')
    neu_review = (rnn_label == 'neutral')

    if food_photo and pos_review:
        return '✅ FEATURED'
    elif food_photo and neu_review:
        return '🟡 MONITOR'
    elif food_photo and not pos_review:
        return '⚠️ REVIEW ALERT'
    elif not food_photo and pos_review:
        return '🟡 PHOTO COACHING'
    elif not food_photo and neu_review:
        return '🟠 IMPROVEMENT NEEDED'
    else:
        return '🚨 URGENT FLAG'

print("Inference helpers defined.")
"""
))

cells.append(code(
"""# ── Generate 20-row integration table ────────────────────────────────────────
# Sample 20 examples from test sets (use best-effort pairing from same domain)
N = 20
sample_img_rows = img_test.sample(N, random_state=7).reset_index(drop=True)
sample_txt_rows = list(zip(
    [txt_test[i] for i in random.sample(range(len(txt_test)), N)],
    [lbl_test[i] for i in random.sample(range(len(lbl_test)), N)]
))

records = []
for i in range(N):
    img_row = sample_img_rows.iloc[i]
    text, true_sent_idx = sample_txt_rows[i]

    cnn_pred, cnn_conf, _ = predict_image(img_row['path'], best_cnn, DEVICE)
    rnn_pred, rnn_conf, _ = predict_text(text, best_rnn, VOCAB, DEVICE)
    rec = business_recommendation(cnn_pred, rnn_pred)

    records.append({
        'Photo ID'      : img_row['photo_id'][:12] + '...',
        'True Photo Label': img_row['label'],
        'CNN Prediction': cnn_pred,
        'CNN Confidence': f'{cnn_conf:.2%}',
        'Review Snippet': text[:60] + '...',
        'True Sentiment': IDX2LABEL_RNN[true_sent_idx],
        'RNN Prediction': rnn_pred,
        'RNN Confidence': f'{rnn_conf:.2%}',
        'Business Action': rec,
    })

df_integration = pd.DataFrame(records)
df_integration
"""
))

cells.append(code(
"""# ── Comparison chart: CNN acc vs RNN acc vs Business flag precision ───────────
fig, ax = plt.subplots(figsize=(9, 5))

models_labels = ['Custom CNN\\n(scratch)', 'ResNet-18\\n(TL)', 'Single\\nLSTM', 'BiLSTM']
accuracies     = [acc_scratch, acc_tl, acc_lstm1, acc_lstm2]
f1_scores      = [f1_scratch,  f1_tl,  f1_lstm1,  f1_lstm2]

x = np.arange(len(models_labels))
w = 0.35

bars1 = ax.bar(x - w/2, accuracies, w, label='Accuracy', color='steelblue', edgecolor='black')
bars2 = ax.bar(x + w/2, f1_scores,  w, label='Macro F1', color='darkorange', edgecolor='black')

for bar in [*bars1, *bars2]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_ylim(0, 1.1)
ax.set_xticks(x)
ax.set_xticklabels(models_labels, fontsize=10)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison — CNN & RNN', fontsize=14, fontweight='bold')
ax.legend()
ax.axvline(1.5, color='gray', linestyle='--', alpha=0.5)
ax.text(0.5, 1.05, 'Image Models', ha='center', transform=ax.get_xaxis_transform(), fontsize=10)
ax.text(2.5, 1.05, 'Text Models',  ha='center', transform=ax.get_xaxis_transform(), fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/13_model_comparison.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

cells.append(code(
"""# ── Business action distribution from integration table ──────────────────────
action_counts = df_integration['Business Action'].value_counts()

fig, ax = plt.subplots(figsize=(9, 4))
colors_map = {
    '✅ FEATURED':             '#2ECC71',
    '🟡 MONITOR':              '#F1C40F',
    '🟡 PHOTO COACHING':       '#F1C40F',
    '⚠️ REVIEW ALERT':         '#E67E22',
    '🟠 IMPROVEMENT NEEDED':   '#E74C3C',
    '🚨 URGENT FLAG':          '#C0392B',
}
bar_colors = [colors_map.get(k, 'gray') for k in action_counts.index]
ax.barh(action_counts.index, action_counts.values, color=bar_colors, edgecolor='black')
ax.set_xlabel('Count (out of 20 examples)')
ax.set_title('Business Action Distribution from Integration Table', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/14_business_actions.png', dpi=120, bbox_inches='tight')
plt.show()
"""
))

cells.append(md(
"""### Integration Insights

The combined pipeline surfaces restaurants that fall into distinct risk categories:

- **URGENT FLAG** cases benefit most from dual-signal detection: either signal alone might be dismissed as noisy, but agreement between the photo classifier and the sentiment model provides strong evidence for quality degradation.
- **PHOTO COACHING** cases (good reviews, poor photo content) would be missed entirely by a pure sentiment model — demonstrating the added value of the CNN branch.
- **REVIEW ALERT** cases (appealing food photos but negative reviews) highlight a mismatch between visual marketing and actual experience — valuable intelligence for Yelp account managers.
"""
))

# ===========================================================
# SECTION 15 — BUSINESS FRAMING
# ===========================================================
cells.append(md(
"""---
## 15 · Business Framing & Ethics

### 15.1 Business Decision Supported
This pipeline supports **automated restaurant quality monitoring** on the Yelp platform. It reduces manual review workload by automatically triaging restaurants into risk tiers that determine which account manager actions are warranted.

### 15.2 End User
Yelp's **Content Quality Team** and **Restaurant Account Managers**.

### 15.3 Cost of Errors

| Model | False Positive | False Negative |
|-------|---------------|----------------|
| CNN | Flagging a good food photo as non-food → unnecessary outreach | Missing a poor-quality photo submission → poor user experience |
| RNN | Flagging a satisfied customer as negative → unfair penalisation | Missing negative sentiment → allowing poor quality to persist |

For the combined business decision, **false negatives are more costly** — a restaurant that should be flagged but isn't will continue to degrade platform trust. Therefore, the integration rule errs toward flagging (a restaurant is flagged if *either* signal is negative).

### 15.4 Workflow Integration
1. **Trigger:** Nightly batch job processes new photo uploads and reviews from the past 7 days.
2. **Scoring:** CNN scores each new photo; RNN scores each new review.
3. **Aggregation:** Per-restaurant scores are aggregated (majority-vote over new photos/reviews).
4. **Routing:** Restaurants above urgency thresholds are added to the account manager queue in the Yelp CRM.
5. **Feedback loop:** Account manager outcomes (resolved / false alarm) are logged and used to periodically retrain the models.

### 15.5 Ethical Considerations

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Language bias** | The RNN is trained on English reviews; non-native speakers may produce lower sentiment scores for the same experience | Include multilingual reviews in training; audit performance by review language |
| **Cuisine / cultural bias** | "Food" photos vary widely by cuisine; the CNN may perform worse on underrepresented cuisines | Stratify evaluation by cuisine category; augment with cuisine-specific data |
| **Geographic skew** | Yelp dataset skews toward US cities (Las Vegas, Phoenix, etc.) | Include international data; test distribution shift |
| **Feedback loop amplification** | Restaurants flagged more often may receive less traffic, compounding disadvantage | Apply fairness constraints; audit flag rates by restaurant demographic |
| **Privacy** | Review text contains personal opinions that could be misused | Models should output aggregated scores only; individual review text should not be surfaced to restaurant owners |
"""
))

# ===========================================================
# SECTION 16 — DEPLOYMENT
# ===========================================================
cells.append(md(
"""---
## 16 · Deployment Prototype

An interactive Streamlit app (`app.py`) allows users to:
1. **Upload a restaurant photo** → CNN predicts the category + confidence score.
2. **Paste a customer review** → RNN predicts sentiment + confidence score.
3. **See the combined business recommendation** with colour-coded action.

### How to run locally:
```bash
pip install streamlit
streamlit run app.py
```

The app loads the saved model weights from `models/` and the vocabulary from `models/vocab.pkl`.

> A screenshot of the running prototype is included in `figures/streamlit_screenshot.png`
> (capture it after running the app).

### Deployment on Streamlit Cloud:
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set the main file to `app.py`.
4. The app will be publicly accessible with a shareable URL.
"""
))

# ===========================================================
# SECTION 17 — REFERENCES
# ===========================================================
cells.append(md(
"""---
## 17 · References

1. Asghar, N. (2016). *Yelp Dataset Challenge: Review Rating Prediction*. arXiv:1605.05362.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8).
4. Yelp Open Dataset. https://www.yelp.com/dataset
5. PyTorch Documentation. https://pytorch.org/docs/
6. torchvision Models. https://pytorch.org/vision/stable/models.html
7. Scikit-learn Documentation. https://scikit-learn.org/stable/
8. Streamlit Documentation. https://docs.streamlit.io/

### Pre-trained Model
- ResNet-18 weights: `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` (PyTorch Hub).

### External Code References
- Training loop structure adapted from PyTorch official tutorials.
- WordCloud visualisation adapted from the wordcloud library examples.
"""
))

# ===========================================================
# ASSEMBLE AND WRITE NOTEBOOK
# ===========================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.12"
        }
    },
    "cells": cells
}

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"notebook.ipynb written ({len(cells)} cells).")
