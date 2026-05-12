"""
Streamlit Deployment Prototype
Yelp Restaurant Intelligence: CNN Photo Classifier + LSTM Sentiment Analyzer

Run: streamlit run app.py
"""
import pickle, re
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR     = Path("models")
DEMO_PHOTOS_DIR = Path("demo_photos")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

# ── Label mappings ────────────────────────────────────────────────────────────
PHOTO_CLASSES     = ["food", "inside", "outside"]
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]
IDX2CNN           = {i: l for i, l in enumerate(PHOTO_CLASSES)}
IDX2RNN           = {i: l for i, l in enumerate(SENTIMENT_CLASSES)}

IMG_SIZE    = 224
MAX_SEQ_LEN = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Demo examples ─────────────────────────────────────────────────────────────
DEMO_EXAMPLES = [
    {
        "label": "🍕 Happy diner — food photo + glowing review",
        "photo": "food_pve7D6NU.jpg",
        "review": (
            "Absolutely phenomenal experience! The pasta was cooked to perfection — "
            "al dente with a rich, velvety sauce that had clearly been simmering for hours. "
            "The presentation was stunning and portions were generous. Our server was attentive "
            "and knowledgeable about the wine pairings. Will definitely be back and bringing friends!"
        ),
    },
    {
        "label": "🍔 Disappointed customer — food photo + negative review",
        "photo": "food_H52Er-uB.jpg",
        "review": (
            "Terrible experience from start to finish. The burger arrived cold and soggy, "
            "clearly sitting under a heat lamp for too long. The fries were limp and unsalted. "
            "We waited 45 minutes for our food only to be served the wrong order. "
            "The manager was dismissive when we complained. Save your money and go elsewhere."
        ),
    },
    {
        "label": "🪑 Interior shot — inside photo + positive review",
        "photo": "inside_zsvj7vlo.jpg",
        "review": (
            "What a charming place! The interior has a warm, rustic vibe with exposed brick walls "
            "and soft lighting that made for a really romantic atmosphere. Staff were friendly and "
            "the cocktail menu was creative. We enjoyed a lovely evening here and the noise level "
            "was just right for a conversation. Highly recommend for date night."
        ),
    },
    {
        "label": "🏢 Interior shot — inside photo + mixed review",
        "photo": "inside_QRUgAISg.jpg",
        "review": (
            "Decent place overall. The decor is nice and the location is convenient. "
            "Food was okay — nothing that blew me away but nothing bad either. "
            "Service was a bit slow on a Friday evening but the staff were polite. "
            "Prices are fair for the area. Might come back if I'm nearby but wouldn't "
            "go out of my way."
        ),
    },
    {
        "label": "🏙️ Exterior shot — outside photo + negative review",
        "photo": "outside_HCUdRJHH.jpg",
        "review": (
            "Walked past this place many times and finally decided to try it — big mistake. "
            "The outside looks inviting but inside it was cramped and understaffed. "
            "We were seated next to the kitchen which was incredibly loud and hot. "
            "The menu had barely changed since it opened years ago. Nothing memorable "
            "and overpriced for what you get. Not returning."
        ),
    },
]


# ── Model definitions (must match notebook exactly) ───────────────────────────
class CustomCNN(nn.Module):
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
            conv_block(3, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=2, num_classes=3, dropout=0.3, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * factor, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (hn, _) = self.lstm(emb)
        hidden = torch.cat([hn[-2], hn[-1]], dim=1) if self.bidirectional else hn[-1]
        return self.fc(self.dropout(hidden))


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_vocab():
    path = MODELS_DIR / "vocab.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_cnn():
    path = MODELS_DIR / "cnn_tl_best.pth"
    if not path.exists():
        path = MODELS_DIR / "cnn_scratch_best.pth"
    if not path.exists():
        return None
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    except Exception:
        model = CustomCNN(num_classes=3)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_rnn(vocab_size):
    path = MODELS_DIR / "rnn_bi_best.pth"
    if not path.exists():
        path = MODELS_DIR / "rnn_single_best.pth"
    if not path.exists():
        return None
    try:
        model = SentimentRNN(vocab_size=vocab_size, bidirectional=True)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    except Exception:
        model = SentimentRNN(vocab_size=vocab_size, bidirectional=False, hidden_dim=256)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE).eval()
    return model


# ── Inference helpers ──────────────────────────────────────────────────────────
_img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def tokenize(text):
    return re.findall(r"\b[a-z']+\b", text.lower())


def encode_text(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(ids) >= MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN]
    else:
        ids += [0] * (MAX_SEQ_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)


def predict_image(img, model):
    tensor = _img_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    return IDX2CNN[idx], float(probs[idx]), probs


def predict_text(text, model, vocab):
    tensor = encode_text(text, vocab)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    return IDX2RNN[idx], float(probs[idx]), probs


def business_recommendation(cnn_label, rnn_label):
    food_photo = (cnn_label == "food")
    pos_review = (rnn_label == "positive")
    neu_review = (rnn_label == "neutral")

    if food_photo and pos_review:
        return "✅ FEATURED", "green", "Both signals positive — highlight this restaurant."
    elif food_photo and neu_review:
        return "🟡 MONITOR", "orange", "Good photos, mixed reviews — watch for trends."
    elif food_photo and not pos_review:
        return "⚠️ REVIEW ALERT", "orange", "Great photos but negative sentiment — investigate service/quality."
    elif not food_photo and pos_review:
        return "🟡 PHOTO COACHING", "orange", "Customers love it — help them showcase food imagery."
    elif not food_photo and neu_review:
        return "🟠 IMPROVEMENT NEEDED", "red", "Neither signal strong — recommend overall improvement."
    else:
        return "🚨 URGENT FLAG", "red", "Both signals negative — escalate to account manager immediately."


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yelp Restaurant Intelligence",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Yelp Restaurant Intelligence")
st.markdown(
    "**Advanced ML Final Project** — Upload a restaurant photo and paste a review "
    "to get an automated quality assessment powered by CNN + LSTM deep learning models."
)
st.divider()

# Load models
vocab   = load_vocab()
cnn_mdl = load_cnn()

if vocab is None or cnn_mdl is None:
    st.error(
        "Model weights not found in `models/`. "
        "Please run the notebook first to train and save the models."
    )
    st.stop()

rnn_mdl = load_rnn(len(vocab))

if rnn_mdl is None:
    st.error("RNN weights not found. Please complete model training in the notebook.")
    st.stop()

# ── Demo selector ─────────────────────────────────────────────────────────────
with st.expander("🎲 Try a demo example", expanded=False):
    demo_options = ["— select a scenario —"] + [d["label"] for d in DEMO_EXAMPLES]
    chosen = st.selectbox("Pick a pre-loaded scenario:", demo_options, key="demo_select")

    demo_img  = None
    demo_text = ""

    if chosen != "— select a scenario —":
        demo = next(d for d in DEMO_EXAMPLES if d["label"] == chosen)
        photo_path = DEMO_PHOTOS_DIR / demo["photo"]

        if photo_path.exists():
            demo_img  = Image.open(photo_path).convert("RGB")
            demo_text = demo["review"]

            col_prev, col_rev = st.columns([1, 2])
            with col_prev:
                st.image(demo_img, caption="Demo photo", use_container_width=True)
            with col_rev:
                st.markdown("**Review text:**")
                st.info(demo["review"])

            if st.button("⚡ Run this demo", key="run_demo"):
                st.session_state["demo_img"]  = demo_img
                st.session_state["demo_text"] = demo_text
        else:
            st.warning(f"Demo photo not found: {photo_path}")

# Resolve active image / text (demo takes precedence over manual upload until cleared)
active_img  = st.session_state.get("demo_img")
active_text = st.session_state.get("demo_text", "")

# ── Main two-column input ─────────────────────────────────────────────────────
st.markdown("#### Or upload your own photo and review:")
col_img, col_txt = st.columns(2)

# ── Image side ────────────────────────────────────────────────────────────────
with col_img:
    st.subheader("📷 Photo Analysis (CNN)")
    uploaded = st.file_uploader(
        "Upload a restaurant photo", type=["jpg", "jpeg", "png"], key="photo"
    )
    if uploaded:
        active_img = Image.open(uploaded).convert("RGB")
        st.session_state.pop("demo_img", None)  # manual upload overrides demo

    cnn_label = cnn_conf = None
    if active_img is not None:
        st.image(active_img, caption="Photo", use_container_width=True)
        cnn_label, cnn_conf, cnn_probs = predict_image(active_img, cnn_mdl)
        st.metric("Predicted Category", cnn_label.upper(), f"{cnn_conf:.1%} confidence")
        st.bar_chart(
            {"Confidence": {lbl: float(cnn_probs[i]) for i, lbl in IDX2CNN.items()}}
        )

# ── Text side ─────────────────────────────────────────────────────────────────
with col_txt:
    st.subheader("💬 Review Analysis (LSTM)")
    review_text = st.text_area(
        "Paste a customer review",
        value=active_text,
        height=180,
        placeholder="e.g. The food was absolutely delicious and the service was outstanding...",
        key="review",
    )
    rnn_label = rnn_conf = None
    if review_text.strip():
        rnn_label, rnn_conf, rnn_probs = predict_text(review_text, rnn_mdl, vocab)

        sent_emoji = {"negative": "😠", "neutral": "😐", "positive": "😊"}
        st.metric(
            "Predicted Sentiment",
            f"{sent_emoji.get(rnn_label, '')} {rnn_label.upper()}",
            f"{rnn_conf:.1%} confidence"
        )
        st.bar_chart(
            {"Confidence": {lbl: float(rnn_probs[i]) for i, lbl in IDX2RNN.items()}}
        )

# ── Combined recommendation ───────────────────────────────────────────────────
st.divider()
st.subheader("🏢 Combined Business Recommendation")

if cnn_label and rnn_label:
    action, color, explanation = business_recommendation(cnn_label, rnn_label)

    st.markdown(
        f"""
        <div style="background-color: {'#d4edda' if color=='green' else '#fff3cd' if color=='orange' else '#f8d7da'};
                    border-radius: 10px; padding: 20px; text-align: center;">
            <h2 style="margin: 0; color: {'#155724' if color=='green' else '#856404' if color=='orange' else '#721c24'};">
                {action}
            </h2>
            <p style="margin: 8px 0 0 0; font-size: 1.1em; color: #333;">{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.info(f"**CNN:** Photo classified as **{cnn_label}** ({cnn_conf:.1%} confidence)")
    with detail_col2:
        st.info(f"**RNN:** Review sentiment is **{rnn_label}** ({rnn_conf:.1%} confidence)")
else:
    st.info("Upload a photo **and** enter a review above to see the combined recommendation.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Advanced Machine Learning Final Project · Yelp Open Dataset · "
    "CNN (ResNet-18 Transfer Learning) + BiLSTM · "
    f"Running on: {DEVICE}"
)
