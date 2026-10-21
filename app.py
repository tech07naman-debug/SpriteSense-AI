
# app.py — SpriteSense AI: Retro Sprite Family Classifier
# Run with: streamlit run app.py

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="SpriteSense AI | Sprite Classifier",
    page_icon="👾",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_sprite_classifier.keras"
CLASSES_PATH = BASE_DIR / "class_names.json"

IMAGE_SIZE = (24, 24)

# Fallback only if class_names.json is unavailable.
# The order must match the order used while training.
FALLBACK_CLASSES = [
    "01_animals",
    "02_fantasy_creatures",
    "04_plants_and_fungi",
    "05_food_and_drink",
    "06_containers_and_storage",
    "09_tools_and_crafting",
    "11_machines_and_technology",
    "14_places_and_structures",
    "16_nature_and_landscapes",
    "18_effects_and_celestial",
    "19_symbols_and_documents",
]

# ---------------------------------------------------------
# 2. NEON ARCADE THEME
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800&family=Rajdhani:wght@400;500;600;700&display=swap');

    :root {
        --bg: #090914;
        --panel: #121226;
        --panel2: #191933;
        --cyan: #41f6ff;
        --purple: #a855f7;
        --pink: #ff4fd8;
        --text: #f4f2ff;
        --muted: #aaa9ca;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, #21103d 0%, transparent 36%),
            radial-gradient(ellipse at 95% 12%, #092c3d 0%, transparent 32%),
            var(--bg);
        color: var(--text);
        font-family: 'Rajdhani', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(9, 9, 20, 0.85);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14132b, #0a0a17);
        border-right: 1px solid #34305a;
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1px;
    }

    p, label, div, span {
        font-family: 'Rajdhani', sans-serif;
    }

    .hero {
        text-align: center;
        padding: 28px 16px 20px;
        margin-bottom: 20px;
        border: 1px solid #39315f;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(168,85,247,0.14),
            rgba(65,246,255,0.06),
            rgba(255,79,216,0.09)
        );
        box-shadow: 0 0 35px rgba(168,85,247,0.10);
    }

    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(30px, 5vw, 54px);
        font-weight: 800;
        line-height: 1.2;
        color: #ffffff;
        text-shadow: 0 0 12px rgba(65,246,255,0.65),
                     0 0 30px rgba(168,85,247,0.5);
    }

    .hero-title span {
        color: var(--cyan);
    }

    .hero-subtitle {
        color: #c7c4e8;
        font-size: 19px;
        margin-top: 12px;
    }

    .eyebrow {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        border: 1px solid #51417f;
        color: var(--cyan);
        background: #14142a;
        font-family: 'Orbitron', sans-serif;
        font-size: 11px;
        letter-spacing: 2px;
        margin-bottom: 15px;
    }

    .section-label {
        font-family: 'Orbitron', sans-serif;
        color: var(--cyan);
        font-size: 19px;
        margin: 10px 0 14px;
    }

    .muted {
        color: var(--muted);
        font-size: 16px;
    }

    .panel {
        padding: 20px;
        border: 1px solid #353257;
        border-radius: 18px;
        background: linear-gradient(145deg, #17172e, #101020);
        margin: 8px 0 18px;
    }

    .prediction-card {
        padding: 25px;
        border: 1px solid rgba(65,246,255,0.55);
        border-radius: 20px;
        background: linear-gradient(135deg, #151d35, #251438);
        box-shadow: 0 0 28px rgba(65,246,255,0.08);
        text-align: center;
        margin: 12px 0 18px;
    }

    .prediction-label {
        color: var(--cyan);
        font-family: 'Orbitron', sans-serif;
        font-size: 12px;
        letter-spacing: 2px;
    }

    .prediction-name {
        color: #ffffff;
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(20px, 3vw, 31px);
        font-weight: 700;
        margin: 12px 0;
        overflow-wrap: anywhere;
    }

    .prediction-score {
        color: var(--pink);
        font-size: 30px;
        font-weight: 700;
    }

    .metric-card {
        background: #15152b;
        border: 1px solid #353257;
        border-radius: 14px;
        padding: 15px;
        text-align: center;
        min-height: 90px;
    }

    .metric-value {
        font-family: 'Orbitron', sans-serif;
        color: var(--cyan);
        font-size: 23px;
        font-weight: 700;
    }

    .metric-caption {
        color: var(--muted);
        font-size: 14px;
        margin-top: 4px;
    }

    .sprite-frame {
        background: #0c0c1c;
        border: 1px solid #37315c;
        border-radius: 16px;
        padding: 12px;
        text-align: center;
    }

    .footer {
        text-align: center;
        color: #777697;
        border-top: 1px solid #282641;
        padding: 20px 0 8px;
        margin-top: 30px;
        font-size: 14px;
    }

    .stButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: 10px;
        border: 1px solid #6350a0;
        background: linear-gradient(110deg, #6d28d9, #4f46e5);
        color: white;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: var(--cyan);
        color: white;
        box-shadow: 0 0 18px rgba(65,246,255,0.24);
        transform: translateY(-1px);
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    button[data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
    }

    [data-testid="stFileUploader"] {
        background: #131328;
        border: 1px dashed #51417f;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stExpander"] {
        background: #121226;
        border: 1px solid #353257;
        border-radius: 12px;
    }

    [data-testid="stMetric"] {
        background: #15152b;
        padding: 12px;
        border: 1px solid #353257;
        border-radius: 12px;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #a855f7, #41f6ff);
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 3. LOAD MODEL AND CLASS NAMES
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_class_names():
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            names = json.load(file)

        # Supports either a list or a dictionary saved as JSON.
        if isinstance(names, dict):
            try:
                names = [
                    names[str(i)] if str(i) in names else names[i]
                    for i in range(len(names))
                ]
            except (KeyError, TypeError):
                names = list(names.values())

        return [str(name) for name in names]

    return FALLBACK_CLASSES


try:
    model = load_model()
    class_names = load_class_names()
    model_error = None
except Exception as exc:
    model = None
    class_names = FALLBACK_CLASSES
    model_error = str(exc)


# ---------------------------------------------------------
# 4. GENERATE BUILT-IN PIXEL ART SPRITES
# ---------------------------------------------------------
# Each sprite is drawn on a small pixel canvas and enlarged
# using nearest-neighbor interpolation for a crisp retro look.

PALETTE = {
    "background": "#101025",
    "white": "#f8f7ff",
    "black": "#19162e",
    "cyan": "#41f6ff",
    "purple": "#a855f7",
    "pink": "#ff4fd8",
    "green": "#5dff9a",
    "dark_green": "#167d59",
    "yellow": "#ffe66d",
    "orange": "#ff9f43",
    "red": "#ff5364",
    "brown": "#a7653f",
    "blue": "#538dff",
    "light_blue": "#a5edff",
    "cream": "#ffe4ad",
}


def new_canvas():
    image = Image.new("RGB", (24, 24), PALETTE["background"])
    return image, ImageDraw.Draw(image)


def rect(draw, x1, y1, x2, y2, color):
    draw.rectangle((x1, y1, x2, y2), fill=color)


def make_critter():
    img, d = new_canvas()
    # Ears and body
    rect(d, 5, 4, 8, 9, PALETTE["orange"])
    rect(d, 15, 4, 18, 9, PALETTE["orange"])
    rect(d, 5, 7, 18, 17, PALETTE["orange"])
    rect(d, 7, 5, 16, 15, PALETTE["yellow"])
    rect(d, 3, 10, 5, 13, PALETTE["orange"])
    rect(d, 18, 10, 20, 13, PALETTE["orange"])
    # Eyes, nose and feet
    rect(d, 8, 9, 9, 11, PALETTE["black"])
    rect(d, 14, 9, 15, 11, PALETTE["black"])
    rect(d, 10, 12, 13, 13, PALETTE["pink"])
    rect(d, 7, 17, 9, 19, PALETTE["cream"])
    rect(d, 14, 17, 16, 19, PALETTE["cream"])
    return img


def make_tree():
    img, d = new_canvas()
    rect(d, 10, 12, 13, 21, PALETTE["brown"])
    rect(d, 7, 7, 16, 15, PALETTE["green"])
    rect(d, 4, 10, 8, 14, PALETTE["dark_green"])
    rect(d, 15, 9, 19, 14, PALETTE["dark_green"])
    rect(d, 8, 4, 15, 9, PALETTE["green"])
    rect(d, 9, 6, 10, 7, PALETTE["yellow"])
    return img


def make_mushroom():
    img, d = new_canvas()
    rect(d, 10, 12, 14, 20, PALETTE["cream"])
    rect(d, 6, 10, 17, 13, PALETTE["cream"])
    rect(d, 5, 7, 18, 10, PALETTE["red"])
    rect(d, 7, 5, 16, 7, PALETTE["red"])
    rect(d, 9, 7, 10, 8, PALETTE["white"])
    rect(d, 14, 6, 15, 7, PALETTE["white"])
    rect(d, 12, 9, 13, 10, PALETTE["white"])
    return img


def make_potion():
    img, d = new_canvas()
    rect(d, 9, 3, 14, 5, PALETTE["yellow"])
    rect(d, 10, 5, 13, 8, PALETTE["cream"])
    rect(d, 7, 8, 16, 18, PALETTE["light_blue"])
    rect(d, 8, 12, 15, 17, PALETTE["purple"])
    rect(d, 9, 10, 10, 11, PALETTE["white"])
    rect(d, 6, 10, 7, 15, PALETTE["light_blue"])
    rect(d, 16, 10, 17, 15, PALETTE["light_blue"])
    rect(d, 9, 18, 14, 19, PALETTE["cyan"])
    return img


def make_berry():
    img, d = new_canvas()
    rect(d, 10, 5, 14, 7, PALETTE["green"])
    rect(d, 8, 7, 16, 10, PALETTE["red"])
    rect(d, 6, 10, 18, 15, PALETTE["red"])
    rect(d, 8, 15, 16, 18, PALETTE["red"])
    rect(d, 10, 18, 14, 20, PALETTE["red"])
    rect(d, 9, 9, 10, 10, PALETTE["white"])
    rect(d, 14, 12, 15, 13, PALETTE["pink"])
    return img


def make_house():
    img, d = new_canvas()
    rect(d, 5, 10, 18, 20, PALETTE["cream"])
    rect(d, 3, 7, 20, 10, PALETTE["red"])
    rect(d, 5, 5, 18, 7, PALETTE["red"])
    rect(d, 8, 3, 16, 5, PALETTE["red"])
    rect(d, 10, 14, 13, 20, PALETTE["brown"])
    rect(d, 6, 12, 8, 14, PALETTE["cyan"])
    rect(d, 15, 12, 17, 14, PALETTE["cyan"])
    return img


def make_star():
    img, d = new_canvas()
    # Pixelated star with a simple block silhouette
    rect(d, 10, 3, 13, 6, PALETTE["yellow"])
    rect(d, 8, 6, 15, 9, PALETTE["yellow"])
    rect(d, 4, 9, 19, 12, PALETTE["yellow"])
    rect(d, 6, 12, 17, 14, PALETTE["yellow"])
    rect(d, 8, 14, 10, 18, PALETTE["yellow"])
    rect(d, 14, 14, 16, 18, PALETTE["yellow"])
    rect(d, 9, 18, 11, 20, PALETTE["yellow"])
    rect(d, 13, 18, 15, 20, PALETTE["yellow"])
    rect(d, 10, 7, 11, 8, PALETTE["white"])
    return img


def make_robot():
    img, d = new_canvas()
    # Antenna and head
    rect(d, 11, 3, 12, 6, PALETTE["cyan"])
    rect(d, 10, 2, 13, 3, PALETTE["pink"])
    rect(d, 6, 6, 17, 14, PALETTE["light_blue"])
    rect(d, 8, 8, 10, 10, PALETTE["purple"])
    rect(d, 13, 8, 15, 10, PALETTE["purple"])
    rect(d, 10, 11, 13, 12, PALETTE["black"])
    # Body and limbs
    rect(d, 8, 15, 15, 19, PALETTE["cyan"])
    rect(d, 5, 15, 7, 18, PALETTE["purple"])
    rect(d, 16, 15, 18, 18, PALETTE["purple"])
    rect(d, 9, 20, 10, 21, PALETTE["pink"])
    rect(d, 13, 20, 14, 21, PALETTE["pink"])
    return img


SPRITES = {
    "Critter": {
        "image": make_critter,
        "description": "A tiny pixel-art animal",
        "hint": "Animals",
    },
    "Tree": {
        "image": make_tree,
        "description": "A leafy green tree",
        "hint": "Plants & fungi",
    },
    "Mushroom": {
        "image": make_mushroom,
        "description": "A classic red-capped mushroom",
        "hint": "Plants & fungi",
    },
    "Potion": {
        "image": make_potion,
        "description": "A fantasy game potion",
        "hint": "Containers & storage",
    },
    "Berry": {
        "image": make_berry,
        "description": "A bright red pixel berry",
        "hint": "Food & drink",
    },
    "House": {
        "image": make_house,
        "description": "A small pixel-art house",
        "hint": "Places & structures",
    },
    "Star": {
        "image": make_star,
        "description": "A glowing collectible star",
        "hint": "Effects & celestial",
    },
    "Robot": {
        "image": make_robot,
        "description": "A little technology bot",
        "hint": "Machines & technology",
    },
}


def upscale_sprite(image, scale=8):
    return image.resize(
        (image.width * scale, image.height * scale),
        Image.Resampling.NEAREST,
    )


# ---------------------------------------------------------
# 5. IMAGE PREPROCESSING AND PREDICTION
# ---------------------------------------------------------
def prepare_image(image):
    """
    Converts an image to RGB, resizes it to 24x24, and
    normalizes pixel values to [0, 1].

    This assumes training images were also divided by 255
    before being passed to the model.
    """
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.NEAREST)

    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def get_prediction(image):
    if model is None:
        raise RuntimeError(
            "Model file not found or failed to load. "
            "Check final_sprite_classifier.keras."
        )

    processed = prepare_image(image)
    output = np.asarray(model.predict(processed, verbose=0))

    # Most multiclass classifiers return shape (1, number_of_classes).
    scores = output[0].astype(float)

    if len(scores) != len(class_names):
        raise ValueError(
            f"The model returned {len(scores)} scores, but "
            f"class_names.json contains {len(class_names)} classes. "
            "Check that the model and class names belong to the same run."
        )

    # If the model outputs logits rather than probabilities,
    # convert them to probabilities. Softmax outputs already sum to ~1.
    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1.0, atol=0.02)
    ):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / exp_scores.sum()

    order = np.argsort(scores)[::-1]

    results = [
        {
            "family": class_names[int(i)],
            "confidence": float(scores[int(i)]),
        }
        for i in order
    ]

    return results


def display_family_name(name):
    """Makes labels such as 01_animals more readable."""
    name = str(name)
    if "_" in name:
        name = name.split("_", 1)[1]
    return name.replace("_", " ").title()


def show_results(image, results):
    if not results:
        st.warning("No predictions are available.")
        return

    best = results[0]
    top_three = results[:3]

    st.markdown('<div class="section-label">⚡ AI ANALYSIS</div>',
                unsafe_allow_html=True)

    left, right = st.columns([0.9, 1.3], gap="large")

    with left:
        st.markdown('<div class="sprite-frame">', unsafe_allow_html=True)
        st.image(
            upscale_sprite(image.convert("RGB")),
            caption="Image processed by SpriteSense AI",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">TOP PREDICTION</div>
                <div class="prediction-name">
                    {display_family_name(best["family"])}
                </div>
                <div class="prediction-score">
                    {best["confidence"] * 100:.2f}%
                </div>
                <div class="muted">Model confidence score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                "Predicted family",
                display_family_name(best["family"]),
            )
        with m2:
            st.metric(
                "Classes compared",
                len(results),
            )

    st.markdown('<div class="section-label">🏆 TOP 3 GUESSES</div>',
                unsafe_allow_html=True)

    for rank, result in enumerate(top_three, start=1):
        col_rank, col_name, col_score = st.columns([0.12, 0.58, 0.30])

        with col_rank:
            st.markdown(f"**#{rank}**")

        with col_name:
            st.write(display_family_name(result["family"]))
            st.progress(
                min(max(result["confidence"], 0.0), 1.0),
                text=f'{result["confidence"] * 100:.2f}% confidence',
            )

        with col_score:
            st.write(f'**{result["confidence"] * 100:.2f}%**')

    with st.expander("🔬 Explore all class probabilities"):
        st.caption(
            "These scores show how the model distributed its confidence "
            "across the supported families. They are not guaranteed to be "
            "calibrated probabilities."
        )

        score_df = pd.DataFrame(
            {
                "Family": [
                    display_family_name(item["family"])
                    for item in results
                ],
                "Confidence (%)": [
                    round(item["confidence"] * 100, 3)
                    for item in results
                ],
            }
        )

        st.dataframe(
            score_df,
            hide_index=True,
            use_container_width=True,
        )

        chart_df = score_df.set_index("Family")
        st.bar_chart(chart_df, y="Confidence (%)")

    st.caption(
        "Remember: a high score does not guarantee that a prediction is "
        "correct. Results depend on the training data and image quality."
    )


# ---------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 18px;">
            <div style="font-size:42px;">👾</div>
            <div style="font-family:Orbitron; font-size:20px;
                        color:#41f6ff; font-weight:800;">
                SpriteSense AI
            </div>
            <div style="color:#aaa9ca; font-size:14px;">
                Retro Sprite Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 🎮 SYSTEM STATUS")

    if model is not None:
        st.success("Classifier loaded")
    else:
        st.error("Classifier not loaded")

    st.markdown("### 🧠 MODEL SPECS")

    st.markdown(
        f"""
        <div class="panel">
            <div style="color:#aaa9ca;">Architecture</div>
            <b>Convolutional Neural Network</b>
            <hr style="border-color:#353257;">
            <div style="color:#aaa9ca;">Input resolution</div>
            <b>24 × 24 pixels</b>
            <hr style="border-color:#353257;">
            <div style="color:#aaa9ca;">Supported families</div>
            <b>{len(class_names)} classes</b>
            <hr style="border-color:#353257;">
            <div style="color:#aaa9ca;">Framework</div>
            <b>TensorFlow / Keras</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 💡 HOW TO USE")
    st.markdown(
        """
        1. Choose a built-in sprite or upload your own.
        2. Click the classify button.
        3. Explore the top guesses.
        4. Expand the probability table for more detail.
        """
    )

    st.divider()
    st.caption("SpriteSense AI • Sprite AI Explorer")


# ---------------------------------------------------------
# 7. MAIN HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ NEURAL PIXEL ENGINE ONLINE ✦</div>
        <div class="hero-title">Sprite<span>Sense AI</span></div>
        <div class="hero-subtitle">
            Tiny sprites. Neural networks. Big imagination.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_error:
    st.error(f"Could not load the model: {model_error}")

if model is None:
    st.warning(
        "The trained model is not available. Place "
        "`final_sprite_classifier.keras` in the same folder as `app.py`, "
        "then restart the app."
    )
    st.code(
        "project_folder/\n"
        "├── app.py\n"
        "├── final_sprite_classifier.keras\n"
        "├── class_names.json\n"
        "└── requirements.txt"
    )

# ---------------------------------------------------------
# 8. INTERACTIVE CLASSIFICATION MODES
# ---------------------------------------------------------
tab_samples, tab_upload = st.tabs(
    ["🎮  Sprite Arcade", "📤  Upload Your Sprite"]
)

# Store selection across Streamlit reruns.
if "selected_sprite" not in st.session_state:
    st.session_state.selected_sprite = "Critter"

if "sample_prediction" not in st.session_state:
    st.session_state.sample_prediction = None

if "upload_prediction" not in st.session_state:
    st.session_state.upload_prediction = None

# ---------------------------------------------------------
# TAB 1: BUILT-IN SPRITE ARCADE
# ---------------------------------------------------------
with tab_samples:
    st.markdown(
        '<div class="section-label">🎨 CHOOSE YOUR PIXEL CHARACTER</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Select a sprite from the gallery and let '
        'the neural network predict its family.</p>',
        unsafe_allow_html=True,
    )

    sprite_names = list(SPRITES.keys())

    for start in range(0, len(sprite_names), 4):
        cols = st.columns(4, gap="medium")
        batch = sprite_names[start:start + 4]

        for col, sprite_name in zip(cols, batch):
            with col:
                sprite_image = SPRITES[sprite_name]["image"]()

                st.image(
                    upscale_sprite(sprite_image),
                    use_container_width=True,
                )

                st.markdown(
                    f"**{sprite_name}**",
                    unsafe_allow_html=True,
                )
                st.caption(SPRITES[sprite_name]["description"])

                selected = st.session_state.selected_sprite == sprite_name
                button_label = (
                    "✓ Selected" if selected else "Select sprite"
                )

                if st.button(
                    button_label,
                    key=f"select_{sprite_name}",
                    use_container_width=True,
                ):
                    st.session_state.selected_sprite = sprite_name
                    st.session_state.sample_prediction = None
                    st.rerun()

    selected_name = st.session_state.selected_sprite
    selected_image = SPRITES[selected_name]["image"]()

    st.markdown("---")
    preview_col, info_col = st.columns([1, 2], gap="large")

    with preview_col:
        st.markdown("### Selected sprite")
        st.image(
            upscale_sprite(selected_image, scale=10),
            caption=selected_name,
            use_container_width=True,
        )

    with info_col:
        st.markdown(f"### {selected_name}")
        st.write(SPRITES[selected_name]["description"])
        st.markdown(
            f"**Example theme:** {SPRITES[selected_name]['hint']}"
        )
        st.info(
            "The example theme is only a description. "
            "The model makes its own prediction and may choose another family."
        )

        if st.button(
            "⚡ CLASSIFY THIS SPRITE",
            key="classify_sample",
            type="primary",
            use_container_width=True,
        ):
            if model is None:
                st.error(
                    "Model unavailable. Check the model file and restart."
                )
            else:
                with st.spinner("Neural network is analysing pixels..."):
                    try:
                        st.session_state.sample_prediction = get_prediction(
                            selected_image
                        )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

    if st.session_state.sample_prediction is not None:
        show_results(
            selected_image,
            st.session_state.sample_prediction,
        )


# ---------------------------------------------------------
# TAB 2: USER IMAGE UPLOAD
# ---------------------------------------------------------
with tab_upload:
    st.markdown(
        '<div class="section-label">📤 UPLOAD YOUR OWN SPRITE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="muted">
        Upload a PNG, JPG, or JPEG image. For best results, use a small,
        clear sprite with a simple background. The model will resize it
        to 24 × 24 pixels before classification.
        </p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop your sprite here or browse files",
        type=["png", "jpg", "jpeg"],
        key="sprite_uploader",
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")

            preview_col, details_col = st.columns([1, 1.5], gap="large")

            with preview_col:
                st.image(
                    uploaded_image,
                    caption="Uploaded image",
                    use_container_width=True,
                )

            with details_col:
                st.markdown("### Image details")
                st.write(f"**Dimensions:** {uploaded_image.width} × {uploaded_image.height}")
                st.write("**Colour mode:** RGB")
                st.write("**Model input:** 24 × 24 pixels")
                st.write("**Preprocessing:** Resize and normalize")

                if st.button(
                    "🚀 CLASSIFY MY IMAGE",
                    key="classify_upload",
                    type="primary",
                    use_container_width=True,
                ):
                    if model is None:
                        st.error(
                            "Model unavailable. Check the model file and restart."
                        )
                    else:
                        with st.spinner("Scanning your sprite..."):
                            try:
                                st.session_state.upload_prediction = (
                                    get_prediction(uploaded_image)
                                )
                            except Exception as exc:
                                st.error(f"Prediction failed: {exc}")

            if st.session_state.upload_prediction is not None:
                show_results(
                    uploaded_image,
                    st.session_state.upload_prediction,
                )

        except Exception as exc:
            st.error(f"Could not open this image: {exc}")

    else:
        st.markdown(
            """
            <div class="panel" style="text-align:center; padding:35px 18px;">
                <div style="font-size:48px;">🖼️</div>
                <h3 style="color:#41f6ff;">Your sprite goes here</h3>
                <p class="muted">
                    Choose an image above to begin the classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# 9. FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <div style="font-family:Orbitron; color:#41f6ff;">
            SpriteSense AI
        </div>
        Built with Streamlit, TensorFlow and a little pixel magic ✦
        <br>
        Predictions are model estimates, not guaranteed labels.
    </div>
    """,
    unsafe_allow_html=True,
)