# Retro Sprite Classifier

A TensorFlow/Keras CNN that predicts one of 11 sprite-family labels from a pixel-art image, with a Streamlit interface.

## Files

- `final_sprite_classifier.ipynb` — documented training, validation, test evaluation, and export workflow.
- `requirements.txt` — Python dependencies.
- `app.py` — Streamlit user interface.
- `final_sprite_classifier.keras` — trained model artifact exported by the notebook.
- `class_names.json` — ordered class labels exported by the notebook.

## 1. Create an environment

Python 3.10 or 3.11 is recommended. From this directory, run:

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Export the model

Run `final_sprite_classifier.ipynb` in Jupyter or Google Colab. At the end, download/copy these generated artifacts into the same directory as `app.py`:

- `final_sprite_classifier.keras`
- `class_names.json`

The trained model file is not bundled here because it must be exported from the training notebook.

## 3. Launch the app

```bash
streamlit run app.py
```

## Notes

- Input images are resized to 24×24 RGB using nearest-neighbour interpolation.
- The model returns predictions across 11 selected families.
- Softmax scores are not guarantees of correctness. The model can be wrong on unfamiliar or ambiguous sprites.
- The best validation accuracy observed during development was 57.69%. Report the held-out test accuracy printed by the final notebook evaluation.
