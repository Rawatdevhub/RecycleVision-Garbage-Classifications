# RecycleVision

Garbage image classification using transfer learning and Streamlit.

## What it does

RecycleVision classifies an uploaded image into one of 10 waste categories using a MobileNetV2 image classifier.

Dataset: [Garbage Dataset on Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

Expected classes: `metal`, `glass`, `biological`, `paper`, `battery`, `trash`, `cardboard`, `shoes`, `clothes`, `plastic`.

## Repository layout

```text
recyclevision/
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── class_names.json
├── src/
│   └── predict.py
├── data/              # local only, never commit images
├── models/            # local model output
└── reports/           # evaluation outputs
```

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. Get the Kaggle data

Download and unzip the dataset from Kaggle, then set `DATA_DIR` to the folder containing one subfolder per class. You can also use the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d sumn2u/garbage-classification-v2 -p data --unzip
```

Do not commit `kaggle.json`, image files, or credentials.

## 3. Train

```bash
python train.py --data_dir data/garbage-classification-v2 --epochs 12
```

The script creates `models/recyclevision.keras` and `models/class_names.json`.

## 4. Evaluate

```bash
python evaluate.py --data_dir data/garbage-classification-v2 --model_path models/recyclevision.keras
```

Evaluation outputs are written to `reports/metrics.json` and `reports/confusion_matrix.png`.

## 5. Run Streamlit

```bash
streamlit run app.py
```

For Streamlit Community Cloud, push this repository to GitHub, choose `app.py` as the entry point, and add the trained model under `models/` using Git LFS or download it from a model store at startup. For a small demo, Git LFS is recommended.

## Important limitations

The model is a prototype. Real performance depends on lighting, camera angle, background clutter, class balance, and local recycling rules. Use the confidence score as a review signal, not as a guarantee.
