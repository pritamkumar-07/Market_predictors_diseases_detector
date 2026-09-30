# Market Predictor & Disease Detector for Farmer

A Python-based academic project combining:
- Python
- NumPy
- Pandas
- Statistics & Probability
- Machine Learning / ML Pipeline
- Deep Learning
- Neural Networks
- CNN (Convolutional Neural Network)
- Streamlit user interface
- Matplotlib / Seaborn

## Important
This project is designed to be runnable and extensible, but the disease detector requires a real labeled crop-leaf image dataset before it can make meaningful disease predictions. The repository deliberately does **not** fabricate disease images, model accuracy, or trained disease results.

## Features

### 1. Market Predictor
- Uses a sample agricultural market dataset.
- Cleans and preprocesses data.
- Creates time/season features.
- Trains a Random Forest regression model.
- Evaluates with MAE, RMSE and R².
- Predicts an estimated market price for new inputs.
- Shows descriptive statistics and plots.

### 2. Disease Detector
- CNN training pipeline using TensorFlow/Keras.
- Supports folder-based datasets:
  `disease_dataset/train/<class_name>/...`
  `disease_dataset/validation/<class_name>/...`
  `disease_dataset/test/<class_name>/...`
- Saves a trained CNN to `models/disease_cnn.keras`.
- Streamlit app accepts a crop-leaf image and predicts a class after a real model has been trained.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

## Run the market model

```bash
python scripts/train_market_model.py
```

This creates:
`models/market_model.joblib`

## Run the application

```bash
streamlit run app.py
```

## Train the disease CNN

Place a real labeled dataset in the folders described above, then run:

```bash
python scripts/train_disease_cnn.py
```

The script saves:
- `models/disease_cnn.keras`
- `models/class_names.json`

Then restart Streamlit.

## Disease dataset format

Example:

```text
disease_dataset/
├── train/
│   ├── Healthy/
│   ├── Tomato_Early_Blight/
│   └── Tomato_Late_Blight/
├── validation/
│   ├── Healthy/
│   ├── Tomato_Early_Blight/
│   └── Tomato_Late_Blight/
└── test/
    ├── Healthy/
    ├── Tomato_Early_Blight/
    └── Tomato_Late_Blight/
```

Class names must match across train/validation/test.

## Project architecture

```text
                  Farmer
                    |
             Streamlit Interface
              /              \
             /                \
     Market Predictor      Disease Detector
           |                     |
     Pandas + NumPy         Image Preprocessing
           |                     |
     Statistics              CNN / Deep Learning
           |                     |
   Random Forest             Classification
           |                     |
    Estimated Price          Disease Class
```

## Academic honesty

The sample market dataset is clearly synthetic/demo data so the application can run immediately. Replace it with a real agricultural market dataset for an actual research result.

The disease module does not claim real-world accuracy until a genuine labeled crop-disease dataset is supplied and the CNN is trained/evaluated.
