# CMU Machine Learning Fundamentals — Homework 4: Fashion MNIST Classification

This repository contains modularized code and analysis for **Homework 4** of the CMU course **46-886 Machine Learning Fundamentals**. The assignment focuses on implementing, training, and analyzing both fully connected and convolutional neural networks using PyTorch on the Fashion MNIST dataset.

---

## 📁 Project Structure

```
hw4_project/
│
├── hw4.py                     # Assignment-compliant: one training run per model
├── hw4_training_analysis.py  # Extended multi-epoch analysis, visualizations, and metrics
│
├── models/
│   ├── __init__.py
│   └── models.py             # MLP and CNN class definitions
│
├── utils/
│   ├── __init__.py
│   ├── data_utils.py         # Data loading and preprocessing
│   └── train_utils.py        # Training, evaluation, and plotting functions
│
├── data/
│   └── fashion_mnist.p       # Pickled Fashion MNIST dataset (provided)
│
├── figures/                  # Automatically generated training curves and confusion matrices
│
└── README.md                 # Project overview (you are here)
```

---

## ✅ hw4.py – Assignment Submission

Trains and evaluates four models using a **single training run**:

| Model Name  | Architecture                         |
|-------------|--------------------------------------|
| SimpleNN    | Linear(784→128) → ReLU → Linear(10)  |
| DeepNN      | Linear(784→128) → ReLU → Linear(32) → ReLU → Linear(10) |
| CNN1        | Conv(1→40, 6×6) → ReLU → Linear(…) → 10 |
| CNN2        | Conv(1→32, 5×5) → ReLU → Linear(32) → ReLU → Linear(10) |

### Metrics Reported:
- Cross-Entropy Loss (Train)
- Accuracy (Train/Test)

---

## 📊 hw4_training_analysis.py – Deep Dive Analysis

This script extends the assignment to:

- Train each model with **10, 50, and 100 epochs**
- Track and plot:
  - Training loss curves
  - Training accuracy curves
  - Confusion matrices
- Compute:
  - Final training accuracy
  - Test accuracy
  - Macro-averaged F1 score

Use this script to evaluate model behavior over time and support conclusions on underfitting vs. overfitting.

---

## 🧠 Modular Components

- **`models/models.py`** – Houses `SimpleNN`, `DeepNN`, `CNN1`, and `CNN2`.
- **`utils/data_utils.py`** – Loads and reshapes Fashion MNIST into flat or CNN-compatible format.
- **`utils/train_utils.py`** – Training loops, evaluation metrics, visualizations.

Each module is designed to be importable and clean for future reuse or packaging.

---

## ▶️ How to Run

1. **Install dependencies**:
   ```bash
   pip install torch matplotlib scikit-learn
   ```

2. **Run basic assignment version**:
   ```bash
   python hw4.py
   ```

3. **Run extended training analysis**:
   ```bash
   python hw4_training_analysis.py
   ```

> Make sure the dataset file `fashion_mnist.p` is placed in the `data/` directory.

---

## 🔍 Output Visuals

All plots are automatically saved in the `figures/` directory:

- `*_curves_10ep.png`, `*_curves_50ep.png`, etc.
- `*_confusion_10ep.png`, etc.

These illustrate how each model performs across training durations.

---

## 📌 Notes

- The analysis is designed to be both **submission-ready** and **portfolio-ready**.
- Supports **Apple Silicon acceleration** with `num_workers=14`.
- Fully self-contained: can be extended with additional models or datasets easily.

---

## 👩‍💻 Author

**Michelle Wu**  
Carnegie Mellon University  
Spring 2025 — Machine Learning Fundamentals  
[GitHub: @michellewu](https://github.com/michellewu)
