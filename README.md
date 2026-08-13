# Pathological Speech Recognition Using Deep Learning

## Overview

This repository presents deep learning approaches for pathological speech processing, acoustic modeling, and speech recognition.

The project investigates acoustic features, speech preprocessing, and deep learning architectures for the analysis and recognition of pathological speech.

This work is part of research activities in acoustic modeling and pathological speech assessment.

---

## Research Objectives

The main objectives of this project are:

- Preprocess pathological speech signals.
- Investigate speech enhancement techniques.
- Extract relevant acoustic and spectral features.
- Develop deep learning models for pathological speech recognition.
- Investigate CNN-based architectures.
- Investigate BiLSTM-based temporal modeling.
- Develop hybrid CNN-BiLSTM architectures.
- Compare different acoustic representations.
- Evaluate recognition performance using appropriate metrics.

---

## Research Pipeline

```text
                         Speech Signal
                              |
                              v
                    Audio Preprocessing
                              |
                              v
                     Speech Enhancement
                              |
                              v
                    Feature Extraction
                              |
             +----------------+----------------+
             |                |                |
            MFCC             PNCC       Mel-Spectrogram
             |                |                |
             +----------------+----------------+
                              |
                              v
                       Deep Learning
                              |
                  +-----------+-----------+
                  |                       |
                 CNN                   BiLSTM
                  |                       |
                  +-----------+-----------+
                              |
                              v
                         CNN-BiLSTM
                              |
                              v
                     Speech Recognition
                              |
                              v
                         Evaluation
```

---

## Acoustic Features

The project investigates several speech representations.

### MFCC

Mel-Frequency Cepstral Coefficients are used to represent the spectral characteristics of speech.

### PNCC

Power-Normalized Cepstral Coefficients are investigated as an alternative acoustic representation for speech processing.

### Mel-Spectrogram

Log-Mel spectrograms are used to represent the time-frequency characteristics of speech and can be provided as input to convolutional neural networks.

### Prosodic and Voice-Quality Features

Additional acoustic characteristics may include:

- Fundamental frequency (F0)
- Jitter
- Shimmer

These characteristics are particularly relevant for pathological speech analysis.

---

## Deep Learning Models

The repository contains implementations of several neural network architectures.

### CNN

Convolutional Neural Networks are used to learn local patterns from spectral speech representations.

### BiLSTM

Bidirectional Long Short-Term Memory networks are used to model temporal dependencies in speech sequences.

### CNN-BiLSTM

The hybrid CNN-BiLSTM architecture combines convolutional feature extraction with bidirectional temporal modeling.

---

## Datasets

The project can be adapted to research speech databases including:

- UASpeech
- MEEI
  

The original speech datasets are not distributed with this repository.

Users must obtain the datasets from their official sources and comply with the corresponding terms of use and licenses.

See `data/README.md` for dataset organization information.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BelSou01/pathological-speech-recognition.git
cd pathological-speech-recognition
```


### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Technologies

The project primarily uses:

- Python
- PyTorch
- NumPy
- SciPy
- Librosa
- Scikit-learn
- Matplotlib
- Pandas

MATLAB may also be used for complementary signal-processing experiments.

---

## Project Structure

```text
pathological-speech-recognition/
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
├── data/
│   └── README.md
│
├── preprocessing/
│   ├── audio_preprocessing.py
│   └── ...
│
├── features/
│   ├── mfcc.py
│   ├── pncc.py
│   ├── mel_spectrogram.py
│   └── prosodic_features.py
│
├── models/
│   ├── cnn.py
│   ├── bilstm.py
│   └── cnn_bilstm.py
│
├── training/
│   └── train.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── confusion_matrix.py
│
├── experiments/
│   └── experiments.md
│
└── results/
    └── results.md
```

---

## Related Publication

This repository is associated with the following research work:

**Pathological voice classification system based on CNN-BiLSTM network using speech enhancement and multi-stream approach**

**Authors:** Soumeya Belabbas, Djamel Addou & Sid Ahmed Selouani

**Journal:** International Journal of Speech Technology

**Year:** [05 July 2024]

**DOI:** https://doi.org/10.1007/s10772-024-10120-w

**Publication:** Belabbas, S., Addou, D. & Selouani, S.A. Pathological voice classification system based on CNN-BiLSTM network using speech enhancement and multi-stream approach. Int J Speech Technol 27, 483–502 (2024). https://doi.org/10.1007/s10772-024-10120-w

---

## Experimental Evaluation

The experiments investigate the effect of:

- Acoustic feature representation
- Speech preprocessing
- Speech enhancement
- Neural network architecture
- Training configuration

Depending on the experiment, evaluation may include:

- Accuracy
- Confusion Matrix



## Reproducibility

Each experiment should document:

1. Dataset
2. Dataset partition
3. Sampling rate
4. Feature representation
5. Model architecture
6. Training parameters
7. Loss function
8. Optimizer
9. Evaluation metrics
10. Experimental results

---

## Research Context

This repository is part of research work in:

- Speech Processing
- Pathological Speech
- Acoustic Modeling
- Speech Enhancement
- Automatic Speech Recognition
- Deep Learning

The research investigates computational approaches for pathological speech analysis and recognition.


---

## Author

**Soumeya Belabbas**

PhD in Telecommunications and Information Processing

### Research Interests

- Speech Processing
- Pathological Speech
- Speech Enhancement
- Automatic Speech Recognition
- Acoustic Modeling
- Deep Learning
- CNN
- BiLSTM
- Transformers

---

## Disclaimer

This repository is intended for research and educational purposes.

The repository does not contain restricted speech datasets or original patient recordings.

Users are responsible for obtaining the datasets from their official sources and complying with their respective terms of use.
