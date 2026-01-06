# IMDB Sentiment Analysis Project

## Overview
Train and evaluate BERT-based sentiment classifiers on IMDB movie reviews using PyTorch Lightning. Supports tiny BERT models for fast experimentation.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Modify config.yaml
1. Available models: prajjwal1/bert-tiny (4M params), huawei-noah/TinyBERT_General_4L_312D (14M params)
2. Update paths.models_dir and paths.logger_dir for your model.
3. Adjust batch_size, learning_rate, max_epochs as needed

### 3. Train the model
```
python train.py
```
Output: sentiment-analyzer.ckpt (best model) saved in models_dir

#### Monitor Training:
```
tensorboard --logdir <logger_dir>
```
View loss/accuracy curves at http://localhost:6006

### 4. Evaluate the model
```
python evaluate.py 
```
Output: Error analysis metrics printed.

