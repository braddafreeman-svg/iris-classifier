# Iris Classifier (Decision Tree)

## Overview
This project builds a Decision Tree classifier on the classic Iris dataset as part of the AI Fundamentals module.  
It follows a full machine-learning workflow:

- Load the Iris dataset
- Split into train/test sets
- Train a DecisionTreeClassifier
- Predict on test data
- Evaluate using accuracy and a confusion matrix
- Save outputs to the `outputs/` folder

## Quick Start

```bash
git clone https://github.com/braddafreeman-svg/iris-classifier.git
cd iris-classifier

# Create + activate virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows

# Install requirements
pip install -r requirements.txt

# Run the training script
python src/train.py --test-size 0.2 --random-state 42
```
