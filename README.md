# Enron Email Dataset

Kaggle Dataset Containing Emails and other meta-information
[Kaggle Link]https://www.kaggle.com/wcukierski/enron-email-dataset/download

## Functionalities (Part 1)

Aim to Create a heuristics-based linguistic model for detecting actionable items from the email. A rule-based model to classify sentences to actionable sentence and non-actionable sentence


### Packages Required
```
pandas
tqdm
nltk
```


### Directory Structure and Important Files

```
./data/ : Email Data is stored in this folder. Not uploading on github. You can download directly from Kaggle Link.
./outputs/ : Save the output in this folder
main.py : Python Code for main file.
```

## Installation

Use the Anaconda with Python3 Environment

```bash
conda install pandas
conda install tqdm
conda install nltk
```

## Usage

```python
python main.py
```

### Evaluation

Usage of precision and recall for evaluation and using pos tagger patterns .