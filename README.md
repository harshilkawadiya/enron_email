# Enron Email Dataset

Kaggle Dataset Containing Emails and other meta-information
[Kaggle Link]https://www.kaggle.com/wcukierski/enron-email-dataset/download

## Functionalities (Part 1)

Aim to Create a heuristics-based linguistic model for detecting actionable items from the email. A rule-based model to classify sentences to actionable sentence and non-actionable sentence

## Functionalities (Part 2)

Train a model to detect whether a given sentence is an actionable item or not. 

```
Actionable item => A sentence which asks someone to do something
example: "Please create an assignment and forward it by EOD"
```



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
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Usage

```python
python main.py
```

### Evaluation

Usage of precision and recall for evaluation and using POS tagger patterns .



