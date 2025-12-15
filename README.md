# Job Description Skill Extractor

Automated NLP system that extracts skill entities from raw job descriptions using Named Entity Recognition (NER). The project compares a lightweight Tok2Vec‑based model with a RoBERTa transformer model and exposes the final model through a Gradio demo for interactive testing.

## Overview

Recruiters and job seekers deal with long, unstructured job postings that mix responsibilities, benefits, and requirements. This project turns those texts into structured skill lists by training NER models on annotated job descriptions. Two architectures are implemented and evaluated:

- **Tok2Vec‑based NER** – a spaCy baseline with a lightweight encoder  
- **Transformer‑based NER (RoBERTa)** – a contextual model using `spacy-transformers`  

On a held‑out dev set of 500 examples, the transformer achieves higher recall and a ~15‑point F1 improvement over the Tok2Vec baseline.

## Model Download

The main transformer model is larger than GitHub’s per‑file size limit, so it is distributed via Google Drive instead of being stored directly in this repository.  

Download the trained transformer model here:

**[Download transformer model from Google Drive](https://drive.google.com/drive/folders/1YA9UIN6InLSZ9H1YSUMcbKosUr3Rb0Cp?usp=sharing)**

After downloading, place the `model-best` directory at:


## Data and Models

- **Data**
  - Job postings collected from LinkedIn and related sources  
  - Cleaned and merged into train / dev corpora stored as `.spacy` binaries with `SKILL` entity annotations  

- **Models**
  - `output_tok2vec/`: spaCy NER pipeline with `tok2vec` + `ner`  
  - `tf_skill_ner_model_8k_clean/`: spaCy transformer pipeline using a RoBERTa encoder + NER head (weights downloaded separately as described above)

## Notebook and Workflow

The main notebook (`code file.ipynb`) walks through the full pipeline:

- Loading and inspecting job description data  
- Converting CSV data to spaCy `.spacy` format  
- Training Tok2Vec and transformer NER models  
- Evaluating on a 500‑example dev set (precision, recall, F1)  
- Plotting skills‑per‑job distributions and model metrics  

The notebook also contains a simple **Gradio demo** that lets you paste a job description and see the extracted skills in real time.

## Installation

Create a virtual environment (recommended), then install the dependencies:

pip install -r requirements.txt

