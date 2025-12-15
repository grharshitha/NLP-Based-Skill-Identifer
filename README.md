# Job Description Skill Extractor

Automated NLP system that extracts skill entities from raw job descriptions using Named Entity Recognition (NER). The project compares a lightweight Tok2Vec‑based model with a RoBERTa transformer model, and exposes the final model through a Gradio demo for interactive testing. 

## Project Overview

Recruiters and job seekers deal with long, unstructured job postings that mix responsibilities, benefits, and requirements. This project turns those texts into structured skill lists by training NER models on annotated job descriptions. 

Two architectures are implemented and evaluated:

- **Tok2Vec‑based NER** – a spaCy baseline with a lightweight encoder. 
- **Transformer‑based NER (RoBERTa)** – a contextual model using `spacy-transformers`.  

On a held‑out dev set of 500 examples, the transformer achieves higher recall and a ~15‑point F1 improvement over the Tok2Vec baseline. 

## Repository Structure

.
├─ data/
│ ├─ skill_ner_weak_train_clean.spacy # weakly labeled training data
│ ├─ skill_ner_train_8k_clean.spacy # main 8k training set
│ └─ skill_ner_dev_500_clean.spacy # 500-example dev set
├─ models/
│ ├─ tf_skill_ner_model_8k_clean/ # trained transformer NER model
│ └─ output_tok2vec/ # trained Tok2Vec NER model
├─ code file.ipynb # main Colab notebook
├─ requirements.txt
└─ README.md


The `.spacy` files are spaCy binary docs containing annotated job descriptions with `SKILL` entities. 

## Key Components

- **Data**  
  - Job postings collected from LinkedIn and related sources.  
  - Cleaned and merged into train / dev corpora stored as `.spacy` files.   

- **Models**  
  - `output_tok2vec/`: spaCy NER pipeline with `tok2vec` + `ner`.  
  - `tf_skill_ner_model_8k_clean/`: spaCy transformer pipeline using RoBERTa encoder + NER head. 

- **Notebook (`job_skills_-4.ipynb`)**  
  - Data loading and inspection.  
  - Converting CSV data to spaCy format.  
  - Training Tok2Vec and transformer models.  
  - Evaluation (precision, recall, F1) on the 500‑example dev set.  
  - Visualizations: skills‑per‑job histogram, model metric bar chart. 

- **Demo (Gradio, in notebook)**  
  - Simple text box where users paste a job description.  
  - Backend runs the best NER model and returns extracted skills. 

## Installation

Create a virtual environment (recommended), then install dependencies:

pip install -r requirements.txt


Download any spaCy transformer model you reference in the config (if not already included):

python -m spacy download en_core_web_trf



## Running the Notebook Locally

1. Clone the repository and move into it:

git clone https://github.com/<your-username>/nlp-skill-extractor.git
cd nlp-skill-extractor



2. Install requirements as above.

3. Open the notebook:

jupyter notebook code file.ipynb


4. In the notebook, update any paths if needed so they point to:

base_path = "./" # or the correct local path

text

5. Run the cells in order to:
   - Load `.spacy` corpora from `data/`.  
   - Train or reload the NER pipelines from `models/`.  
   - Reproduce evaluation metrics and plots.

## Using the Trained Models

You can load the provided models directly in Python:

import spacy

tok2vec_nlp = spacy.load("models/output_tok2vec")
tf_nlp = spacy.load("models/tf_skill_ner_model_8k_clean/model-best")

text = "We are looking for a Software Engineer with strong Python, SQL, and AWS experience."
doc = tf_nlp(text)
skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
print(skills)

text

## Gradio Demo (Interactive UI)

Inside the notebook there is a Gradio interface that:

- Takes a job description as input.  
- Runs the transformer‑based NER model.  
- Displays the list of extracted skills 

Typical structure:

import gradio as gr
import spacy

nlp = spacy.load("models/tf_skill_ner_model_8k_clean/model-best")

def extract_skills(text):
doc = nlp(text)
return ", ".join(sorted({ent.text for ent in doc.ents if ent.label_ == "SKILL"}))

demo = gr.Interface(
fn=extract_skills,
inputs=gr.Textbox(lines=8, label="Job Description"),
outputs=gr.Textbox(label="Extracted Skills"),
title="Job Description Skill Extractor"
)
demo.launch()


## Results (Summary)

On the 500‑example dev set: 

- **Transformer (RoBERTa)**  
  - Precision ≈ 48.56%  
  - Recall ≈ 36.81%  
  - F1 ≈ 41.87%  

- **Tok2Vec baseline**  
  - Precision ≈ 43.55%  
  - Recall ≈ 19.59%  
  - F1 ≈ 27.03%  

The transformer nearly doubles recall while keeping precision strong, leading to a ~15‑point F1 improvement over Tok2Vec. 

## Notes

- The training corpora in `data/` are provided as spaCy binaries rather than raw text to avoid sharing full source job postings. 
- Large, full‑scale datasets used to construct these corpora are not included in the repository. 