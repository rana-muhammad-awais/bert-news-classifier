# Task 1: News Topic Classifier Using BERT

## Objective
Fine-tune a transformer model (BERT) to classify news headlines into four topic categories: World, Sports, Business, and Sci/Tech.

## Methodology / Approach
1. **Dataset Loading & Preprocessing**: 
 - Used the Hugging Face `datasets` library to download the `ag_news` dataset.
 - Employed `BertTokenizer` to tokenize the text sequences to a maximum length of 128 tokens with padding and truncation.
2. **Model Fine-Tuning**: 
 - Initialized `BertForSequenceClassification` from the `bert-base-uncased` checkpoint with 4 labels.
 - Used the Hugging Face `Trainer` API to fine-tune the model. A small subset (2000 samples) was used for demonstration purposes to allow for quick training, though the code supports full dataset training.
3. **Evaluation**:
 - The model is evaluated on a holdout set, calculating Accuracy and a weighted F1-score to determine performance.
4. **Deployment**:
 - Built a `Streamlit` application (`app.py`) that loads the fine-tuned model and provides a user-friendly web interface for live classification of custom news headlines.

## Key Results
- The fine-tuned BERT model correctly infers contextual meaning and categorizes unseen news snippets.
- Streamlit provides real-time confidence scores for the top 4 classes.

## How to Run

### 1. Install Dependencies
```bash
pip install torch numpy datasets transformers scikit-learn streamlit
```

### 2. Fine-tune the Model
Run the training script to fine-tune the model and save it locally:
```bash
python task1_bert_classifier.py
```
*Note: This will output the trained model to the `./saved_bert_model` directory.*

### 3. Launch the Streamlit App
Once the model is saved, start the Streamlit interactive application:
```bash
streamlit run app.py
```
Open your browser to the URL provided (usually `http://localhost:8501`) to test the classifier live.
