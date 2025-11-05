# Fake News Detection (NLP) with BERT & Streamlit

[Insert a GIF of your Streamlit app in action]

This project is an end-to-end NLP workflow to classify real vs. fake news. It demonstrates a critical data science concept: building a simple baseline model, identifying its flaws (data leakage), and then building a robust, deep learning model (BERT) to solve the problem correctly.

**Final Result:** The fine-tuned **DistilBERT** model achieved **99.6% accuracy** on the *titles* alone, proving it learned the linguistic patterns of fake news, not just a data leak.

---

### 1. The Aim & Baseline Model (TF-IDF)

The initial goal was to build a classifier using the text of news articles.
1.  **Preprocessing:** I cleaned the text from two datasets (real and fake) using regex and NLP libraries to remove stopwords and punctuation.
2.  **Baseline Model:** I built a baseline using **TF-IDF (Term Frequency-Inverse Document Frequency)**. TF-IDF converts text into a numerical matrix by finding words that are **frequent in one document (High TF)** but **rare in all other documents (High IDF)**.
3.  **Baseline Result:** This TF-IDF matrix was fed into a Logistic Regression model, which achieved a suspiciously high **98.8% F1-score**.

### 2. The Critical Insight: Data Leakage

Upon investigation, I discovered the baseline's high score was due to **data leakage**. The real news articles almost all contained the word "Reuters" in the text, while the fake ones did not. The model didn't learn to "detect fake news"; it just learned to "detect the word 'Reuters'."

### 3. The Advanced Model (BERT)

To build a *real* model, I moved to a more advanced architecture and a harder, cleaner dataset (the **titles** only).

1.  **Model:** I used **BERT (Bidirectional Encoder Representations from Transformers)**, a massive, pre-trained language model from Google. Unlike TF-IDF, BERT understands *context* by using a **Self-Attention** mechanism to analyze words in relation to all other words in a sentence (up to a 512-token limit).
2.  **Tools:** I used the Hugging Face `transformers` library:
    * `AutoTokenizer` to load the specific tokenizer for `DistilBERT`.
    * `AutoModelForSequenceClassification` to load the pre-trained BERT "brain" and add a new, untrained classifier "head" on top.
3.  **Fine-Tuning (Transfer Learning):** This is the key. I **froze** the parameters of the massive BERT "brain" (so it wouldn't alter its understanding of language) and **trained only the tiny new "head"** on our "fake" vs. "real" labels. The new head learned to interpret the brain's complex output.

### 4. Deployment (Streamlit)
The final, fine-tuned BERT model was saved and deployed as an interactive web app using **Streamlit**. The app takes a user's headline and predicts in real-time if it's real or fake.

### 5. Tools Used
* **Python**
* **Data Cleaning:** Pandas, Regex, NLTK
* **Baseline Model:** Scikit-learn, TF-IDF, Logistic Regression
* **Advanced Model:** PyTorch, Hugging Face `transformers` (DistilBERT), `datasets`
* **Deployment:** Streamlit
