import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. SET UP PAGE ---
# We set the page title and icon (emoji)
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# --- 2. LOAD THE SAVED MODEL AND TOKENIZER ---
# We use @st.cache_resource to load the model only once, making the app faster
@st.cache_resource
def load_model():
    print("Loading model...")
    model_path = "./bert_fake_news_model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

tokenizer, model = load_model()

if model is None:
    st.stop()

# --- 3. BUILD THE USER INTERFACE ---

# Title with icon
st.title("Fake News Detector 📰")

# App description
st.markdown("""
This app uses a fine-tuned **DistilBERT** model to predict whether a news headline is **Real** or **Fake**. 
The model was trained on the `title` column of the "Fake and Real News" dataset, 
forcing it to learn linguistic patterns, not just data leaks.
""")
st.divider()

# Create a "card" for the input using a container
with st.container(border=True):
    st.header("Enter a Headline to Classify")
    
    # We'll use a form so the page doesn't re-run on every keypress
    with st.form("news_form"):
        user_input = st.text_area("News Headline:", "BREAKING: Secret Documents Reveal Lizards Are Running the White House", height=100)
        
        # Add a submit button with use_container_width to make it full-width
        submit_button = st.form_submit_button("Classify Headline", type="primary", use_container_width=True)

# --- 4. MAKE AND DISPLAY THE PREDICTION ---

if submit_button and user_input:
    # 1. Tokenize the input text (with the "max_length" fix)
    inputs = tokenizer(user_input, 
                       padding="max_length", # <-- This is the critical fix
                       truncation=True, 
                       return_tensors="pt")

    # 2. Get the model's prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # 3. Get the probabilities
    probs = torch.softmax(logits, dim=1)
    
    # 4. Get the most likely class (0 for 'Real', 1 for 'Fake')
    prediction = torch.argmax(probs, dim=1).item()
    
    # 5. Get the confidence score
    confidence = probs[0][prediction].item()

    st.divider()
    
    # 6. Display the result in a new "card"
    with st.container(border=True):
        st.header("Prediction Result")
        
        # Use columns to put the metric (the score) and the text side-by-side
        col1, col2 = st.columns([1, 2])
        
        if prediction == 1:
            # Column 1: The big score
            with col1:
                st.metric("Confidence", f"{confidence:.2%}", "FAKE")
            # Column 2: The explanation
            with col2:
                st.error("**Prediction: FAKE NEWS**")
                st.write("The model is highly confident that this headline is fake. It likely contains linguistic patterns (e.g., sensationalism, unusual claims) associated with fake news.")
        else:
            # Column 1: The big score
            with col1:
                st.metric("Confidence", f"{confidence:.2%}", "REAL")
            # Column 2: The explanation
            with col2:
                st.success("**Prediction: REAL NEWS**")
                st.write("The model is confident this headline is real. Its language and structure align with patterns found in trusted news sources from the dataset.")