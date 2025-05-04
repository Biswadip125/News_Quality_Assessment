import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()

# Map POS tags to WordNet tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

#function for removing the special characters and punctuation
def clean_text(text):
     # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 2. Remove special characters (but keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    #3 lowercase 
    text = text.lower();
    
    # 4. Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # 5. Strip leading and trailing spaces
    text = text.strip()
    
     # 6. Tokenize and POS tag
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    
    # 7. Lemmatize and remove stopwords
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged_tokens if word not in stop_words
    ]
    
    return " ".join(lemmatized)

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("vc_model.pkl")

st.title("Fake News Detector")
st.write("Enter a news article below to check whether it is Fake or Real")

news_input = st.text_area("News Article", "")

cleaned_input = clean_text(news_input)


if st.button("Check News"):
  if cleaned_input.strip():
    transform_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(transform_input)

    if prediction[0] == 0:
      st.error("The News is Fake!")
    else: 
      st.success("The News is Real!")
  else:
    st.warning("Please enter some text to analyze. ")