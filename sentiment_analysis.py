import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

# Step 1: Load the en_core_web_sm spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

# Step 2: Load the dataset and preprocess the text data
def preprocess_text(text):
    doc = nlp(text)
    cleaned_text = " ".join(token.lemma_ for token in doc if not token.is_stop and token.is_alpha)
    return cleaned_text

def load_and_preprocess_data(file_path):
    # Load the dataset
    dataframe = pd.read_csv(file_path)

    # Step 2.1: Select the 'review.text' column
    reviews_data = dataframe['reviews.text']

    # Step 2.2: Remove missing values
    clean_data = dataframe.dropna(subset=['reviews.text'])

    # Apply text preprocessing to the 'review.text' column
    clean_data['cleaned_reviews'] = clean_data['reviews.text'].apply(preprocess_text)

    return clean_data

# Step 3: Create a function for sentiment analysis
def sentiment_analysis(review):
    # Analyze sentiment using spaCy and TextBlob
    doc = nlp(review)
    polarity = doc._.polarity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return sentiment


# Step 4: Test the model on sample product reviews
def test_model(sample_reviews):
    for review in sample_reviews:
        sentiment = sentiment_analysis(review)
        print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")


# Main
# Load and preprocess the data
file_path = 'amazon_product_reviews2.csv'  
clean_data = load_and_preprocess_data(file_path)

# Test the model on sample product reviews
test_model(clean_data['cleaned_reviews'].head(20))
