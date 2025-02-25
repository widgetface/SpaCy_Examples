import spacy

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_sm")

# texts = ["I loved this hotel", "I really enjoyed  the film", "The food was awful"]

# labels = [1, 1, 0]


# def preprocess(text):
#     doc = nlp(text)
#     lemma_tokens = [token.lemma_ for token in doc if not token.is_stop]
#     return " ".join(lemma_tokens)


# processed_texts = [preprocess(text) for text in texts]

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(processed_texts)

# classifier = MultinomialNB()
# classifier.fit(X, labels)

# predicted_labels = classifier.predict(X)
# print(f"Predicted Labels: = {predicted_labels}")

nlp.add_pipe("spacytextblob")

reviews = [
    "It is the best hotel in town",
    "The car broke don several times do not hire from this company",
    "I love the food at this cafe",
]


for index, review in enumerate(reviews):
    doc = nlp(review)
    print(f"""
Review: '{reviews[index]}'.
{doc._.blob.polarity}                           # Polarity
{doc._.blob.subjectivity}                       # Subjectivity
{doc._.blob.sentiment_assessments.assessments}  # Assessments
{doc._.blob.ngrams()} # ngrams  """)
