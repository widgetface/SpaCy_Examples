# A NLP take on an old interview classic palidrome question
import spacy
from spacy.tokens import Token


nlp = spacy.load("en_core_web_sm")

Token.set_extension(
    "is_palindrome", getter=lambda token: token.text == token.text[::-1]
)

doc = nlp("The band ABBA were great")

for token in doc:
    print(f"Token {token.text} is palindorme ={token._.is_palindrome}")
