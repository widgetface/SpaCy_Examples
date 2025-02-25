import spacy
from spacy.symbols import ORTH
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
from spacy.util import update_exc

nlp = spacy.load("en_core_web_sm")
custom_exception = {"ME/CFS": [{ORTH: "ME/CFS"}]}
nlp.tokenizer.rules = update_exc(BASE_EXCEPTIONS, custom_exception)
doc = nlp(" ME/CFS")

print([token.text for token in doc])
