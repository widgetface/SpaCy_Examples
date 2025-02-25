import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex

nlp = spacy.blank("en")

custom_prefixes = nlp.Defaults.prefixes + [r"\\£", r"\£"]  # Handle GB pound sign
custom_suffixes = nlp.Defaults.suffixes + [r"%"]  # Handle percentages
custom_infixes = nlp.Defaults.infixes + [r"\.\.\."]  # Handle elipses

prefix_re = compile_prefix_regex(custom_suffixes)
suffix_re = compile_suffix_regex(custom_prefixes)
infix_re = compile_infix_regex(custom_infixes)

custom_tokenizer = Tokenizer(
    nlp.vocab,
    prefix_search=prefix_re.search,
    suffix_search=suffix_re.search,
    infix_finditer=infix_re.finditer,
)

doc = custom_tokenizer(
    "The price of a kilogramme of cheese has incresed from £1.50 to £1.80 up 20%  in a year ..."
)

print([token.text for token in doc])
