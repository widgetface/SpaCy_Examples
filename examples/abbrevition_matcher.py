from typing import Tuple, List, Optional, Set, Dict
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span, Doc
from collections import defaultdict

THRESHOLD_WORD_RANGE = range(1, 9)


def filter_short_form(span: Span) -> bool:
    # All words are between length 2 and 10

    if not all([len(x) in THRESHOLD_WORD_RANGE for x in span]):
        return False

    # At least 50% of the short form should be alpha
    if (sum([c.isalpha() for c in span.text]) / len(span.text)) < 0.5:
        return False

    # The first character of the short form should be alpha
    if not span.text[0].isalpha():
        return False
    return True


def has_unbalanced_parentheses(span: Span) -> bool:
    right_parenthesis: False
    left_parenthesis = False
    for token in span:
        if token.text == "(":
            right_parenthesis = True
        if token.text == ")":
            left_parenthesis = True

    return right_parenthesis and left_parenthesis


def filter_matches(
    matcher_output: List[Tuple[int, int, int]], doc: Doc
) -> List[Tuple[Span, Span]]:
    # Filter into two cases:
    # 1. <Short Form> ( <Long Form> )
    # 2. <Long Form> (<Short Form>) [this case is most common].
    candidates = []
    for match in matcher_output:
        start = match[1]
        end = match[2]

        print(start, end)
        # Ignore spans with more than 8 words in them, and spans at the start of the doc
        if end - start > 8 or start == 1:
            continue
        if end - start > 3:
            # Long form is inside the parens.
            # Take one word before.
            short_form_candidate = doc[start - 2 : start - 1]
            long_form_candidate = doc[start:end]

            # make sure any parentheses inside long form are balanced
            if has_unbalanced_parentheses(long_form_candidate):
                continue
        else:
            # Normal case.
            # Short form is inside the parens.
            short_form_candidate = doc[start:end]
            print(type(doc), doc)
            print(type(doc[start:end]), doc[start:end])
            # Sum character lengths of contents of parens.
            abbreviation_length = sum([len(x) for x in short_form_candidate])
            max_words = min(abbreviation_length + 5, abbreviation_length * 2)
            # Look up to max_words backwards
            print(f"max(start - max_words - 1, 0) -  {max(start - max_words - 1, 0)}")
            long_form_candidate = doc[max(start - max_words - 1, 0) : start - 1]

        # add candidate to candidates if candidates pass filters
        if filter_short_form(short_form_candidate):
            candidates.append((long_form_candidate, short_form_candidate))

    return candidates


def find_abbreviation(
    long_form_candidate: Span, short_form_candidate: Span
) -> Tuple[Span, Optional[Span]]:
    """
    Implements the abbreviation detection algorithm in "A simple algorithm
    for identifying abbreviation definitions in biomedical text.", (Schwartz & Hearst, 2003).

    The algorithm works by enumerating the characters in the short form of the abbreviation,
    checking that they can be matched against characters in a candidate text for the long form
    in order, as well as requiring that the first letter of the abbreviated form matches the
    _beginning_ letter of a word.

    Parameters
    ----------
    long_form_candidate: Span, required.
        The spaCy span for the long form candidate of the definition.
    short_form_candidate: Span, required.
        The spaCy span for the abbreviation candidate.

    Returns
    -------
    A Tuple[Span, Optional[Span]], representing the short form abbreviation and the
    span corresponding to the long form expansion, or None if a match is not found.
    """

    long_form = " ".join([x.text for x in long_form_candidate])
    short_form = " ".join([x.text for x in short_form_candidate])

    long_index = len(long_form) - 1
    short_index = len(short_form) - 1

    while short_index >= 0:
        current_char = short_form[short_index].lower()
        # We don't check non alpha-numeric characters.
        if not current_char.isalnum():
            short_index -= 1
            continue

            # Does the character match at this position? ...
        while (
            (long_index >= 0 and long_form[long_index].lower() != current_char)
            or
            # .... or if we are checking the first character of the abbreviation, we enforce
            # to be the _starting_ character of a span.
            (
                short_index == 0
                and long_index > 0
                and long_form[long_index - 1].isalnum()
            )
        ):
            long_index -= 1

        if long_index < 0:
            return short_form_candidate, None

        long_index -= 1
        short_index -= 1

    # The last subtraction will either take us on to a whitespace character, or
    # off the front of the string (i.e. long_index == -1). Either way, we want to add
    # one to get back to the start character of the long form
    long_index += 1

    # Now we know the character index of the start of the character span,
    # here we just translate that to the first token beginning after that
    # value, so we can return a spaCy span instead.
    word_lengths = 0
    starting_index = None
    for i, word in enumerate(long_form_candidate):
        # need to add 1 for the space characters
        word_lengths += len(word.text_with_ws)
        if word_lengths > long_index:
            starting_index = i
            break

    return short_form_candidate, long_form_candidate[starting_index:]


def find_matches_for(
    filtered: List[Tuple[Span, Span]], doc: Doc
) -> List[Tuple[Span, Set[Span]]]:
    rules = {}
    all_occurences: Dict[Span, Set[Span]] = defaultdict(set)
    already_seen_long: Set[str] = set()
    already_seen_short: Set[str] = set()
    global_matcher = Matcher(nlp.vocab)
    for long_candidate, short_candidate in filtered:
        short, long = find_abbreviation(long_candidate, short_candidate)
        print(f"FIND ABBREV = {short, long}")
        # We need the long and short form definitions to be unique, because we need
        # to store them so we can look them up later. This is a bit of a
        # pathalogical case also, as it would mean an abbreviation had been
        # defined twice in a document. There's not much we can do about this,
        # but at least the case which is discarded will be picked up below by
        # the global matcher. So it's likely that things will work out ok most of the time.
        new_long = long.text not in already_seen_long if long else False
        new_short = short.text not in already_seen_short
        if long is not None and new_long and new_short:
            already_seen_long.add(long.text)
            already_seen_short.add(short.text)
            all_occurences[long].add(short)
            rules[long.text] = long
            # Add a rule to a matcher to find exactly this substring.
            global_matcher.add(long.text, [[{"ORTH": x.text} for x in short]])
    to_remove = set()
    global_matches = global_matcher(doc)
    for match, start, end in global_matches:
        string_key = global_matcher.vocab.strings[match]  # type: ignore
        to_remove.add(string_key)
        all_occurences[rules[string_key]].add(doc[start:end])
    for key in to_remove:
        # Clean up the global matcher.
        global_matcher.remove(key)

    return list((k, v) for k, v in all_occurences.items())


# Load a SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize the DependencyMatcher
matcher = Matcher(nlp.vocab)

# Define the pattern for identifying the abbreviation and its full name
pattern = [{"ORTH": "("}, {"OP": "+"}, {"ORTH": ")"}]

# Add the pattern to the matcher
matcher.add("abbreviation", [pattern])

# Process a sample text
# text = "The first patient suffered from chronic fatigue syndrome (CFS) and had problems sleeping."
# " The climber was found on the mountain experiencing High altitude pulmonary edema, (HAPE)"
text = "1O milimol of phosphate inorganic (PI) as added"
text = text.replace(",", "")
doc = nlp(text)

# Apply the matcher to the document
matches = matcher(doc)
# Remove brackets round the abbreviation
matches_no_brackets = [(x[0], x[1] + 1, x[2] - 1) for x in matches]

filtered = filter_matches(matches_no_brackets, doc)
print(filtered)

if len(matches) > 0:
    abbreviations = find_matches_for(filtered, doc)
    print(abbreviations)
