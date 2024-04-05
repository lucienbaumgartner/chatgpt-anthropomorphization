import zstandard as zstd
import json
import spacy
from spacy.language import Language
import re
from tqdm import tqdm
import language_tool_python
from langdetect import detect

def read_zst_file(file_path):
    with open(file_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            decompressed = reader.read().decode('utf-8')
            for line in decompressed.splitlines():
                yield json.loads(line)
@Language.component("custom_sentencizer")
#### this has to be adapted
def custom_sentecizer(doc):
    # Iterate through token indices
    for i, token in enumerate(doc):
        if token.text == "\n":
            # If the current token is a newline, mark the next token as the start of a sentence
            doc[i].is_sent_start = False  # The newline itself is not a sentence start
            if i + 1 < len(doc):
                doc[i + 1].is_sent_start = True
    return doc

def preprocess_text(text):
    # Remove specific patterns with regex
    if detect(text) != 'en':
        return None
    #text = tool.correct(text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\b\w*;#x\w*\b(;)?", "", text)
    return text

def extract_phrase_and_dep(token):
    """Extracts the full phrase related to a given token and its dependency tags."""
    phrase = ' '.join([child.text for child in token.subtree])
    dep_tags = ' '.join([child.dep_ for child in token.subtree])
    return phrase, dep_tags

def extract_subject_verb(sentence, target_terms):

    # Find the subject and verb associated with it
    subject = None
    verb = None
    #verbs = []

    for token in sentence:
        if token.dep_ == "nsubj" and token.text.lower() in target_terms:
            subject = token
            verb = token.head
            for child in verb.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    verb = child
                    break

            if verb.lemma_ == "have":
                for token in sentence:
                    if token.dep_ == "dobj":
                        for child in token.children:
                            if child.dep_ == "relcl" and child.pos_ == "VERB":
                                verb = child
                                #verbs = [verb.lemma_]
                                break

    # Check if the verb is in active form and not part of a question
    if subject and (verb.dep_ in ["ROOT", "xcomp", "relcl"]) and verb.pos_ == "VERB" and not is_question(sentence):
        return subject.text, verb.text, verb.lemma_, verb.dep_
    else:
        return None, None, None, None

def process_text(text):
    # Check if the text is a float
    if isinstance(text, float):
        # Convert float to string
        text = str(text)

    # Process the text using spaCy
    text = preprocess_text(text)
    if text is not None:
        doc = nlp(text)
        return doc

def is_question(doc):
    # Check if the sentence is a question by looking for question marks or question words
    for token in doc:
        if token.text.lower() == "?" or token.text.lower() in ["who", "what", "when", "where", "why", "how"]:
            return True
    return False

def filter_texts(texts, target_terms):
    filtered_texts = []
    for text in tqdm(texts, desc="Filtering texts"):
        doc = process_text(text["selftext"])
        if doc is None:
            continue
        else:
            for sentence in doc.sents:
                #if re.match(r"\-", sentence[0].text):
                #    continue
                #else:
                subject, verb, verb_lemma, dep = extract_subject_verb(sentence, target_terms)
                if subject and verb != "ai":
                    filtered_texts.append((sentence, subject, verb, verb_lemma, dep))
    return filtered_texts

nlp = spacy.load("en_core_web_sm", disable=["ner"])
# Check if the parser exists in the pipeline and add the custom segmenter before it
if "parser" in nlp.pipe_names:
    nlp.add_pipe("custom_sentencizer", before="parser")
else:
    # If there's no parser in the pipeline, just add the segmenter at the end
    nlp.add_pipe("custom_sentencizer", last=True)

tool = language_tool_python.LanguageTool('en-US')

target_terms = ["chatgpt", "openai"]
file_path = '../../output/data/filtered/submissions/output_submissions.zst'
json_generator = read_zst_file(file_path)
filtered_texts = filter_texts(json_generator, target_terms)
print(filtered_texts[0:6])