import zstandard as zstd
import json
import spacy
from spacy.language import Language
import re
from tqdm import tqdm
import language_tool_python
from langdetect import detect
import pickle

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
        if re.search(r"\n+|\s+", token.text):
            # If the current token is a newline, mark the next token as the start of a sentence
            doc[i].is_sent_start = False  # The newline itself is not a sentence start
            if i + 1 < len(doc):
                doc[i + 1].is_sent_start = True
    return doc

def preprocess_text(text):
    # Remove specific patterns with regex
    try:
        if detect(text) != 'en':
            return None
    except:
        return None

    #text = tool.correct(text)
    ## SLOT IN MORE PREPROCESSING!!!
    text = text.lower()
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\b\w*;\w*\b(;)?", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\*|•|#", "", text)
    return text

def extract_sov(sentence, target_terms):
    # Find the subject and verb associated with it

    svo_combinations = []

    subject = verb = object_ = None
    object_phrase = None
    # verbs = []

    for token in sentence:
        if token.dep_ == "nsubj":
            subject = token
            subject_det = any(token.pos_ == "DET" for token in subject.children)
            verb = token.head

            for child in verb.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    verb = child
                    break

            # Check for extended verb clauses
            extended_verbs = ([verb] +
                              [child for child in verb.children if child.dep_ == "xcomp" and child.pos_ == "VERB"] +
                              [child for child in verb.children if child.dep_ == "conj" and child.pos_ == "VERB"]
                              )

            for verb in extended_verbs:
                if verb.lemma_ == "have":
                    for token in sentence:
                        if token.dep_ == "dobj":
                            for child in token.children:
                                if child.dep_ == "relcl" and child.pos_ == "VERB":
                                    verb = child
                                    break

                verb_negated = any(token.dep_ == "neg" for token in verb.children)

                # Now, find the object related to the verb
                for child in verb.children:
                    # Look for direct objects or objects of prepositions
                    if child.dep_ in ["dobj", "pobj", "acomp", "prep"]:
                        object_ = child
                        # Extract the full object phrase by traversing the subtree of the object token
                        object_phrase = ' '.join([token.text for token in object_.subtree]).strip()
                        object_deps = ' '.join([token.dep_ for token in object_.subtree if token.tag_ != "_SP"])
                        object_tags = ' '.join([token.tag_ for token in object_.subtree if token.tag_ != "_SP"])
                        if (object_.dep_ == "prep"):
                            for grandchild in object_.children:
                                if ("obj" in grandchild.dep_):
                                    object_ = grandchild
                        break

                # Check if the verb is in active form and not part of a question
                if verb.dep_ in ["ROOT", "xcomp", "relcl", "conj"] and verb.pos_ in ["VERB", "AUX"]:
                    #
                    # Check if subject is not None and then proceed
                    if subject is not None:
                        subject_text = subject.text
                    else:
                        subject_text = None  # Or some default value or logic as per requirements

                    # Similarly, handle object_phrase being None
                    if object_phrase is not None:
                        object_phrase_text = object_phrase
                    else:
                        object_phrase_text = None  # Or some default value or logic as per requirements

                    # Now, modify the condition to ensure it works even if subject or object_phrase are None
                    # This is done by checking for the presence of target terms in a way that avoids NoneType errors
                    if any(x for x in [subject_text, object_phrase_text] if
                           x is not None and any(s in x for s in target_terms)) and not is_question(sentence):
                        # Initialize object_ related variables to avoid NameError
                        object_text = object_.text if object_ is not None else None
                        object_lemma = object_.lemma_ if object_ is not None else None
                        object_dep = object_deps if 'object_deps' in locals() or 'object_deps' in globals() else None
                        object_tag = object_tags if 'object_tags' in locals() or 'object_tags' in globals() else None

                        svo_combinations.append({
                            "subject": subject_text,
                            "subject_det": subject_det,
                            "subject_pos": subject.pos_,
                            "verb": verb.text,
                            "verb_lemma": verb.lemma_,
                            "verb_dep": verb.dep_,
                            "verb_tag": verb.tag_,
                            "verb_negated": verb_negated,
                            "object": object_text,
                            "object_lemma": object_lemma,
                            "object_dep": object_dep,
                            "object_tag": object_tag,
                            "object_phrase": object_phrase_text
                        })
    if svo_combinations != []:

        return svo_combinations

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

def dep_check(to_check, target_terms):
    for token in to_check:
        # Check if the token's text is "chatgpt" or "openai" (case insensitive)
        # and if the token's dependency is one of the specified types
        if token.text.lower() in target_terms and token.dep_ in ["nsubj", "pobj", "dobj"]:
            return True  # If condition is met, return True

def is_question(doc):
    # Check if the sentence is a question by looking for question marks or question words
    for token in doc:
        if token.text.lower() == "?" or token.text.lower() in ["who", "what", "when", "where", "why", "how"]:
            return True
    return False

def generate_sov(texts, target_terms):
    filtered_texts = []
    for text in tqdm(texts, desc="Generate Subject-Object-Verb Data"):
        doc = process_text(text["selftext"])
        if doc is None:
            continue
        else:
            for sentence in doc.sents:
                #print(sentence)
                if not any(token.text in target_terms for token in sentence) or not dep_check(sentence, target_terms):
                    continue
                else:
                    result = extract_sov(sentence, target_terms)
                    if result != None:
                        result = [dict(item, **{"created_utc": text["created_utc"], "sentence": sentence.text.strip()}) for item in result]
                        filtered_texts.append(result)
    filtered_texts = [x for xs in filtered_texts for x in xs]
    return filtered_texts

nlp = spacy.load("en_core_web_sm", disable=["ner"])
# Check if the parser exists in the pipeline and add the custom segmenter before it
if "parser" in nlp.pipe_names:
    nlp.add_pipe("custom_sentencizer", before="parser")
else:
    # If there's no parser in the pipeline, just add the segmenter at the end
    nlp.add_pipe("custom_sentencizer", last=True)

#tool = language_tool_python.LanguageTool('en-US')

target_terms = ["chatgpt", "openai"]
file_path = '../../output/data/reddit-query/submissions/output_submissions.zst'
json_generator = read_zst_file(file_path)
sov = generate_sov(json_generator, target_terms)

out_path = '../../output/data/sov/submissions/sov_submissions.pkl'
filehandler = open(out_path, 'wb')
pickle.dump(sov, filehandler)