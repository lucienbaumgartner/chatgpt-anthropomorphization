import spacy
import re
from tqdm import tqdm
import pickle
import itertools
from dependency_extraction_tools import dep_check, is_question, extract_object, extract_verb_phrase, extract_verb_chain
from data_processing import read_zst_file, process_text, custom_sentecizer

def extract_sov(sentence, target_terms):

    svo_combinations = []

    for token in sentence:
        # Find the subject and verb associated with it
        if token.dep_ == "nsubj":
            subject = token
            subject_det = any(token.pos_ == "DET" for token in subject.children)
            subject_phrase = ' '.join([token.text for token in subject.subtree]).strip()
            verb = token.head

            for child in verb.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    verb = child
                    break

            # Get all connected verb chains
            verb_chain = extract_verb_chain(verb, subject)
            #print(f'extended verb list: {extended_verbs}')

            # Loop over verbs in the verb chain
            for verb in verb_chain:
                # Resolve auxiliary verbs
                if verb.lemma_ in ["have", "be"] and verb.pos_ != "AUX":
                    for token in sentence:
                        if token.dep_ == "dobj":
                            for child in token.children:
                                if child.dep_ == "relcl" and child.pos_ == "VERB":
                                    verb = child
                                    break

                # Skip weird verbs
                if verb.text in ["openai", "ai"]:
                    continue

                # Annotate whether the verb is negated
                verb_negated = any(token.dep_ == "neg" for token in verb.children)

                # Extract verb phrase
                verb_phrase = extract_verb_phrase(verb, subject)

                # Now, find the object related to the verb
                object_phrase, object_deps, object_tags = extract_object(verb, subject)

                # Check if the verb is in active form and not part of a question
                if verb.dep_ in ["ROOT", "xcomp", "ccomp", "relcl", "conj", "aux", "advcl"] and verb.pos_ in ["VERB", "AUX"]:

                    if any(x for x in [subject.text, object_phrase] if
                           x is not None and any(s in x for s in target_terms)):
                        # Create sentence as string with the token whitespace
                        sentence_tokenized = ' '.join([token.text_with_ws for token in sentence])
                        sentence_tokenized = re.sub(r'\s+', ' ', sentence_tokenized)

                        svo_combinations.append({
                            "subject": subject.text,
                            "subject_det": subject_det,
                            "subject_pos": subject.pos_,
                            "subject_phrase": subject_phrase,
                            "verb": verb.text,
                            "verb_lemma": verb.lemma_,
                            "verb_dep": verb.dep_,
                            "verb_tag": verb.tag_,
                            "verb_negated": verb_negated,
                            "verb_phrase": verb_phrase,
                            "object_phrase": object_phrase,
                            "object_dep": object_deps,
                            "object_tag": object_tags,
                            "sentence": sentence_tokenized
                        })

    if svo_combinations != []:

        return svo_combinations

def generate_sov(texts, target_terms,subsample=None):
    filtered_texts = []
    if subsample != None:
        texts = itertools.islice(texts, subsample)

    for text in tqdm(texts, desc="Generate Subject-Object-Verb Data"):
        doc = process_text(text["selftext"], nlp=nlp)
        if doc is None:
            continue
        else:
            for sentence in doc.sents:
                #print(sentence)
                if not any(token.text in target_terms for token in sentence) or not dep_check(sentence, target_terms) or is_question(sentence):
                    continue
                else:
                    result = extract_sov(sentence, target_terms)
                    if result != None:
                        sentence_tokenized = ' '.join([token.text_with_ws for token in sentence])
                        sentence_tokenized = re.sub(r'\s+', ' ', sentence_tokenized)
                        result = [dict(item, **{"created_utc": text["created_utc"], "sentence": sentence_tokenized}) for item in result]
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

target_terms = ["chatgpt", "openai"]

# Create generator to read in json data
file_path = '../../output/data/reddit-query/submissions/output_submissions.zst'
json_generator = read_zst_file(file_path)

# Extract SOV structures
sov = generate_sov(json_generator, target_terms)

# Remove duplicates
sov = list({frozenset(item.items()): item for item in sov}.values())

# Write out data
out_path = '../../output/data/sov/submissions/sov_submissions.pkl'
filehandler = open(out_path, 'wb')
pickle.dump(sov, filehandler)