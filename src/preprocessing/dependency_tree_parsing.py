import zstandard as zstd
import json
import spacy

nlp = spacy.load("en_core_web_sm")

def read_zst_file(file_path):
    with open(file_path, 'rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            decompressed = reader.read().decode('utf-8')
            for line in decompressed.splitlines():
                yield json.loads(line)


def extract_phrase_and_dep(token):
    """Extracts the full phrase related to a given token and its dependency tags."""
    phrase = ' '.join([child.text for child in token.subtree])
    dep_tags = ' '.join([child.dep_ for child in token.subtree])
    return phrase, dep_tags


def extract_svo_and_dep(sent):
    # Initialize placeholders
    subject = verb = object_ = None
    subject_dep = verb_dep = object_dep = None

    # Extract subject, verb, and object
    for token in sent:
        if token.pos_ == "VERB":
            verb = token.text
            verb_dep = token.dep_

            # Once a verb is found, check for connected subject and object
            for child in token.children:
                if "subj" in child.dep_:
                    subject, subject_dep = extract_phrase_and_dep(child)
                elif "obj" in child.dep_:
                    object_, object_dep = extract_phrase_and_dep(child)

    return (subject, verb, object_), (subject_dep, verb_dep, object_dep)

def print_selftext_subject_object_pairs(json_generator, num_entries=6):
    count = 0
    for entry in json_generator:
        if count >= num_entries:
            break
        selftext = entry.get('selftext', None)
        if selftext:
            doc = nlp(selftext)
            for sent in doc.sents:
                text_tuples, dep_tuples = extract_svo_and_dep(sent)
                for pair, deps in zip(text_tuples, dep_tuples):
                    print(f"Entry {count+1}, Sentence: {sent.text}\nSubject-Object Pair: {pair}\nDep: {deps}\n")
        count += 1

# Change 'your_file_path.zst' to the path of your .zst file
file_path = '../../output/data/filtered/submissions/output_submissions.zst'
json_generator = read_zst_file(file_path)

print_selftext_subject_object_pairs(json_generator)

