import torch
from transformers import DistilBertModel, DistilBertTokenizer
import spacy
from spacy.tokens import Doc
import pickle
import re
from tqdm import tqdm
import h5py
import numpy as np

def map_spacy_distilbert_tokens(spacy_doc, distilbert_tokens, print_=False):
    spacy_to_distilbert_map = []
    distilbert_token_index = 0

    # Concatenate all RoBERTa tokens to match against text
    distilbert_text = ''.join([t.replace('##', ' ') if t.startswith('##') else t for t in distilbert_tokens])
    spacy_text_index = 0

    for token in spacy_doc:

        # Normalize spaCy token text for comparison
        spacy_token_text = token.text.replace("’", "'")

        # Find the start of the current spaCy token in the concatenated RoBERTa text
        while distilbert_text[spacy_text_index:spacy_text_index + len(spacy_token_text)] != spacy_token_text:
            spacy_text_index += 1
            if spacy_text_index >= len(distilbert_text):  # Safety condition
                break

        # Find all RoBERTa tokens that overlap with this spaCy token
        start_index = spacy_text_index
        end_index = spacy_text_index + len(token.text)
        current_distilbert_indices = []

        while distilbert_token_index < len(distilbert_tokens) and start_index < end_index:
            current_token = distilbert_tokens[distilbert_token_index]
            current_token_length = len(current_token.replace('##', ' '))
            token_start = start_index
            token_end = start_index + current_token_length

            # Check if the current RoBERTa token overlaps with the spaCy token
            if token_start < end_index and token_end > start_index:
                current_distilbert_indices.append(distilbert_token_index)
            start_index += current_token_length
            distilbert_token_index += 1

        # Map the spaCy token to the list of overlapping RoBERTa token indices
        spacy_to_distilbert_map.append(current_distilbert_indices)
        spacy_text_index += len(token.text)

    if print_:
        print_token_mapping(spacy_doc, distilbert_tokens, spacy_to_distilbert_map)

    return spacy_to_distilbert_map


def print_token_mapping(spacy_doc, distilbert_tokens, mapping):
    print("spaCy Token to RoBERTa Token Mapping:\n")
    for token, indices in zip(spacy_doc, mapping):
        distilbert_token_group = [distilbert_tokens[idx] for idx in indices]
        print(f"spaCy Token: '{token.text}' -> DistilBERT Tokens: {distilbert_token_group}")


def extract_embeddings_by_indices(texts_list, indices_list, print_=False):
    try:
        assert len(texts_list) == len(indices_list)
    except:
        print("Texts and indices do not have the same length.")

    hdf5_file = '/Users/lb/projects/cl_chatty/output/embeddings/sov/submissions/sov_embeddings.h5'
    with h5py.File(hdf5_file, 'w') as f:
        # Initialize an empty dataset with unlimited size and the shape of embeddings
        # Adjust the shape according to your embedding dimension
        max_shape = (None, model.config.hidden_size)
        dset = f.create_dataset('embeddings', shape=(0, model.config.hidden_size), maxshape=max_shape, dtype='float32', compression='gzip', compression_opts=9)

        for text, indices in tqdm(zip(texts_list, indices_list), total=len(texts_list)):

            # Process text with both spaCy and RoBERTa
            if (isinstance(text, Doc)):
                spacy_doc = text
                text = ' '.join([token.text_with_ws for token in spacy_doc])
                text = re.sub(r'\s+', ' ', text)
            else:
                spacy_doc = nlp(text)

            encoded_input = tokenizer(text, return_tensors='pt')
            outputs = model(**encoded_input)

            # Get the last hidden states (embeddings)
            embeddings = outputs.last_hidden_state.squeeze(0)

            # Map spaCy tokens to RoBERTa tokens
            try:
                mapping = map_spacy_distilbert_tokens(spacy_doc, tokenizer.tokenize(text), print_=print_)
            except:
                continue

            # Aggregate embeddings for the provided indices
            aggs = aggregate_embeddings(embeddings, mapping, indices)

            # Extend the dataset and append new data
            new_len = dset.shape[0] + 1
            dset.resize(new_len, axis=0)
            dset[-1] = aggs

def aggregate_embeddings(embeddings, mapping, indices):
    # Initialize an empty list to store all the embeddings for the sentence
    sentence_embeddings = []

    # Collect all relevant token embeddings according to the indices
    for idx in indices:
        distilbert_indices = mapping[idx]
        token_embeddings = embeddings[distilbert_indices]  # This extracts all sub-token embeddings
        sentence_embeddings.extend(token_embeddings)  # Extend the list with these embeddings

    # Convert the list of all token embeddings to a tensor
    sentence_embeddings = torch.stack(sentence_embeddings)

    # Compute the mean across all token embeddings to get a single sentence embedding
    aggregated_sentence_embedding = torch.mean(sentence_embeddings, dim=0).detach().numpy()

    return aggregated_sentence_embedding


filepath = "/Users/lb/projects/cl_chatty/output/data/sov/submissions/sov_submissions.pkl"
with (open(filepath, "rb")) as openfile:
    sov = pickle.load(openfile)

nlp = spacy.load("en_core_web_sm", disable=["ner"])

# Load RoBERTa model
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

texts_list, verb_phrase, indices_list, created_utc = zip(
    *[(element["sentence"], element["verb_phrase"], element["verb_phrase_indices"], element["created_utc"]) for element
      in sov])

extract_embeddings_by_indices(texts_list=texts_list, indices_list=indices_list, print_=False)

