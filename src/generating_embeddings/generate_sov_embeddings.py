import re
import numpy as np
import pickle
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)


def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    """
    Obtains BERT embeddings for tokens.
    """
    # gradient calculation id disabled
    with torch.no_grad():
      # obtain hidden states
      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]
    # concatenate the tensors for all layers
    # use "stack" to create new dimension in tensor
    token_embeddings = torch.stack(hidden_states, dim=0)
    # remove dimension 1, the "batches"
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # swap dimensions 0 and 1 so we can loop over tokens
    token_embeddings = token_embeddings.permute(1,0,2)
    # initialized list to store embeddings
    token_vecs_sum = []
    # "token_embeddings" is a [Y x 12 x 768] tensor
    # where Y is the number of tokens in the sentence
    # loop over tokens in sentence
    for token in token_embeddings:
    # "token" is a [12 x 768] tensor
    # sum the vectors from the last four layers
        sum_vec = torch.sum(token[-4:], dim=0)
        """
        As an alternative to the sum of the last four layers, one can also concenate the last four layers or extract the second to last layer
        """
        #sum_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        #sum_vec = token[-2]
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum

def bert_text_preparation(text, tokenizer):
  """
  Preprocesses text input in a way that BERT can interpret.
  """
  marked_text = "[CLS] " + text + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1]*len(indexed_tokens)
  # convert inputs to tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensor = torch.tensor([segments_ids])
  return tokenized_text, tokens_tensor, segments_tensor

def wrap_embeddings(texts):
    progress_bar = tqdm(total=len(texts), desc="Generating Embeddings", smoothing=1)
    tokens = []
    embeddings = []
    for text in texts:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
        try:
            list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
        except RuntimeError as e:
            continue

        token_embeddings = sum(list_token_embeddings)
        token_text = ''.join(tokenized_text)
        token_text = re.sub(r'\[[A-Z]+\]|#', '', token_text)
        embeddings.append(token_embeddings)
        tokens.append(token_text)
        progress_bar.update(1)

    progress_bar.close()
    return tokens, embeddings

def clean_verbs(strings):
    english_strings = []
    for string in strings:
        try:
            string.encode('ascii')
        except UnicodeEncodeError:
            continue
        else:
            if all(c.isalnum() for c in string):
                english_strings.append(string)
    return english_strings

filepath = "/Users/lb/projects/cl_chatty/output/data/sov/submissions/sov_submissions.pkl"
with (open(filepath, "rb")) as openfile:
    sov = pickle.load(openfile)

verbs = [d['verb_lemma'] for d in sov]
#verbs = verbs[0:100]
verbs = np.array(verbs)
verbs = np.unique(verbs)
verbs = clean_verbs(verbs)

tokens, embeddings = wrap_embeddings(verbs)
assert len(tokens) == len(embeddings)
verb_dict = {tokens[i]: embeddings[i] for i in range(len(tokens))}

out_path = '../../output/embeddings/verbs_only/verb_embeddings.pkl'
filehandler = open(out_path, 'wb')
pickle.dump(sov, filehandler)