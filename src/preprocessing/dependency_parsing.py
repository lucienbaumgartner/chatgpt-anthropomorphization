import spacy
import spacy_experimental

nlp = spacy.load("en_core_web_sm")
nlp_coref = spacy.load("en_coreference_web_trf")

# Use replace_listeners for the coref components
nlp_coref.replace_listeners("transformer", "coref", ["model.tok2vec"])
nlp_coref.replace_listeners("transformer", "span_resolver", ["model.tok2vec"])

# We won't copy over the span cleaner
nlp.add_pipe("coref", source=nlp_coref)
nlp.add_pipe("span_resolver", source=nlp_coref)


def resolve_corefs_and_reconstruct_text(doc):
    # Initialize a list to hold the new document text, preserving the original structure
    new_doc_text = list(doc.text)

    # Iterate over span groups for coreference clusters
    for cluster_key in doc.spans:
        if 'coref_clusters' in cluster_key:
            # Assuming the first span in the group is the most representative
            representative_span = doc.spans[cluster_key][0]
            for span in doc.spans[cluster_key]:
                if span != representative_span:
                    # Calculate the start and end positions in the original text
                    start_pos = span.start_char
                    end_pos = span.end_char

                    # Replace the mention in the original text with the representative mention
                    # Adjust the text only if the lengths differ to prevent unnecessary operations
                    if len(representative_span.text) != len(span.text):
                        new_doc_text[start_pos:end_pos] = representative_span.text + ' ' * (
                                    end_pos - start_pos - len(representative_span.text))
                    else:
                        new_doc_text[start_pos:end_pos] = representative_span.text

    # Reconstruct the text from the list of characters
    resolved_text = ''.join(new_doc_text).replace('  ', ' ')
    return resolved_text.strip()


# Example usage
text = "The cats were startled by the dog as it growled at them."
doc = nlp(text)

resolved_text = resolve_corefs_and_reconstruct_text(doc)
print(resolved_text)
