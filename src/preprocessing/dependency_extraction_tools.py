from collections import OrderedDict

def dep_check(to_check, target_terms):
    for token in to_check:
        # Check if the token's text is "chatgpt" or "openai" (case insensitive)
        # and if the token's dependency is one of the specified types
        if token.text.lower() in target_terms and token.dep_ in ["nsubj", "pobj", "dobj", "poss"]:
            return True  # If condition is met, return True

def is_question(doc):
    # Check if the sentence is a question by looking for question marks or question words
    for token in doc:
        if token.text.lower() == "?" or token.text.lower() in ["who", "what", "when", "where", "why", "how"]:
            return True
    return False

def add_tokens(tokens_dict, subtree, exceptions=("_SP",)):
    for token in subtree:
        if token.tag_ not in exceptions and token.i not in tokens_dict:
            tokens_dict[token.i] = token

def object_locator(verb_children, subject):
    # Initialize an OrderedDict to keep tokens unique and maintain their insertion order based on token index
    tokens_dict = OrderedDict()
    for child in verb_children:
        if (child.dep_ in ["dobj", "pobj", "acomp", "prep", "intj", "attr"] or
            (child.dep_ == "nsubj" and child.text != subject.text) or
            (child.dep_ == "ccomp" and child.pos_ in ["AUX", "VERB"])):

            add_tokens(tokens_dict, child.subtree)

            if child.dep_ == "prep":
                for grandchild in child.children:
                    if "obj" in grandchild.dep_:
                        add_tokens(tokens_dict, grandchild.subtree)

        if (child.dep_ == "conj" and child.pos_ == "VERB" and
            any(k.dep_ == "nsubj" and k.text == subject.text and k.i == subject.i for k in child.children)):
            for grandchild in child.children:
                if grandchild.dep_ in ["dobj", "pobj", "acomp", "prep"]:
                    add_tokens(tokens_dict, grandchild.subtree)

    return(tokens_dict)


def extract_object(verb, subject):
    tokens_dict = object_locator(verb_children=verb.rights, subject=subject)

    if len(tokens_dict) == 0:
        tokens_dict = object_locator(verb_children=verb.lefts, subject=subject)

    # Extracting the properties from tokens
    object_texts = [token.text for token in tokens_dict.values()]
    object_deps = [token.dep_ for token in tokens_dict.values()]
    object_tags = [token.tag_ for token in tokens_dict.values()]

    # Joining the lists into strings for the final output
    object_phrase = ' '.join(object_texts).strip()
    object_deps = ' '.join(object_deps)
    object_tags = ' '.join(object_tags)

    return object_phrase, object_deps, object_tags


def extract_verb_phrase(verb, subject):
    # Extract the full verb phrase
    verb_elements = [verb.text]
    verb_elements_indices = [verb.i]
    for child in verb.children:
        if (child.dep_ in ["aux", "auxpass", "neg", "part"] or
                (child.dep_ == "dative" and child.pos_ == "PRON") or
                (child.dep_ == "ccomp" and child.pos_ != "AUX" and any(
                    k.dep_ == "nsubj" and k.text == subject.text for k in child.children)) or
                (child.dep_ == "prt" and child.pos_ == "ADP" and len(list(child.children)) == 0)):
            verb_elements.insert(0, child.text)  # Prepend auxiliary verbs or negations
            verb_elements_indices.insert(0, child.i)

    for ancestor in verb.ancestors:
        if ancestor.dep_ == "ROOT" and verb.dep_ == "xcomp" and any(
                k.dep_ == "nsubj" and k.text == subject.text for k in ancestor.children):
            verb_elements.insert(0, ancestor.text)  # Prepend root verbs to open clausal complements
            verb_elements_indices.insert(0, ancestor.i)

        for sibling in ancestor.children:
            if ancestor.dep_ == "ROOT" and sibling.dep_ == "aux" and any(
                    k.dep_ == "nsubj" and k.text == subject.text for k in ancestor.children):
                verb_elements.insert(0, sibling.text)  # Prepend auxiliary verbs of the root verb
                verb_elements_indices.insert(0, sibling.i)
        """
        # It might be cool to extract the preposition of verbs, but this here is to greedy:
        if child.dep_ in ["dobj"]:
            for grandchild in child.children:
                if grandchild.dep_ in ["prep"]:
                    verb_elements.append(grandchild.text)   # Append prepositions
        """

    # Sort elements by their position in the sentence to maintain the original order
    paired = list(zip(verb_elements_indices, verb_elements))  # Step 1: Zip the indices and elements together
    paired_sorted = sorted(paired)  # Step 2: Sort the pairs
    _, verb_elements_sorted = zip(*paired_sorted)  # Step 3: Unzip to get the sorted elements
    verb_elements_sorted = list(verb_elements_sorted)  # Step 4: Convert the result back to a list
    verb_phrase = " ".join(verb_elements_sorted)  # Step 5: Join the elements to form the verb phrase

    return verb_phrase

def extract_verb_chain(verb, subject):
    extended_verbs = [verb] + \
                     [child for child in verb.children if child.dep_ == "xcomp" and child.pos_ == "VERB"] + \
                     [child for child in verb.children
                      if child.dep_ == "conj" and child.pos_ == "VERB" and
                      any(grandchild.dep_ == "nsubj" and grandchild.text == subject.text for grandchild in
                          child.children)] + \
                     [child for child in verb.children
                      if child.dep_ == "conj" and child.pos_ == "VERB" and
                      not any(grandchild.dep_ == "nsubj" for grandchild in child.children)]
    return extended_verbs
