from collections import defaultdict
from typing import Dict, List
import spacy
from spellchecker import SpellChecker

# Global workers
_worker_nlp = None
_worker_spell_checker = None

def initialize_worker():
    global _worker_nlp
    global _worker_spell_checker
    if _worker_nlp is None:
        _worker_nlp = spacy.load("en_core_web_sm")
    if _worker_spell_checker is None:
        _worker_spell_checker = SpellChecker()

def preprocess_text_worker(text: str) -> str:
    global _worker_nlp
    global _worker_spell_checker
    if _worker_nlp is None or _worker_spell_checker is None:
        initialize_worker()
    tokens = _worker_nlp(text)
    corrected_tokens = []
    for idx, token in enumerate(tokens):
        if token.is_alpha and not token.ent_type_:
            corrected_word = _worker_spell_checker.correction(token.text)
            if corrected_word is None:
                corrected_word = token.text
            corrected_tokens.append(corrected_word)
        else:
            token_text = token.text if token.text is not None else ""
            corrected_tokens.append(token_text)
    corrected_text = ' '.join(corrected_tokens)
    return corrected_text

def extract_and_resolve_entities_worker(text: str) -> List[Dict[str, str]]:
    global _worker_nlp
    if _worker_nlp is None:
        initialize_worker()
    relevant_entity_types = {'PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT'}
    doc = _worker_nlp(text)
    entities = list(set([ent.text for ent in doc.ents if ent.label_ in relevant_entity_types]))
    entity_groups = defaultdict(list)
    for entity in entities:
        key = ''.join(e.lower() for e in entity if e.isalnum())
        entity_groups[key].append(entity)
    resolved_entities = []
    for group in entity_groups.values():
        primary_entity = max(group, key=len)
        doc_primary = _worker_nlp(primary_entity)
        if doc_primary.ents:
            entity_type = doc_primary.ents[0].label_
        else:
            entity_type = 'UNKNOWN'  # Set to 'UNKNOWN' if no entity type found
        resolved_entities.append({"text": primary_entity, "type": entity_type})
    return resolved_entities
