# app/prompts.py

from string import Template

SPELL_CORRECTION_TEMPLATE = Template("""
The following entity name may be misspelled based on the context.
Given the context: "${context}", provide the most likely correct spelling for the entity name: "${entity}".
""")

ENTITY_SUGGESTION_TEMPLATE = Template("""
Given the entity '${entity}' and the context: "${context}", suggest alternative names that might be used to find it on Wikipedia.
""")

SENTIMENT_ANALYSIS_TEMPLATE = Template("""
Analyze the sentiment of the following text and respond in JSON format with a single key "sentiment" whose value is one of "positive", "negative", or "neutral".

Text:
"${text}"
""")

CONCEPT_EXTRACTION_TEMPLATE = Template("""
You are an expert in extracting key concepts from text. Identify the main concepts in the following passage and list them as a JSON array.

Examples:

Text: I enjoy hiking in the mountains during summer.
Concepts: ["hiking", "mountains", "summer"]

Text: The advancements in artificial intelligence are remarkable.
Concepts: ["artificial intelligence", "advancements"]

Text:
"${text}"
Concepts:
""")

ENTITY_SENTIMENT_ANALYSIS_TEMPLATE = Template("""
In the following text, what is the sentiment towards '${entity}'? Respond in JSON format with a single key "sentiment" whose value is one of "positive", "negative", or "neutral".

Text:
"${context}"
""")

RELATION_EXTRACTION_TEMPLATE = Template("""
Extract relationships between the following entities in the text: ${entities}.

Text:
"${text}"

List the relationships:
""")

TRANSLATION_TEMPLATE = Template("""
Translate the following text to ${target_lang}:

"${text}"
""")

# Emotion Analysis Template
EMOTION_ANALYSIS_TEMPLATE = Template("""
Analyze the emotions expressed in the following text and respond in JSON format with a key "emotions" whose value is a list of emotions detected (e.g., "joy", "sadness", "anger").

Text:
"${text}"
Emotions:
""")
