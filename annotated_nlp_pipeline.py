

''''This document contains a program suitable for the analysis of a non-standard Italian variety, aiming to examine the category of psychological verbs.''.
# -*- coding: utf-8 -*-
"""
Annotated Italian NLP Pipeline
This script reads an Italian text, cleans and preprocesses it, segments it semantically,
extracts linguistic features, computes TTR and TF-IDF, visualizes results, and identifies
psychological phenomena.
"""

# 1. Mount Google Drive (Colab) for file access
from google.colab import drive
drive.mount('/content/drive')

# 2. Install required libraries and models (run once per session)
#    - nltk: classic NLP toolkit
#    - spacy: industrial-strength NLP library
#    - it_core_news_lg: large Italian model
#    - scikit-learn: machine learning utilities
#    - sentence-transformers: SBERT embeddings
#    - wordcloud: generate word clouds
!pip install nltk
!pip install spacy
!python -m spacy download it_core_news_lg
!pip install scikit-learn
!pip install sentence-transformers
!pip install wordcloud

# 3. Import libraries and set configurations
import os, re, string
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
from IPython.display import display

# Paths and segmentation parameters
TXT_FILE_PATH = '/path/to/your/data'
OUTPUT_DIR = '/path/to/your/output/'
MIN_TOKENS_SEGMENT = 1000    # minimum tokens per segment
SEMANTIC_THRESH = 0.75       # SBERT similarity threshold
CTX_WINDOW = 50              # context window for similarity checks

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load spaCy Italian model
try:
    nlp = spacy.load("it_core_news_lg")
except OSError:
    spacy.cli.download("it_core_news_lg")
    nlp = spacy.load("it_core_news_lg")

# Load SBERT model for semantic similarity
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 4. Read and clean the input text
try:
    with open(TXT_FILE_PATH, 'r', encoding='utf-8') as f:
        testo_estratto = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {TXT_FILE_PATH}")

def clean_text(text: str) -> str:
    """
    Remove extra whitespace, apostrophes, punctuation, and lowercase everything.
    """
    txt = " ".join(text.split())
    txt = txt.replace("'", " ")
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    return txt.lower()

clean = clean_text(testo_estratto)
display(clean)

# 5. Initial tokenization with NLTK for Italian
tokens_nltk = word_tokenize(clean, language='italian')
print(f"Total tokens before filtering: {len(tokens_nltk)}")

# 6. POS-tagging with spaCy (coarse and fine-grained tags)
doc_pos = nlp(" ".join(tokens_nltk))
pos_tags = [(t.text, t.pos_, t.tag_) for t in doc_pos]
print("Sample POS tags:", pos_tags[:10])

# 7. Build custom stopword list and filter tokens
custom_stop = {
    # discourse markers, abbreviations, slang, standard stopwords...
    "cioè","insomma","allora","dunque","praticamente","ovviamente","cio","insomm",
    # (list continues) ...
}
nltk_sw = set(stopwords.words('italian'))
all_stop = nltk_sw.union(custom_stop)

filtered_tokens = []
for tok in tokens_nltk:
    low = tok.lower()
    if low in all_stop:
        # keep verbs even if in stopword list
        if nlp(low)[0].pos_ == 'VERB':
            filtered_tokens.append(tok)
    else:
        filtered_tokens.append(tok)

print(f"Total tokens after filtering: {len(filtered_tokens)}")

# 8. Semantic segmentation using chunking and SBERT similarity
grammar = r"""
  NP: {<DET>?<ADJ>*<NOUN>+}
  VP: {<VERB>+<ADV>*}
"""
chunker = RegexpParser(grammar)
doc_seg = nlp(clean)
# Extract token POS and index
pos_idx = [(t.text, t.pos_, t.i) for t in doc_seg if not t.is_space]
tree = chunker.parse(pos_idx)
chunk_ends = {
    leaves[-1][2]
    for sub in tree.subtrees() if sub.label_ in ("NP","VP")
    for leaves in [sub.leaves()]
}

segments = []
buf, cnt, sid = [], 0, 1
for tok, pos, idx in pos_idx:
    buf.append(tok); cnt += 1
    if cnt >= MIN_TOKENS_SEGMENT and idx in chunk_ends:
        # compute semantic shift
        pre = " ".join(buf[-CTX_WINDOW:])
        next_tokens = [t.text for t in doc_seg if t.i > idx][:CTX_WINDOW]
        sim = torch.nn.functional.cosine_similarity(
            sbert.encode(pre, convert_to_tensor=True).unsqueeze(0),
            sbert.encode(" ".join(next_tokens), convert_to_tensor=True).unsqueeze(0),
            dim=1
        ).item()
        if sim < SEMANTIC_THRESH:
            segments.append((sid, buf.copy()))
            sid += 1; buf = []; cnt = 0
if buf:
    segments.append((sid, buf.copy()))

# 9. Reconstruct sentences and build DataFrame
rows = []
gid = 1
for seg_id, toks in segments:
    sent_buf = []
    for w in toks:
        sent_buf.append(w)
        if w in {'.','!','?'}:
            sent = ' '.join(sent_buf).strip(' .!?')
            rows.append({'row_number': gid, 'segment_id': seg_id, 'sentence_text': sent})
            gid += 1; sent_buf = []
    if sent_buf:
        sent = ' '.join(sent_buf)
        rows.append({'row_number': gid, 'segment_id': seg_id, 'sentence_text': sent})
        gid += 1

df_segments = pd.DataFrame(rows)
df_segments['sentence_index_in_segment'] = df_segments.groupby('segment_id').cumcount()+1
df_segments['rows_per_segment'] = df_segments.groupby('segment_id')['sentence_text'].transform('count')

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_segments.to_csv(os.path.join(OUTPUT_DIR,'segments_with_sbert.csv'), index=False)

# 10. Lemmatization of NLTK tokens using spaCy in batch mode
lemmi_da_nltk = []
for doc_tok in nlp.pipe(tokens_nltk, disable=["parser","ner"]):
    lemmi_da_nltk.append(doc_tok[0].lemma_)
print(f"Unique lemmas: {len(set(lemmi_da_nltk))}")

# 11. Compute Type-Token Ratios (TTR)
words = clean.split()
filtered_words = [w for w in words if w.lower() not in all_stop]
ttr_fleshi = len(set(filtered_words)) / len(filtered_words)
ttr_lemmi = len(set(lemmi_da_nltk)) / len(lemmi_da_nltk)
print(f"TTR fleshed forms: {ttr_fleshi:.4f}")
print(f"TTR lemmas: {ttr_lemmi:.4f}")

# 12. TF-IDF keyword extraction (nouns, pronouns, proper nouns)
pos_filters = ["NOUN","PROPN","PRON"]
pronomini = ["io","tu","noi","voi"]
tok_val = [
    token.lemma_.lower() for token in nlp(clean)
    if token.pos_ in pos_filters and token.is_alpha
]
vec = TfidfVectorizer()
X = vec.fit_transform([' '.join(tok_val)])
df_tfidf = pd.DataFrame({'Parola': vec.get_feature_names_out(), 'TFIDF': X.toarray()[0]})
df_tfidf = df_tfidf[~df_tfidf['Parola'].isin(exclude)].sort_values('TFIDF', ascending=False)

# 13. Plot top TF-IDF scores
top30 = df_tfidf.head(30).reset_index()
top30.plot(kind='scatter', x='index', y='TFIDF', s=32, alpha=0.8)
plt.gca().spines[['top','right']].set_visible(False)
plt.title("TF-IDF Distribution")
plt.show()

# 14. Generate word cloud of keywords
wc_freq = dict(zip(df_tfidf['Parola'].head(100), df_tfidf['TFIDF'].head(100)))
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wc_freq)
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')

# 15. Extract psychological phenomena via regex
verbi = ["amare","piacere","temere","ricordare"]  # etc.
verb_patterns = {v: re.compile(rf"(?u)\b{v}\w*\b", re.IGNORECASE) for v in verbi}
records = []
for seg_id, toks in segments:
    text = ' '.join(toks)
    for lemma, pat in verb_patterns.items():
        for m in pat.finditer(text):
            records.append({
                "Segment ID": seg_id,
                "Pattern": lemma,
                "Match": m.group(),
                "Position": m.start()
            })
df_psych = pd.DataFrame(records)
df_psych.to_csv(os.path.join(OUTPUT_DIR,'psychological_events.csv'), index=False)

# 16. Deduplicate and compute relative frequencies
df_unique = df_psych.drop_duplicates(subset=["Segment ID","Pattern","Match"])
freq_rel = df_unique['Pattern'].value_counts(normalize=True).round(2).reset_index(name='RelFreq')
freq_rel.to_csv(os.path.join(OUTPUT_DIR,'psych_freq_rel.csv'), index=False)
