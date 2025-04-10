#Created on Monday, April 7, 2025 at 10:00 PM
# Import the spacy library for natural language processing (NLP)
import spacy
#The string module provides functions to manage strings (e.g., `string.punctuation` for punctuation removal).
import string
# Imports the type_token_ratio function from the 'ttr' module of the TRUNAJOD library.
from TRUNAJOD.ttr import type_token_ratio
#Importing the CSV module to export the analyzed data to a CSV file.
import csv
#Imports the pyplot module from the matplotlib library.
import matplotlib.pyplot as plt
## Imports the WordCloud class from the wordcloud library.
from wordcloud import WordCloud
#The module identifies the operating system in use and selects the appropriate command to open the CSV file
import platform
# The module executes the necessary system command to open the CSV file with the default application for the operating system.
import subprocess
##------------------------------------------------------##

# Load the Italian language model
nlp = spacy.load("it_core_news_lg")

# Stop words for the Italian language 
italian_stop_words = spacy.lang.it.stop_words.STOP_WORDS

# Cognitive and perceptive lexicon = {
verb_categories = {
    "cognitive": {"pensare", "credenza", "credere", "sapere", "conoscenza", "immaginare", "immaginazione", "valutare", "valutazione", "riflettere", "riflessione", "ricordare", "ricordo"},
    "perceptive": {"vedere", "visione", "guardare", "osservare", "osservazione", "sentire", "udito", "ascoltare", "percepire", "percezione"},
}

# Tokenization 
def analyze_text_spacy(text, italian_stop_words, nlp):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(filtered_tokens)

# Parsing dependencies and identifying the roots of cognitive/perceptive verbs and words.
def analyze_dependencies(doc):
    """Analyzes the syntactic dependencies and extracts relevant information."""
    def extract_dependencies(token):
        token_dependencies = {
            "pronoun_1st_person": [], "implicit_subject": [], "negation_cognitive_perceptive": [],
            "connective": [], "possessive_pronoun": [], "place": [], "person": [],
            "date": [], "adverb": [], "verb": [], "adjective": [],
        }
        if token.pos_ == "PRON" and token.morph.get("Person") == ["1"]:
            token_dependencies["pronoun_1st_person"].append(token.text)
        elif token.dep_ == "expl":
            token_dependencies["implicit_subject"].append(token.text)
        elif token.dep_ == "neg" or token.text.lower() == "non":
            if token.i + 1 < len(doc) and doc[token.i + 1].lemma_ in verb_categories["cognitive"].union(verb_categories["perceptive"]):
                token_dependencies["negation_cognitive_perceptive"].append(token.text)
        elif token.dep_ == "cc":
            token_dependencies["connective"].append(token.text)
        elif token.pos_ == "PRON" and token.morph.get("Poss") == ["Yes"]:
            token_dependencies["possessive_pronoun"].append(token.text)
        elif token.ent_type_ == "LOC":
            token_dependencies["place"].append(token.text)
        elif token.ent_type_ == "PER":
            token_dependencies["person"].append(token.text)
        elif token.ent_type_ == "DATE":
            token_dependencies["date"].append(token.text)
        elif token.pos_ == "ADV":
            token_dependencies["adverb"].append(token.text)
        elif token.pos_ == "VERB":
            token_dependencies["verb"].append(token.text)
        elif token.pos_ == "ADJ":
            token_dependencies["adjective"].append(token.text)
        return token_dependencies

    results = []
    for sent in doc.sents:
        root = [token for token in sent if token.head == token]
        if root:
            root = root[0]
            if root.lemma_ in verb_categories["cognitive"].union(verb_categories["perceptive"]):
                filtered_tokens = [token for token in sent if token.text.lower() not in italian_stop_words and token.text not in string.punctuation and token.pos_ not in ("DET", "ADP", "CCONJ")]
                if filtered_tokens:
                    filtered_root = [token for token in filtered_tokens if token.head == token][0]
                    dependencies = {
                        "pronoun_1st_person": [], "implicit_subject": [], "negation_cognitive_perceptive": [],
                        "connective": [], "possessive_pronoun": [], "place": [], "person": [],
                        "date": [], "adverb": [], "verb": [], "adjective": [],
                    }
                    for token in list(filtered_root.lefts) + list(filtered_root.rights):
                        token_dependencies = extract_dependencies(token)
                        for key, value in token_dependencies.items():
                            if value:
                                dependencies[key].extend(value)
                    results.append((filtered_root, dependencies))
                else:
                    results.append((None, None))
            else:
                results.append((None, None))
        else:
            results.append((None, None))
    return results

##Extracting lemmas of cognitive and perceptive tokens via POS and dependency analysis.
def analyze_cognitive_perceptive_roots(doc):
    """Identifies the roots (verbs or deverbal nouns) of the cognitive and perceptive semantic sphere."""
    cognitive_perceptive_roots = []
    for token in doc:
        if token.lemma_ in verb_categories["cognitive"].union(verb_categories["perceptive"]):
            if token.pos_ == "VERB" or (token.pos_ == "NOUN" and token.dep_ in ["ROOT", "compound", "nsubj", "dobj"]):
                cognitive_perceptive_roots.append(token.lemma_)
    return cognitive_perceptive_roots

def visualize_analysis(doc):
    #Visualization of the analysis of cognitive and perceptive tokens.
    print("\nAnalysis of cognitive and perceptive tokens:")
    for token in doc:
        if token.lemma_ in verb_categories["cognitive"]:
            print(f"Cognitive token: {token.text} (lemma: {token.lemma_})")
        elif token.lemma_ in verb_categories["perceptive"]:
            print(f"Perceptive token: {token.text} (lemma: {token.lemma_})")


    #Type-Token Ratio (TTR).
def calculate_ttr(normalized_text):
    tokens = normalized_text.split()
    return type_token_ratio(tokens)

 
 #Wordcloud.
def generate_wordcloud(normalized_text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(normalized_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
 
 #Barchart.
def generate_simple_bar_chart(roots, ttr):
    cognitive_frequency = {}
    perceptive_frequency = {}
    for root in roots:
        if root in verb_categories["cognitive"]:
            cognitive_frequency[root] = cognitive_frequency.get(root, 0) + 1
        elif root in verb_categories["perceptive"]:
            perceptive_frequency[root] = perceptive_frequency.get(root, 0) + 1

    total_roots = len(roots)
    relative_cognitive_frequency = {root: frequency / total_roots for root, frequency in cognitive_frequency.items()}
    relative_perceptive_frequency = {root: frequency / total_roots for root, frequency in perceptive_frequency.items()}

    cognitive_roots = list(relative_cognitive_frequency.keys())
    cognitive_frequencies = list(relative_cognitive_frequency.values())
    perceptive_roots = list(relative_perceptive_frequency.keys())
    perceptive_frequencies = list(relative_perceptive_frequency.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35

    x_cognitive = range(len(cognitive_roots))
    x_perceptive = [i + bar_width for i in range(len(perceptive_roots))]

    ax.bar(x_cognitive, cognitive_frequencies, bar_width, label='Cognitive', color='lightblue', alpha=0.7)
    ax.bar(x_perceptive, perceptive_frequencies, bar_width, label='Perceptive', color='lightcoral', alpha=0.7)

    ax.set_xlabel('Lemmas')
    ax.set_ylabel('Relative Frequency')
    ax.set_title('Frequency of Cognitive and Perceptive Vocabulary Domain')

    all_labels = cognitive_roots + perceptive_roots
    all_positions = list(x_cognitive) + list(x_perceptive)
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')

    ax.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')

    for i, freq in enumerate(cognitive_frequencies + perceptive_frequencies):
        ax.text(i * bar_width, freq + 0.01, f'{freq:.1%}', ha='center')

    ax.text(0.95, 0.95, f'TTR: {ttr:.3f}', transform=ax.transAxes, ha='right', va='top')
    ax.text(0.01, -0.15, 'Deverbal nouns are also considered in the calculation.', transform=ax.transAxes, ha='left', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


#Exporting CSV file.
def export_table_csv(doc, output_file="formatted_table.csv"):
    data = []
    for token in doc:
        data.append([
            token.i,
            token.text,
            token.lemma_,
            token.pos_,
            token.tag_,
            token.head.i,
            token.dep_,
            token.idx,
            token.idx + len(token.text),
            token.ent_type_ if token.ent_type_ != "" else "_",
            token.morph if token.morph else "_"
        ])

    headers = ["id", "form", "lemma", "upos", "xpos", "head", "deprel", "Start char", "End char", "ner", "feats"]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"Formatted table exported to {output_file}")

#Opens the specified CSV file using the operating system's default application.
def open_csv_file(file_path):
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
        elif platform.system() == "Windows":  # Windows
            subprocess.run(["start", file_path], shell=True, check=True)
        elif platform.system() == "Linux":  # Linux
            subprocess.run(["xdg-open", file_path], check=True)
        else:
            print(f"Unsupported operating system. Unable to open file {file_path}.")
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")

