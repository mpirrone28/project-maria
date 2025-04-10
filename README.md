
**Project Description:**

This project performs text analysis in Italian using the SpaCy library. The pipeline includes text preprocessing (tokenization, stop word removal, analysis of syntactic dependencies) and semantic analysis related to cognitive and perceptive verbs. The visualization of the analyzed data leverages WordCloud and bar charts to explore the relative frequency of cognitive and perceptive lemmas and verbs in the analyzed text. Finally, the results can be exported to a CSV file named `formatted_table.csv`.

To automatically open the CSV file, the code checks the operating system in use and selects the appropriate command.

**Requirements:**

To run this code, you need Python 3.x and the following libraries installed:

* SpaCy (for natural language processing)
* Matplotlib (for generating graphs)
* Wordcloud (for generating the word cloud)
* TRUNAJOD (for calculating the Type-Token Ratio, TTR)
* Csv (for exporting results)

You can install the dependencies using the pip command:

```bash
pip install spacy matplotlib wordcloud TRUNAJOD
Furthermore, it is necessary to download the SpaCy language model for Italian:

Bash

python -m spacy download it_core_news_lg
Usage:

Input text: it is possible to provide it directly in the code or as external input.

Analysis execution: the code will execute the process of tokenization, stop word removal, analysis of syntactic dependencies, and identification of cognitive and perceptive verbs.

Visualization of results:

A WordCloud will be generated to visualize the most frequent words.
A bar chart will show the relative frequency of cognitive and perceptive verbs and lemmas.
Export of results: The details of the analyzed data are exported to a CSV file, which can be viewed through the default application of the operating system.
