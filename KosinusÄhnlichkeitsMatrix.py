from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


def read_input_text(file_path):
    file = open(file_path, "r")
    file_data = file.readlines()
    rit_input_text = file_data[0].replace(" - ", " ").replace(".\" ", ". ").replace("?\" ", ". ") \
        .replace("!\" ", ". ").replace("\"", "").split(". ")
    sentences = []
    print("Original Text:")

    for rit_sentence in rit_input_text:
        print(rit_sentence)
        sentences.append(rit_sentence.replace("[^a-ZA-Z]", " ").split(" "))

    sentences.pop()
    return sentences


def sentence_similarity(sentence1, sentence2, stop_words=None):
    if stop_words is None:
        stop_words = []

    sentence1 = [ss_word.lower() for ss_word in sentence1]
    sentence2 = [ss_word.lower() for ss_word in sentence2]

    all_words = list(set(sentence1 + sentence2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for ss_word in sentence1:
        if ss_word in stop_words:
            continue
        vector1[all_words.index(ss_word)] += 1

    for ss_word in sentence2:
        if ss_word in stop_words:
            continue
        vector2[all_words.index(ss_word)] += 1

    print(all_words)
    print(vector1)
    print(vector2)
    return 1 - cosine_distance(vector1, vector2)


def create_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for sentence1 in range(len(sentences)):
        for sentence2 in range(len(sentences)):
            if sentence1 == sentence2:
                continue
            similarity_matrix[sentence1][sentence2] = \
                sentence_similarity(sentences[sentence1], sentences[sentence2], stop_words)

    return similarity_matrix


def create_summary(sentences, summary_length=5):
    stop_words = set(stopwords.words('english'))
    summary = []

    sentence_similarity_matrix = create_similarity_matrix(sentences, stop_words)

    sentences_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentences_similarity_graph)

    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    for cs_sentence in sentences:
        for i in range(summary_length):
            if cs_sentence == ranked_sentence[i][1]:
                summary.append(cs_sentence)

    return summary


input_text = read_input_text(r"Fariytale_Texts/The_Wolf_And_The_Seven_Little_Kids_NoNewLine.txt")
sentence = ""
input_text = create_summary(input_text, int(len(input_text) / 4))

print("Summarized Text:")
for sentence in input_text:
    for word in sentence:
        print(" ", end='')
        print(word, end='')
    print(".", end='')

print("\n Summarized Text Length: " + str(len(input_text)))
