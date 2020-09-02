import nltk
import re
import math
import operator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

stopwords = set(stopwords.words('english'))
word_lemmatizer = WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()


def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(word_lemmatizer.lemmatize(word))
    return lemmatized_words


def stem_words(words):
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def remove_special_characters(rsc_text):
    regex = r'[^a-zA-Z0-9\s]'
    rsc_text = re.sub(regex, '', rsc_text)
    return rsc_text


def frequency(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        dict_freq[word] = words.count(word)
    return dict_freq


def pos_tagging(pt_text):
    pos_tag = nltk.pos_tag(pt_text.split())
    pos_tagged_noun_verb = []
    for word, tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
            pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb


def tf_score(word, tf_sentence):
    frequency_sum = 0
    word_frequency_in_sentence = 0
    sentence_length = len(tf_sentence)
    for word_in_sentence in tf_sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf = word_frequency_in_sentence / sentence_length
    return tf


def idf_score(idf_number_sentences, word, sentences):
    number_sentence_containing_word = 0
    for idf_sentence in sentences:
        idf_sentence = remove_special_characters(str(idf_sentence))
        idf_sentence = idf_sentence.split()
        idf_sentence = [word for word in idf_sentence if word.lower() not in stopwords and len(word) > 1]
        idf_sentence = [word.lower() for word in idf_sentence]
        idf_sentence = [word_lemmatizer.lemmatize(word) for word in idf_sentence]
        if word in idf_sentence:
            number_sentence_containing_word = number_sentence_containing_word + 1
    idf = math.log10(idf_number_sentences / number_sentence_containing_word)
    return idf


def tf_idf_score(tf, idf):
    return tf * idf


def word_tf_idf(word, sentences, tf_idf_sentence):
    tf = tf_score(word, tf_idf_sentence)
    idf = idf_score(len(sentences), word, sentences)
    tf_idf = tf_idf_score(tf, idf)
    return tf_idf


def sentence_importance(si_sentence, sentences):
    sentence_score = 0
    si_sentence = remove_special_characters(str(si_sentence))
    pos_tagged_sentence = pos_tagging(si_sentence)

    if not pos_tagged_sentence:
        return 0.0

    si_counter = 0
    for word in pos_tagged_sentence:
        if word.lower() not in stopwords and word not in stopwords and len(word) > 1:
            word = word.lower()
            word = word_lemmatizer.lemmatize(word)
            sentence_score = sentence_score + word_tf_idf(word, sentences, si_sentence)
            si_counter = si_counter + 1

    return sentence_score


file = 'Fariytale_Texts/HanselAndGretel_NoNewLine.txt'
file = open(file, 'r')
input_text = file.read()

tokenized_sentence = sent_tokenize(input_text)

input_text = remove_special_characters(str(input_text))
input_text = re.sub(r'\d+', '', input_text)

tokenized_words_with_stopwords = word_tokenize(input_text)
tokenized_words = [word for word in tokenized_words_with_stopwords if word not in stopwords]
tokenized_words = [word for word in tokenized_words if len(word) > 1]
tokenized_words = [word.lower() for word in tokenized_words]
tokenized_words = lemmatize_words(tokenized_words)
word_frequency = frequency(tokenized_words)

percent_of_original = 25
number_sentences = int((percent_of_original * len(tokenized_sentence)) / 100)

counter = 1
sentence_with_importance = {}
for sentence in tokenized_sentence:
    sentence_importance_value = sentence_importance(sentence, tokenized_sentence)
    sentence_with_importance[counter] = sentence_importance_value
    counter = counter + 1
sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1), reverse=True)

counter = 0
summary = []
sentence_number = []
for word_probability in sentence_with_importance:
    if counter < number_sentences:
        sentence_number.append(word_probability[0])
        counter = counter + 1
    else:
        break
sentence_number.sort()

counter = 1
for sentence in tokenized_sentence:
    if counter in sentence_number:
        summary.append(sentence)
    counter = counter + 1

summary = " ".join(summary)
print("Summary:")
print(summary)
