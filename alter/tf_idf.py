## import NLP packages
from nltk import word_tokenize, sent_tokenize, PorterStemmer
from nltk.corpus import stopwords
import math

# we create a dictionary for the word frequency table.
# For this, we should only use the words that are not part of the stopWords array.
# Removing stop words and making frequency table
# Stemmer - an algorithm to bring words to its root word.


def _create_frequency_table(text: str) -> dict:
    stopwords_set = set(stopwords.words("english"))
    words = word_tokenize(text)
    ps = PorterStemmer()

    freq_table = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopwords_set:
            continue
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    return freq_table


def _create_frequency_matrix(sentences: list) -> list:
    frequency_matrix = []
    stopwords_set = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sentence in sentences:
        freq_table = dict()
        for word in word_tokenize(sentence):
            word = word.lower()
            word = ps.stem(word)
            if word in stopwords_set:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix.append(freq_table)
    return frequency_matrix

def _create_tf_matrix(freq_matrix: list) -> list:
    tf_matrix = []
    for sentence in freq_matrix:
        tf_sentence = []
        for word in sentence:
            tf_sentence.append(sentence[word] / sum(sentence.values()))
        tf_matrix.append(tf_sentence)
    return tf_matrix

def _create_documents_per_words(freq_matrix: list) -> dict:
    doc_per_words = dict()
    for sentence in freq_matrix:
        for word in sentence:
            if word in doc_per_words:
                doc_per_words[word] += 1
            else:
                doc_per_words[word] = 1
    return doc_per_words

def _create_idf_matrix(freq_matrix: list, doc_per_words: dict, total_documents: int) -> list:
    idf_matrix = []
    for sentence in freq_matrix:
        idf_sentence = []
        for word in sentence:
            idf_sentence.append(math.log10(total_documents / doc_per_words[word]))
        idf_matrix.append(idf_sentence)
    return idf_matrix


def _create_tf_idf_matrix(tf_matrix: list, idf_matrix: list) -> list:
    tf_idf_matrix = []
    for i in range(len(tf_matrix)):
        tf_idf_sentence = []
        for j in range(len(tf_matrix[i])):
            tf_idf_sentence.append(tf_matrix[i][j] * idf_matrix[i][j])
        tf_idf_matrix.append(tf_idf_sentence)
    return tf_idf_matrix

def _score_sentences(tf_idf_matrix: list) -> dict:
    sentenceValue = dict()

    for i in range(len(tf_idf_matrix)):
        sentenceValue[i] = sum(tf_idf_matrix[i])

    return sentenceValue

def _find_average_score(sentenceValue: dict) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average

def _generate_summary(sentences: list, sentenceValue: dict, threshold: int) -> str:
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[0] in sentenceValue and sentenceValue[sentence[0]] >= (threshold):
            summary += " " + sentence[0]
            sentence_count += 1

    return summary

def run_summarization(text: str, threshold: int) -> str:
    # 1. Create the word frequency table
    freq_table = _create_frequency_table(text)
    # 2. Create the word frequency matrix
    freq_matrix = _create_frequency_matrix(sent_tokenize(text))
    # 3. Create the TF Matrix
    tf_matrix = _create_tf_matrix(freq_matrix)
    # 4. Create the IDF Matrix
    doc_per_words = _create_documents_per_words(freq_matrix)
    total_documents = len(freq_matrix)
    idf_matrix = _create_idf_matrix(freq_matrix, doc_per_words, total_documents)
    # 5. Compute the TF-IDF matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    # 6. Determine the score of each sentence
    sentence_scores = _score_sentences(tf_idf_matrix)
    # 7. Find the threshold
    threshold = _find_average_score(sentence_scores)
    # 8. Generate the summary
    summary = _generate_summary(sent_tokenize(text), sentence_scores, threshold)

    return summary