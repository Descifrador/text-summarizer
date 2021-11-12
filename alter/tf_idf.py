## import NLP packages
from nltk import word_tokenize, sent_tokenize, PorterStemmer
from nltk.corpus import stopwords
import math


def _create_frequency_table(text: str) -> dict:
    """
    Returns a dictionary of words and their frequency in the text.
    :param text: The text to be summarized.
    :return: A dictionary of words and their frequency in the text.
    """
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


def _create_frequency_matrix(sentences: list) -> dict:
    """
    Returns a frequency matrix of the words in each sentence.
    :param sentences: The sentences to be summarized.
    :return: A frequency matrix of the words in each sentence.
    """
    frequency_matrix = dict()
    stopwords_set = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sentence in sentences:
        freq_table = dict()
        words = word_tokenize(sentence)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopwords_set:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        frequency_matrix[sentence[:15]] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix: dict) -> dict:
    """
    Returns a term frequency matrix for the given frequency matrix.
    :param freq_matrix: The frequency matrix to be summarized.
    :return: A term frequency matrix for the given frequency matrix.
    """
    tf_matrix = {}
    for sentence, freq_table in freq_matrix.items():
        tf_table = {}
        count_words_in_sentence = len(freq_table)
        for word, frequency in freq_table.items():
            tf_table[word] = frequency / count_words_in_sentence
        tf_matrix[sentence] = tf_table
    return tf_matrix


def _create_documents_per_words(freq_matrix: dict) -> dict:
    """
    Returns a dictionary of words and the number of documents in which they
    appear.
    :param freq_matrix: The frequency matrix to be summarized.
    :return: A dictionary of words and the number of documents in which they appear.
    """
    doc_per_words = dict()
    for sentence, freq_table in freq_matrix.items():
        for word, frequency in freq_table.items():
            if word in doc_per_words:
                doc_per_words[word] += 1
            else:
                doc_per_words[word] = 1
    return doc_per_words


def _create_idf_matrix(freq_matrix: dict, doc_per_words: dict,
                       total_documents: int) -> dict:
    """
    Returns an idf matrix for the given frequency matrix.
    :param freq_matrix: The frequency matrix to be summarized.
    :param doc_per_words: The number of documents in which each word appears.
    :param total_documents: The total number of documents.
    :return: An idf matrix for the given frequency matrix.
    """
    idf_matrix = dict()
    for word, frequency in freq_matrix.items():
        idf_table = dict()
        for sentence in frequency.keys():
            if sentence in idf_table:
                continue
            idf_table[sentence] = math.log10(total_documents /
                                             doc_per_words[word])
        idf_matrix[word] = idf_table
    return idf_matrix


def _create_tf_idf_matrix(tf_matrix: dict, idf_matrix: dict) -> dict:
    """
    Returns a tf-idf matrix for the given frequency matrix.
    :param tf_matrix: The term frequency matrix to be summarized.
    :param idf_matrix: The inverse document frequency matrix.
    :return: A tf-idf matrix for the given frequency matrix.
    """
    tf_idf_matrix = dict()

    for (sentence1, tf_table1), (sentence2,
                                 tf_table2) in zip(tf_matrix.items(),
                                                   idf_matrix.items()):
        tf_idf_table = dict()
        for (word1, value1), (word2, value2) in zip(tf_table1.items(),
                                                    tf_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sentence1] = tf_idf_table
    return tf_idf_matrix


def _score_sentences(tf_idf_matrix: dict) -> dict:
    """
    Finds the score of each sentence in the text based on the tf-idf values.
    :param tf_idf_matrix: The term frequency-inverse document frequency matrix.
    :return: The score of each sentence in the text.
    """
    sentenceValue = dict()

    for sentence, freq_table in tf_idf_matrix.items():
        score = 0
        count_words_in_sentence = len(freq_table)
        for word, value in freq_table.items():
            score += value
        if count_words_in_sentence > 0:
            sentenceValue[sentence] = score / count_words_in_sentence
    return sentenceValue


def _find_average_score(sentenceValue: dict) -> int:
    """
    Finds the average score from the sentence value dictionary.
    :param sentenceValue: The sentence value dictionary.
    :return: The average score from the sentence value dictionary.
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues /
               len(sentenceValue)) if len(sentenceValue) > 0 else sumValues

    return average


def _generate_summary(sentences: dict, sentenceValue: dict,
                      threshold: int) -> str:
    """
    Generates a summary of the text.
    :param sentences: The sentences of the text.
    :param sentenceValue: The sentence value dictionary.
    :param threshold: The threshold value.
    :return: The summary of the text.
    """
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (
                threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


def run_summarization(text: str) -> str:
    # Tokenize the sentences
    sentences = sent_tokenize(text)
    total_documents = len(sentences)

    # Frequency matrix of the words in each sentence
    freq_matrix = _create_frequency_matrix(sentences)

    # Term frequency matrix
    tf_matrix = _create_tf_matrix(freq_matrix)

    # Number of times each word appears in whole corpus
    doc_per_words = _create_documents_per_words(freq_matrix)

    # Inverse document frequency
    idf_matrix = _create_idf_matrix(freq_matrix, doc_per_words,
                                    total_documents)

    # TF-IDF matrix
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)

    # Score of each sentence
    sentence_score = _score_sentences(tf_idf_matrix)

    # Find average score
    threshold = _find_average_score(sentence_score)

    # Generate summary
    summary = _generate_summary(sentences, sentence_score, threshold)

    return summary
