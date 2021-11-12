from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

## create a dictionary of words and their frequencies


def _create_word_frequency_table(text: str) -> dict:
    """
    Create a dictionary of words and their frequencies
    :param text:
    :return:
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


def _score_sentences_by_words(sentences: list, freq_table: dict) -> dict:
    """
    Score sentences by the frequency of words
    :param sentences:
    :param freq_table:
    :return:
    """
    sentence_score = dict()
    for sentence in sentences:
        sentence_word_count = (len(word_tokenize(sentence)))
        word_count_except_stop_words = 0
        for word_freq in freq_table:
            if word_freq in sentence.lower():
                word_count_except_stop_words += 1
                if sentence[:10] in sentence_score:
                    sentence_score[sentence[:10]] += freq_table[word_freq]
                else:
                    sentence_score[sentence[:10]] = freq_table[word_freq]

        if sentence[:10] in sentence_score:
            sentence_score[sentence[:10]] = sentence_score[
                sentence[:10]] / word_count_except_stop_words

        # Average score = sum of all sentence scores / total sentence count
        # sentence_score[sentence] = sentence_score[sentence] / sentence_word_count
    return sentence_score


def _find_average_score(sentence_score: dict) -> int:
    """
    Find the average score
    :param sentence_score:
    :return:
    """
    sum_values = 0
    for entry in sentence_score:
        sum_values += sentence_score[entry]
    average_score = (sum_values / len(sentence_score))
    return average_score


def _generate_summary(sentences: list, sentence_score: dict,
                      threshold: float) -> str:
    """
    Generate the summary
    :param sentences:
    :param sentence_score:
    :param threshold:
    :return:
    """
    sentence_counter = 0
    summary = ''
    for sentence in sentences:
        if sentence[:10] in sentence_score and sentence_score[
                sentence[:10]] >= threshold:
            summary += " " + sentence
            sentence_counter += 1
    return summary


def run_summarization(text: str) -> str:
    """
    Run the whole process
    :param text:
    :param threshold:
    :return:
    """
    freq_table = _create_word_frequency_table(text)
    sentences = sent_tokenize(text)
    sentence_scores = _score_sentences_by_words(sentences, freq_table)
    threshold = _find_average_score(sentence_scores)
    summary = _generate_summary(sentences, sentence_scores, threshold)
    return summary