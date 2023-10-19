import numpy as np
import os
import takahe
import nltk.data
import spacy

# nltk.download('punkt')
# nltk.download('wordnet')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
spacynlp = spacy.load("en_core_web_sm")


def path_length_normalization_score(nbest_compressions):
    reranked_compressions = []

    for cummulative_score, path in nbest_compressions:
        score = cummulative_score / len(path)
        bisect.insort(reranked_compressions, (score, path))

    return reranked_compressions


def read_file(input_file_path):
    f = open(input_file_path, "r", encoding='utf-8')
    lines = f.readlines()
    src_list = []
    tag = "story_separator_special_tag"
    for line in lines:
        line = line.replace(tag, "")
        sent_list = sent_detector.tokenize(line.strip())
        src_list.append(sent_list)
    return src_list


def tag_pos(str_text):
    doc = spacynlp(str_text)
    textlist = []

    for item in doc:
        source_token = item.text
        source_pos = item.tag_
        textlist.append(source_token + '/' + source_pos)
    return ' '.join(textlist)
