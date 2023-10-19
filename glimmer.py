from sentence_graph import SentenceGraph
from utils import *
from sklearn.cluster import SpectralClustering
import torch
import nltk.data
# from bertopic import BERTopic
# from top2vec import Top2Vec
# import tensorflow_hub as hub
from cal_nbclusters import *
import pickle
from lexicalrichness import LexicalRichness
# import matplotlib.pyplot as plt
import math
import re
import bisect
from language_model import LanguageModel
import numpy as np
# import umap
# from sklearn.manifold import TSNE
# from sklearn.decomposition import KernelPCA
from eigengap import predict_k
import warnings

warnings.filterwarnings("ignore")


class GLIMMER():

    def __init__(
            self,
            method: str = "ttr",
            beta: int = 4,
            sigma: float = 0.05,
            nb_words: int = 6,
            ita: float = 0.98,
            seed: int = 88,
            w2v_file: str = "resources/word2vec.txt",
            lm_path: str = "",
            use_lm: bool = False,
            output_file_path: str = "Summary.txt"
    ):

        self.method = method
        self.beta = beta  # for GLIMMER-TTR
        self.sigma = sigma  # for GLIMMER-TTR
        self.nb_words = nb_words
        self.ita = ita
        self.seed = seed
        self.use_lm = use_lm
        self.lm = LanguageModel(model_path='resources/en-70k-0.2.lm')
        self.output_file_path = output_file_path

        if not self.use_lm:
            self.w2v = self._get_w2v_embeddings(w2v_file)
            self.lm_tokenizer = ""
            self.lm_model = ""
        else:
            from transformers import GPT2Tokenizer, GPT2Model
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_path)
            self.lm_model = GPT2Model.from_pretrained(lm_path,
                                                      output_hidden_states=True,
                                                      output_attentions=False)
            self.w2v = ""

        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def _get_w2v_embeddings(self, w2v_file):
        word_embeddings = {}
        f = open(w2v_file, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()
        return word_embeddings

    def construct_sentence_graph(self, sentences_list):
        graph = SentenceGraph(sentences_list, self.w2v, self.use_lm, self.lm_model, self.lm_tokenizer, self.ita)
        X = graph.build_sentence_graph()
        return X

    def path_length_normalization_score(self, nbest_compressions):
        reranked_compressions = []

        for cummulative_score, path in nbest_compressions:
            score = cummulative_score / len(path)
            bisect.insort(reranked_compressions, (score, path))

        return reranked_compressions

    def fluency_score(self, nbest_compressions, normalization=True):
        all_scores = []
        for w, sentence in nbest_compressions:
            sentence_clean = ' '.join([word[0] for word in sentence])
            score = self.lm.get_sentence_score(
                sentence=sentence_clean,
                n=3,
                unknown_word_prob=1e-5,
                normalization=normalization
            )
            all_scores.append(score)
        all_scores = np.array(all_scores)
        return all_scores

    def remove_sentence_tag(self, nbest_compressions):
        for i in range(len(nbest_compressions)):
            sentence = nbest_compressions[i][1]
            sentence_clean = ' '.join([word[0] for word in sentence])
            nbest_compressions[i] = (nbest_compressions[i][0], sentence_clean)
        return nbest_compressions

    def cluster_graph(self, X, sentences_list):
        try:
            clustering = SpectralClustering(n_clusters=self.nb_clusters, random_state=self.seed,
                                            affinity='precomputed').fit(X)
        except:
            return {0: sentences_list}

        clusterIDs = clustering.labels_

        num_clusters = max(clusterIDs) + 1
        cluster_dict = {new_list: [] for new_list in range(num_clusters)}

        for i, clusterID in enumerate(clusterIDs):
            cluster_dict[clusterID].append(sentences_list[i])
        return cluster_dict

    def convert_sents_to_tagged_sents(self, sent_list):
        tagged_list = []
        if (len(sent_list) > 0):
            for s in sent_list:
                s = s.replace("/", "")
                temp_tagged = tag_pos(s)
                tagged_list.append(temp_tagged)
        else:
            tagged_list.append(tag_pos("."))
        return tagged_list

    def get_compressed_sen(self, sentences):
        compresser = takahe.word_graph(sentence_list=sentences, nb_words=self.nb_words, lang='en', punct_tag=".")
        candidates = compresser.get_compression(50)  # backbone 1, new 50
        # backbone
        # if len(candidates) > 0:
        #     score, path = candidates[0]
        #     result = ' '.join([u[0] for u in path])
        # else:
        #     result = ' '
        # return result

        # new
        nbest_compressions = self.path_length_normalization_score(candidates)
        # nbest_compressions = candidates
        fl_score = self.fluency_score(nbest_compressions, normalization=True)
        for i, compression in enumerate(nbest_compressions):
            nbest_compressions[i] = (compression[0] / fl_score[i], compression[1])
        sorted_by_score = sorted(nbest_compressions, key=lambda tup: tup[0])
        for a in self.remove_sentence_tag(sorted_by_score[:1]):
            return a[1]
        # end new

        reranker = takahe.keyphrase_reranker(sentences, candidates, lang='en')

        reranked_candidates = reranker.rerank_nbest_compressions()
        # print(reranked_candidates)
        if len(reranked_candidates) > 0:
            score, path = reranked_candidates[0]
            result = ' '.join([u[0] for u in path])
        else:
            result = ' '
        return result

    def compress_cluster(self, cluster_dict):
        summary = []
        for k, v in cluster_dict.items():
            tagged_sens = self.convert_sents_to_tagged_sents(v)
            compressed_sent = self.get_compressed_sen(tagged_sens)
            summary.append(compressed_sent)
            final_summary = " ".join(summary)
            final_summary = re.sub(' +', ' ', final_summary)
        return final_summary

    def summarize(self, src_list):
        summary_list = []

        # 1. TTR
        if self.method == 'ttr':
            print('using GLIMMER-TTR...')
            for idx, sentences_list in enumerate(src_list):
                print('No.' + str(idx))
                num_sents = len(sentences_list)
                sentences = ' '.join(sentences_list)

                lex = LexicalRichness(sentences)
                ttr_sample = lex.ttr

                try:
                    D = lex.vocd()
                except ValueError as e:
                    print(e)
                    summary = " ".join(sentences_list)
                    summary_list.append(summary)
                    with open(self.output_file_path, 'a', encoding='utf-8', errors='ignore') as f:
                        summ = summary.replace("\n", "")
                        f.write(summ + '\n')
                    continue
                low = 0
                high = 0
                for sentence in sentences_list:
                    lex = LexicalRichness(sentence)
                    N = lex.words
                    try:
                        ttr_true = lex.ttr
                    except ZeroDivisionError as e:
                        print(e)
                        continue
                    temp = math.sqrt(1 + 2 * N / D)
                    ttr_pred = (D / N) * (temp - 1)
                    if (ttr_true / ttr_pred) < (1 - self.sigma):
                        low += 1
                    if (ttr_true / ttr_pred) > (1 + self.sigma):
                        high += 1
                number = int((num_sents - (self.beta * low) + (self.beta * high)) * ttr_sample)
                if number > num_sents:
                    number = num_sents
                if number < 2:
                    number = 2

                X = self.construct_sentence_graph(sentences_list)
                try:
                    clustering = SpectralClustering(n_clusters=number, random_state=self.seed,
                                                    affinity='precomputed').fit(X)
                except:
                    print('error!')
                    with open(self.output_file_path, 'a', encoding='utf-8', errors='ignore') as f:
                        summ = " ".join(sentences_list)
                        summ = summ.replace("\n", "")
                        f.write(summ + '\n')
                    continue
                clusterIDs = clustering.labels_
                num_clusters = max(clusterIDs) + 1
                cluster_dict = {new_list: [] for new_list in range(num_clusters)}
                cluster_dict_index = {new_list: [] for new_list in range(num_clusters)}
                cluster_dict_sorted = {new_list: [] for new_list in range(num_clusters)}
                for i, clusterID in enumerate(clusterIDs):
                    cluster_dict[clusterID].append(sentences_list[i])
                    cluster_dict_index[clusterID].append(i)
                cluster_index_sorted = sorted(cluster_dict_index.items(), key=lambda item: item[1][0])
                cluster_dict_index_sorted = {}
                for i in cluster_index_sorted:
                    cluster_dict_index_sorted[i[0]] = i[1]
                for index, v in enumerate(cluster_dict_index_sorted.values()):
                    temp_list = []
                    for i in v:
                        temp_list.append(sentences_list[i])
                    cluster_dict_sorted[index] = temp_list

                summary = self.compress_cluster(cluster_dict_sorted)
                summary_list.append(summary)

                with open(self.output_file_path, 'a', encoding='utf-8', errors='ignore') as f:
                    summ = summary.replace("\n", "")
                    f.write(summ + '\n')

        # 2. distance
        if self.method == 'distance':
            print('using GLIMMER-Distance...')
            for idx, sentences_list in enumerate(src_list):
                print('No.' + str(idx))
                num_sents = len(sentences_list)

                X = self.construct_sentence_graph(sentences_list)
                dis = floyd(X)
                Scores = []

                for k in range(1, num_sents + 1):
                    try:
                        clustering = SpectralClustering(n_clusters=k, random_state=self.seed,
                                                        affinity='precomputed').fit(X)
                        clusterIDs = clustering.labels_
                        num_clusters = max(clusterIDs) + 1
                        cluster_dict = {new_list: [] for new_list in range(num_clusters)}
                        for i, clusterID in enumerate(clusterIDs):
                            cluster_dict[clusterID].append(i)
                        Scores.append(cal_point(dis, cluster_dict))
                    except:
                        print('error!')
                        Scores.append(0.)
                i = range(1, num_sents + 1)
                number = Scores.index(max(Scores)) + 1

                try:
                    clustering = SpectralClustering(n_clusters=number, random_state=self.seed,
                                                    affinity='precomputed').fit(X)
                except:
                    print('error!')
                    with open(self.output_file_path, 'a', encoding='utf-8', errors='ignore') as f:
                        summ = " ".join(sentences_list)
                        summ = summ.replace("\n", "")
                        f.write(summ + '\n')
                    continue

                clusterIDs = clustering.labels_
                num_clusters = max(clusterIDs) + 1
                cluster_dict = {new_list: [] for new_list in range(num_clusters)}
                cluster_dict_index = {new_list: [] for new_list in range(num_clusters)}
                cluster_dict_sorted = {new_list: [] for new_list in range(num_clusters)}
                for i, clusterID in enumerate(clusterIDs):
                    cluster_dict[clusterID].append(sentences_list[i])
                    cluster_dict_index[clusterID].append(i)

                cluster_index_sorted = sorted(cluster_dict_index.items(), key=lambda item: item[1][0])
                cluster_dict_index_sorted = {}
                for i in cluster_index_sorted:
                    cluster_dict_index_sorted[i[0]] = i[1]
                for index, v in enumerate(cluster_dict_index_sorted.values()):
                    temp_list = []
                    for i in v:
                        temp_list.append(sentences_list[i])
                    cluster_dict_sorted[index] = temp_list

                summary = self.compress_cluster(cluster_dict_sorted)
                summary_list.append(summary)
                with open(self.output_file_path, 'a', encoding='utf-8', errors='ignore') as f:
                    summ = summary.replace("\n", "")
                    f.write(summ + '\n')

        # 3.eigengap
        if self.method == 'eigengap':
            print('using GLIMMER-eigengap...')
            for idx, sentences_list in enumerate(src_list):
                print('No.' + str(idx))

                X = self.construct_sentence_graph(sentences_list)
                number = predict_k(X)

                try:
                    clustering = SpectralClustering(n_clusters=number, random_state=self.seed,
                                                    affinity='precomputed').fit(X)
                except:
                    print('error!')
                    with open(self.output_file_path, 'a', encoding='utf-8', errors='ignore') as f:
                        summ = " ".join(sentences_list)
                        summ = summ.replace("\n", "")
                        f.write(summ + '\n')
                    continue
                clusterIDs = clustering.labels_
                num_clusters = max(clusterIDs) + 1
                cluster_dict = {new_list: [] for new_list in range(num_clusters)}
                cluster_dict_index = {new_list: [] for new_list in range(num_clusters)}
                cluster_dict_sorted = {new_list: [] for new_list in range(num_clusters)}
                for i, clusterID in enumerate(clusterIDs):
                    cluster_dict[clusterID].append(sentences_list[i])
                    cluster_dict_index[clusterID].append(i)
                cluster_index_sorted = sorted(cluster_dict_index.items(), key=lambda item: item[1][0])
                cluster_dict_index_sorted = {}
                for i in cluster_index_sorted:
                    cluster_dict_index_sorted[i[0]] = i[1]
                for index, v in enumerate(cluster_dict_index_sorted.values()):
                    temp_list = []
                    for i in v:
                        temp_list.append(sentences_list[i])
                    cluster_dict_sorted[index] = temp_list

                summary = self.compress_cluster(cluster_dict_sorted)
                summary_list.append(summary)
                with open(self.output_file_path, 'a', encoding='utf-8', errors='ignore') as f:
                    summ = summary.replace("\n", "")
                    f.write(summ + '\n')

        return summary_list
