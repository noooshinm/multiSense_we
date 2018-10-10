import tensorflow as tf
import numpy as np
import gensim
import random
from collections import Counter
import string
import itertools
import codecs
import re
import os

import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora.wikicorpus import filter_wiki


#processing wikipedia and 20NewsGroup in Gibbs format
class wikiUtils(object):

    def tokenize(self, doc):
        doc = doc.lower().strip().split()
        return [token.replace(" ", "") for token in doc if
                2 < len(token.replace(" ", "")) < 20 and token.replace(" ", "") not in STOPWORDS]

    def remove_punc(self, doc):
        table = str.maketrans('', '', string.punctuation)
        punc_rmv = doc.translate(table)       
        return punc_rmv

    def remove_digits(self, doc):
        digist = '0123456789'
        table = str.maketrans('', '', digist)
        dig_rmv = doc.translate(table)
        return dig_rmv

    def process_wiki(self, doc):
        text_ref = self.remove_punc(doc)
        text_ref = self.remove_digits(text_ref)
        text_unicod = filter_wiki(text_ref)
        tokens = self.tokenize(text_unicod)
        return tokens


class UtilNews():
    def parse_msgfile(self, msgfile_input):
        msf_file = open(msgfile_input)
        msg_str = msf_file.read()
        return msg_str

    def remove_header(self, msg):
        header_, seperator_, body = msg.partition('\n\n')
        return body

    def remove_quotes(self, input):
        _QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                               r'|^In article|^Quoted from|^\||^>)')
        good_lines = [line for line in input.split('\n')
                      if not _QUOTE_RE.search(line)]
        return '\n'.join(good_lines)

    def remove_footer(self, msg):
        lines = msg.strip().split('\n')
        for line_num in range(len(lines) - 1, -1, -1):
            line = lines[line_num]
            if line.strip().strip('-') == '':
                break

        if line_num > 0:
            return '\n'.join(lines[:line_num])
        else:
            return msg

    def remove_punctuation(self, msg):
        table = str.maketrans('', '', string.punctuation)
        refined_msg = msg.translate(table)
      return refined_msg

    def remove_digits(self, msg):
        digist = '0123456789'
        table = str.maketrans('', '', digist)
        noDigit_msg = msg.translate(table)
      return noDigit_msg

    def tokenize(self, msg):
        doc = msg.lower().strip().split()
        return [token.replace(" ", "") for token in doc if
                1 < len(token.replace(" ", "")) < 20 and token.replace(" ", "") not in STOPWORDS]

    def process_msg(self, msg_file):
        parsed_msg = self.parse_msgfile(msg_file)
        refined_msg = self.remove_header(parsed_msg)
        refined_msg = self.remove_quotes(refined_msg)
        refined_msg = self.remove_footer(refined_msg)
        refined_msg = self.remove_punctuation(refined_msg)
        refined_msg = self.remove_digits(refined_msg)
        refined_msg = gensim.utils.to_unicode(refined_msg, 'utf-8', errors='ignore').strip()
        document = self.tokenize(refined_msg)
        return document



###after performing gibbsSampling, load wordmap, tassign , phi and theta

#load wordmap
def load_wordmap(wordmap):

    id2word = {}
    with open(wordmap) as f:
        f.readline()
        for l in f:
            word, id = l.strip().split()
            id = int(id)
            id2word[id] = word
    return id2word

#load tassign file and create two files including token and their topics
def load_tokenId_topic(id2word, tokenAssignedTopic, dir, token_fname, tokenId_fname, topic_fname):

    token_path = dir + token_fname
    tokenId_path = dir + tokenId_fname

    token_file = open(token_path, 'w')
    tokenId_file = open(tokenId_path, 'w')

    topic_path = dir + topic_fname
    topic_file = open(topic_path, 'w')

    with open(tokenAssignedTopic) as f:

        for l in f:
            sentence = l.strip().split()
            for w in sentence:
                tokenId, topic = w.strip().split(':')

                tokenId = int(tokenId)
                topic = int(topic)

                if tokenId not in id2word:
                    continue

                token = id2word[tokenId]
                print>> token_file, token,
                print>> tokenId_file, tokenId,
                print>> topic_file, topic,

            print>> token_file
            print>>tokenId_file
            print>> topic_file

        token_file.close()
        tokenId_file.close()
        topic_file.close()


#load word-topic distribution file (phi), where rows are topics and columns are words
def load_wordTopicDist(word_topic_file):
    word_topic_input = open(word_topic_file)
    wt_list = []
    for l in word_topic_input:
        row = l.split()
        wt_list.append(row)
    word_topic_dist = np.transpose(np.asarray(wt_list , dtype=np.float32))
    return word_topic_dist

#load doc-topic distribution file (theta) 
def load_doc_topic_dist(doc_topic_file):
    doc_topic_input = open(doc_topic_file)
    dt_list = []
    for l in doc_topic_input:
        row = l.split()
        dt_list.append(row)
    doc_topic_dist = np.asarray(dt_list, dtype=np.float32)
    return doc_topic_dist


# create lists of token Ids and topics (used in generating training sample, creating index for
    # multiSensed word embedding and unique tokens for scws)
def token_topic_list(tokeId_file, topic_file):
    tokenId_list = []
    topic_list = []
    with open(tokeId_file) as fi:
        for l in fi:
            row = l.split()
            tokenId_list.extend(row)

    with open(topic_file) as f:
        for l in f:
            row = l.split()
            topic_list.extend(row)
    return tokenId_list, topic_list


#for each tokenId in corpus (each vocab), get its multiple senses by getting the number of distinct topics assigned to it
def get_word_multiSenses(tokenIds, topicIds):
    token_topicList = {}
    for i in range(len(tokenIds)):
        k = tokenIds[i]
        if k not in token_topicList:
            token_topicList[k] = []
        token_topicList[k].append(topicIds[i])

    token_num_senses = {}
    for i in token_topicList:
        l = set(token_topicList[i])
        token_num_senses[i]=l
    return token_num_senses


def multiSens_indices (word_multi_senses):
    word_indices = [int(k) for k, v in word_multi_senses.items() for _ in range(len(v))]
    topic_indices = [int(t) for v in word_multi_senses.values() for t in v]
    indices_tuples = zip(word_indices,topic_indices)
    return word_indices, topic_indices, indices_tuples


#get frequency of wordIds 
def word_unigram (tokenId_file, id2word):
    with open(tokenId_file) as f:
        input_string = tf.compat.as_str(f.read())
        token_Ids = input_string.split()
        token_frequency = Counter(token_Ids)
        token_frequency = dict(token_frequency)

    list_unigram = []
    for i in id2word.keys():
        list_unigram.append(token_frequency[str(i)])
    return list_unigram

def build_training_samples (indexed_corpus,topic_file, window_size):
    for index, center in enumerate(indexed_corpus):
        #for each center word pick x random context word within window size
        topic = topic_file[index]
        num_of_contex = random.randint(1, window_size)
        #words to the left of center word
        for context in indexed_corpus[max(0 , index-num_of_contex) : index]:
            yield center, context, topic

        #word to the right of center word
        for context in indexed_corpus[index+1 : index+num_of_contex+1]:
            yield center, context, topic


def get_batch(training_samples, batch_size):
    while True:
        center_batch = np.zeros(batch_size, dtype= np.int32)
        context_batch = np.zeros(batch_size, dtype=np.int32)
        topic_batch = np.zeros(batch_size, dtype= np.int32)
        for i in range(batch_size):
            center_batch[i], context_batch[i], topic_batch[i] = next(training_samples)
        yield center_batch, context_batch, topic_batch



###data preprocessing of scws dataset 

def parse_scws(input_file, vocabulary):
    
    indexed_vocab = dict(zip(vocabulary.values(), vocabulary.keys()))
    doc_list = []
    word_pairs = []
    pair_missed = []
    scores = []
    wdoc_pair = []
    c1_index = 0
    c2_index = 1

    with open(input_file) as fi:
        for index, doc in enumerate(fi):
            doc_split = doc.lower().strip().split('\t')
            scr = [float(s) for s in doc_split[-11:]][1:]

            w1 = doc_split[1]
            w2 = doc_split[3]
            c1 = doc_split[5]
            c2 = doc_split[6]

            w1_ref = wikiUtils().process_wiki(w1)
            w2_ref = wikiUtils().process_wiki(w2)
            c1_ref = wikiUtils().process_wiki(c1)
            c2_ref = wikiUtils().process_wiki(c2)

            cw1 = w1_ref + c1_ref
            cw2 = w2_ref + c2_ref

            doc_list.append(cw1)
            doc_list.append(cw2)

            w1_id = indexed_vocab.get(w1.lower().strip())
            w2_id = indexed_vocab.get(w2.lower().strip())

            if w1_id == None or w2_id == None:
                pmiss = (w1, w2)
                pair_missed.append(pmiss)

            else:
                w_pair = (w1_id, w2_id)
                word_pairs.append(w_pair)
                scores.append(scr)
                w1_doc = (w1_id, c1_index)
                w2_doc = (w2_id, c2_index)
                w_doc_pair = (w1_doc, w2_doc)
                wdoc_pair.append(w_doc_pair)

            c1_index += 2
            c2_index += 2
    return doc_list, word_pairs, pair_missed, scores, wdoc_pair




#get a sorted list of unique token Ids in scws dataset
def scws_token_list (tokenList):
    unique_tokens = map(int, set(tokenList))
    unique_tokens.sort()
    return unique_tokens


#load word_topic assignment for scws data  ((w1,t1),(w2,t2))
def get_scws_wtPair(file_name):
    input_file = open(file_name, 'r')
    all_pairs = []
    wt_pairs = []
    for i , doc in enumerate(input_file):
        doc_split = doc.strip().split()
        wt = doc_split[0]
        w, t = wt.split(':')
        wt = (int(w),int(t))
        all_pairs.append(wt)

    j = 0
    for i in range(len(all_pairs)/2):
        p = (all_pairs[j],all_pairs[j+1])
        wt_pairs.append(p)
        j+=2

  
    '''
    wt_1 = []
    wt_2 = []
    for i, doc in enumerate(input_file):
        doc_split = doc.strip().split()
        wt = doc_split[0]
        w, t = wt.split(':')
        wt = (int(w), int(t))
        if i%2 == 0:
            wt_1.append(wt)
        if i%2 == 1:
            wt_2.append(wt)

    wt_pairs = zip(wt_1,wt_2)
    '''

    return wt_pairs


###parsing 20news dataset
def load_NewsLabel(input_file):
    labels = []
    with open(input_file) as fi:
        for i, label in enumerate(fi):
            labels.append(int(label.strip()))
    return labels

def load_News_assign(input_file, labels):
    wd_pair  = []
    with open(input_file) as fi:
        for i, doc in enumerate(fi):
            dwt = []
            msg = doc.strip().split()
            if len(msg) == 0:
                labels.remove(labels[i])
                continue
            for wt in msg:
                w, t = wt.split(':')
                wt_pair = (int(w), int(t))
                dwt.append(wt_pair)

            
            wd_pair.append(dwt)
    return wd_pair, labels


def news_doclist(tokenId_file):
    input_file = open(tokenId_file)
    docs_list = []
    for doc in input_file:
        doc_ids = []
        row = doc.split()
        if len(row)==0:
            continue
        for wId in row:
            doc_ids.append(int(wId))
        docs_list.append(doc_ids)
    input_file.close()
    return docs_list












