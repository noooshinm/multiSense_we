from collections import Counter
from preprocessing import news_wtPair, news_doclist
from gensim import models
import numpy as np



#in each document obtain frequency of each word in that document , example: s = [[1000,2000,3000,2000],[400,1000,500]]
    # return: [[(1000, 1), (2000, 2), (3000, 1)], [(400, 1), (1000, 1), (500, 1)]]

def docs_tf(doclist):
    corpus_tf = []
    for doc in doclist:
        tf = Counter()
        for word in doc:
            tf[word] += 1
        corpus_tf.append(tf.items())
    return corpus_tf


def docs_tfidf(corpus):
    tfidf = models.TfidfModel(corpus)
    return [tfidf[corpus[i]] for i in range(len(corpus))]



def generate_doc2vec(wt_pairs, labels, tuple_indices, word_embedings, tfidf_docs):
    doc2vec = []
    nodoc = []
    for i, doc in enumerate(wt_pairs):
        doc_embed = 0
        doc_length = 0

        for wt in doc:
            if wt not in tuple_indices:
                continue
            wt_index = tuple_indices.index(wt)
            wt_embedding = word_embedings[wt_index]

            w_id = int(wt[0])
            w_tfidf = dict(tfidf_docs[i]).get(w_id)

            if w_tfidf == None:
                continue

            weighted_embedding = w_tfidf * wt_embedding
            doc_embed = doc_embed + weighted_embedding
            doc_length += 1

        if type(doc_embed) is int:
            nodoc.append(i)
            continue

        doc_embed = doc_embed / doc_length
        doc2vec.append(np.concatenate(([labels[i]], doc_embed), axis=0))
    return doc2vec, nodoc
''''
generate the file format for liblinear classifier  
'''
def lib_inputFormat(newsEmbed_ifile, lib_ofile):
    newsEmbeding = np.loadtxt(newsEmbed_ifile)  
    index = range(1, newsEmbeding.shape[1])
    indexed_docs = []
    for doc in newsEmbeding:
        label = doc[0]
        doc_embed = doc[1:]
        embed_indexed = ['{0}:{1}'.format(i, j) for i, j in zip(index, doc_embed)]
        indexed_docs.append([int(label)] + embed_indexed)

    indexedDoc_ar = np.asarray(indexed_docs)
    np.savetxt(lib_ofile, indexedDoc_ar, fmt="%s")  # /tmp/libTest.t, /tmp/libTrain

