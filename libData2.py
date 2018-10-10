from collections import Counter
from preprocessing2 import news_wtPair, news_doclist
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

#returns [[(2000, 0.8944271909999159), (3000, 0.4472135954999579)], [(400, 0.7071067811865475), (500, 0.7071067811865475)]]
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


def doc2vec(gensim_vocab, wtIndices, trainVec_ofile, testVec_ofile):
    train_list = news_doclist(gensim_vocab, '/tmp/trainToken.file')
    train_tf_corpus = docs_tf(train_list)
    train_tfidf = docs_tfidf(train_tf_corpus)
    train_wtPair, train_label = news_wtPair(gensim_vocab, '/tmp/train.label', '/tmp/trainTokenTopic.file')

    test_list = news_doclist(gensim_vocab, '/tmp/testToken.file')
    test_tf_corpus = docs_tf(test_list)
    test_tfidf = docs_tfidf(test_tf_corpus)
    test_wtPair, test_label = news_wtPair(gensim_vocab, '/tmp/test.label', '/tmp/testTokenTopic.file')

    embed = np.loadtxt('/tmp/newsTrainVec.txt')

    trainDocs_embed, train_missDoc = generate_doc2vec(train_wtPair, train_label, wtIndices, embed, train_tfidf)
    trainDocs_embed_ar = np.asarray(trainDocs_embed)
    np.savetxt(trainVec_ofile, trainDocs_embed_ar)
    print 'News train vectors saved'

    testDocs_embed, test_missDoc = generate_doc2vec(test_wtPair, test_label, wtIndices, embed, test_tfidf)
    testDocs_embed_ar = np.asarray(testDocs_embed)
    np.savetxt(testVec_ofile, testDocs_embed_ar)
    print 'News test vectors saved'

def lib_inputFormat(newsEmbed_ifile, lib_ofile):
    newsEmbeding = np.loadtxt(newsEmbed_ifile)  # /tmp/newsTestEmbed.txt ,/tmp/newsTrainEmbed.txt'
    index = range(1, newsEmbeding.shape[1])
    indexed_docs = []
    for doc in newsEmbeding:
        label = doc[0]
        doc_embed = doc[1:]
        embed_indexed = ['{0}:{1}'.format(i, j) for i, j in zip(index, doc_embed)]
        indexed_docs.append([int(label)] + embed_indexed)

    indexedDoc_ar = np.asarray(indexed_docs)
    np.savetxt(lib_ofile, indexedDoc_ar, fmt="%s")  # /tmp/libTest.t, /tmp/libTrain

def create_libData(gensim_vocab, wtIndices):
    doc2vec(gensim_vocab, wtIndices, '/tmp/newsTrainEmbed.txt', '/tmp/newsTestEmbed.txt')
    lib_inputFormat('/tmp/newsTrainEmbed.txt','/tmp/libTrain')
    lib_inputFormat('/tmp/newsTestEmbed.txt', '/tmp/libTest.t')
    print 'lib files created'