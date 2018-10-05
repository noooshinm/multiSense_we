from collections import Counter
from sg_preprocess import sg_news_words, sg_news_doclist
from gensim import models
import numpy as np


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

def sg_generate_doc2vec(g_vocab ,sg_docs, labels, sg_word_embedings, isWeighted=False, tfidf_docs=None):
    doc2vec = []
    nodoc = []
    for i, doc in enumerate(sg_docs):
        doc_embed = 0
        doc_length = 0

        for w in doc:
            w_id = g_vocab[w].index
            w_embedding = sg_word_embedings[w_id]

            if isWeighted:
                w_tfidf = dict(tfidf_docs[i]).get(w_id)
                if w_tfidf == None:
                    continue
                weighted_embedding = w_tfidf * w_embedding
                doc_embed = doc_embed + weighted_embedding
                doc_length += 1

            else:
                doc_embed = doc_embed + w_embedding
                doc_length += 1


        if type(doc_embed) is int:
            nodoc.append(i)
            continue

        doc_embed = doc_embed / doc_length
        doc2vec.append(np.concatenate(([labels[i]], doc_embed), axis=0))
    return doc2vec, nodoc


def sg_doc2vec(gensim_vocab, sg_embed, trainVec_ofile, testVec_ofile):

    # train_list = sg_news_doclist(gensim_vocab, '/tmp/trainToken.file')
    # train_tf_corpus = docs_tf(train_list)
    # train_tfidf = docs_tfidf(train_tf_corpus)
    train_w, train_label = sg_news_words(gensim_vocab, '/tmp/train.label', '/tmp/trainToken.file')

    # test_list = sg_news_doclist(gensim_vocab, '/tmp/testToken.file')
    # test_tf_corpus = docs_tf(test_list)
    # test_tfidf = docs_tfidf(test_tf_corpus)
    test_w, test_label = sg_news_words(gensim_vocab, '/tmp/test.label', '/tmp/testToken.file')

    #trainDocs_embed, train_missDoc = sg_generate_doc2vec(gensim_vocab,train_w, train_label, sg_embed, train_tfidf)
    trainDocs_embed, train_missDoc = sg_generate_doc2vec(gensim_vocab, train_w, train_label, sg_embed)
    trainDocs_embed_ar = np.asarray(trainDocs_embed)
    np.savetxt(trainVec_ofile, trainDocs_embed_ar)
    print ('News train vectors saved')

    #testDocs_embed, test_missDoc = sg_generate_doc2vec(gensim_vocab,test_w, test_label, sg_embed, test_tfidf)
    testDocs_embed, test_missDoc = sg_generate_doc2vec(gensim_vocab, test_w, test_label, sg_embed)
    testDocs_embed_ar = np.asarray(testDocs_embed)
    np.savetxt(testVec_ofile, testDocs_embed_ar)
    print ('News test vectors saved')



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


def sg_create_libData(gensim_vocab, sg_embed):
    sg_doc2vec(gensim_vocab, sg_embed, '/tmp/sgTfNewsTrainEmbedlr001.txt', '/tmp/sgTfNewsTestEmbedlr001.txt')
    lib_inputFormat('/tmp/sgTfNewsTrainEmbedlr001.txt','/tmp/sgTflibTrainlr001')
    lib_inputFormat('/tmp/sgTfNewsTestEmbedlr001.txt', '/tmp/sgTflibTestlr001.t')
    print ('sg lib files created')




