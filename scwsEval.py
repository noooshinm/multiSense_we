import numpy as np
import scipy
import scipy.stats

from preprocessing import scws_wtPair, get_indices, dt_wt_dist, get_scws_unqToken, init_scws


def getScore(scores, wt_pairs, lookup_indices, embeding):
    model_sim = []
    human_sim = []
    pairs_missed = []
    for i, (w1, w2) in enumerate(wt_pairs):
        try:
            w1_index = lookup_indices.index(w1)
            w2_index = lookup_indices.index(w2)

            w1_embed = embeding[w1_index]
            w2_embed = embeding[w2_index]
            cosine_sim = np.dot(w1_embed, w2_embed)

        except:
            pmiss = (w1, w2)
            pairs_missed.append(pmiss)
            cosine_sim = 0

        model_sim.append(cosine_sim)

    for scr in scores:
        human_sim.append(np.mean(scr))

    return scipy.stats.spearmanr(human_sim, model_sim)[0]




def avgSim_maxSimC(scores, center_weight, topic_weight, wt_dist, dt_dist, wd_pair, unq_token_list):
    maxSim_sim = []
    avgSim_sim = []
    human_sim = []
    for i, (wd1, wd2) in enumerate(wd_pair):
        w1_vId = wd1[0]
        w1_docId = wd1[1]

        w2_vId = wd2[0]
        w2_docId = wd2[1]

        w1_embed = center_weight[w1_vId]
        w1_index_in_wtDist = unq_token_list.index(w1_vId)
        w1_wt_prob = wt_dist[w1_index_in_wtDist]
        w1_dt_prob = dt_dist[w1_docId]

        w1_avgSimc = np.sum(w1_embed * topic_weight * w1_wt_prob[:, np.newaxis] * w1_dt_prob[:, np.newaxis], axis=0)

        w1_max_prob = np.argmax(w1_wt_prob * w1_dt_prob)
        w1_maxSimC = w1_embed * topic_weight[w1_max_prob]

        w2_embed = center_weight[w2_vId]
        w2_index_in_wtDist = unq_token_list.index(w2_vId)
        w2_wt_prob = wt_dist[w2_index_in_wtDist]
        w2_dt_prob = dt_dist[w2_docId]

        w2_avgSimc = np.sum(w2_embed * topic_weight * w2_wt_prob[:, np.newaxis] * w2_dt_prob[:, np.newaxis], axis=0)

        w2_max_prob = np.argmax(w2_wt_prob * w2_dt_prob)
        w2_maxSimC = w2_embed * topic_weight[w2_max_prob]

        avgSimCosine = np.dot(w1_avgSimc, w2_avgSimc)
        maxSimCosine = np.dot(w1_maxSimC, w2_maxSimC)

        avgSim_sim.append(avgSimCosine)
        maxSim_sim.append(maxSimCosine)

    for scr in scores:
        human_sim.append(np.mean(scr))

    return scipy.stats.spearmanr(human_sim, avgSim_sim)[0], scipy.stats.spearmanr(human_sim, maxSim_sim)[0]


def scws_eval(multiEmb_ifile, centerEmb_ifile, topicEmb_ifile):
    _, _, _, score, wdPair = init_scws()

    wt_pair = scws_wtPair()
    _, _, lookupIndices = get_indices()
    embed = np.loadtxt(multiEmb_ifile)  
    spearman_score = getScore(score, wt_pair, lookupIndices, embed)

    centerVec = np.loadtxt(centerEmb_ifile) 
    topicVec = np.loadtxt(topicEmb_ifile) 
    word_topic_dist, doc_topic_dist = dt_wt_dist()
    unique_token = get_scws_unqToken()
    avgsim_score, maxsim_score = avgSim_maxSimC(score, centerVec, topicVec, word_topic_dist, doc_topic_dist, wdPair,
                                                unique_token)

    return spearman_score, avgsim_score, maxsim_score
