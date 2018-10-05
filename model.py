import tensorflow as tf
import numpy as np

class MVWEModel():

    def __init__(self, batch_size, vocab_size, n_topic, embed_size, wTopic_dist, n_negSample, unigrams, beta, lr):
        self._batchSize = batch_size
        self._vocabSize = vocab_size
        self._nTopic = n_topic
        self._embedSize = embed_size
        self._wTopicDist = wTopic_dist
        self._nNegSample = n_negSample
        self._unigrams = unigrams
        self._beta = beta
        self._lr = lr


    def create_placeholders(self):
        with tf.name_scope('input_data'):
            self.X_center = tf.placeholder(tf.int32, shape=[self._batchSize], name='center_words')
            self.X_context = tf.placeholder(tf.int32, shape=[self._batchSize], name='context_words')
            self.X_topic = tf.placeholder(tf.int32, shape=[self._batchSize],name='topic_words')

    def create_factorMatrix(self):
        with tf.name_scope('embeddings_matrix'):
            self.weight_center = tf.Variable(
                tf.random_uniform(shape=[self._vocabSize, self._embedSize], minval=-0.1, maxval=0.1), name='center_embed')
            self.weight_context = tf.Variable(
                tf.random_uniform(shape=[self._vocabSize, self._embedSize],  minval=-0.1, maxval=0.1), name='context_embed')
            self.weight_topic = tf.Variable(
                tf.random_uniform(shape=[self._nTopic, self._embedSize], minval=-0.1, maxval=0.1), name='topic_embed')

            self.w_t_dist = tf.Variable(self._wTopicDist, trainable=False, name='word_topic_dist')  #shape = [vocab_size, n_topic]
            self.weight_wordTopic = tf.matmul(self.w_t_dist, self.weight_topic, name='word_topic_embed') #shape = [vocab_size, embed_size]


    def create_bias(self):
        with tf.name_scope('bias'):
            self.bias_center = tf.Variable(tf.zeros([self._embedSize]), name='center_bias')
            self.bias_context = tf.Variable(tf.zeros([self._embedSize]), name='context_bias')
            self.bias_topic = tf.Variable(tf.zeros([self._embedSize]), name='topic_bias')

    def inference(self):
        # shape = [batch_size, embed_size]
        self.center_embed = tf.nn.embedding_lookup(self.weight_center, self.X_center, name='center_embed') + self.bias_center
        self.context_embed = tf.nn.embedding_lookup(self.weight_context, self.X_context, name='context_embed') + self.bias_context
        self.topic_embed = tf.nn.embedding_lookup(self.weight_wordTopic, self.X_center, name='topic_embed') + self.bias_topic

    def create_loss(self):

        # shape = [batch_size, embed_size]
        center_context = tf.mul(self.center_embed, self.context_embed)
        center_context_topic = tf.mul(center_context, self.topic_embed)

        # shape = [batch_size,1]
        true_score = tf.reduce_sum(center_context_topic, reduction_indices=[1], keep_dims=True)

        true_labels = tf.reshape(tf.cast(self.X_context, dtype=tf.int64), shape=[self._batchSize, 1])
        word_topic = tf.mul(self.center_embed, self.topic_embed)

        # negative samples
        neg_samples_id, _, _ = tf.nn.fixed_unigram_candidate_sampler(true_classes=true_labels, num_true=1,
                                                                     num_sampled=self._nNegSample,
                                                                     unique=True, range_max=self._vocabSize, distortion=0.75,
                                                                     unigrams=self._unigrams)
        # get embedding of negative samples, shape = [num_neg_samples, embed_size]
        neg_embed = tf.nn.embedding_lookup(self.weight_context, neg_samples_id)
        # shape = [batch_size, num_neg_sample]
        corrupted_score = tf.matmul(word_topic, neg_embed, transpose_b=True)

        # shape = [batch_size, num_neg_samples]
        loss_matrix = tf.maximum(0., 1. - true_score + corrupted_score)
        loss_batch = tf.reduce_sum(loss_matrix)  # this gives scalar representing the loss over the batch

        regularizer = tf.nn.l2_loss(self.weight_center) + tf.nn.l2_loss(self.weight_context) + tf.nn.l2_loss(self.weight_topic)
        self.loss = tf.reduce_mean(loss_batch + self._beta * regularizer)


    def create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self._lr).minimize(self.loss)


    def build_graph(self):
        self.create_placeholders()
        self.create_factorMatrix()
        self.create_bias()
        self.inference()
        self.create_loss()
        self.create_optimizer()


    def generate_single_embed(self):
        self.normalized_weightCenter = tf.nn.l2_normalize(self.weight_center,1)
        self.normalized_weightTopic = tf.nn.l2_normalize(self.weight_topic, 1)

    def generate_multi_embed(self):

        self.w_indices = tf.placeholder(dtype=tf.int32)
        self.t_indices = tf.placeholder(dtype=tf.int32)
        f_w_embeding = tf.nn.embedding_lookup(self.weight_center, self.w_indices)
        f_t_embeding = tf.nn.embedding_lookup(self.weight_topic, self.t_indices)

        word_multi_embed = tf.mul(f_w_embeding, f_t_embeding)
        self.normalized_word_multi_embed = tf.nn.l2_normalize(word_multi_embed, 1)



def train_MVWE(model,n_epochs,batch_generation, wt_indices, skip_step):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0
        for i in range(n_epochs):

            batch_center, batch_context, batch_topic = batch_generation.next()
            batch_loss, _ = sess.run([model.loss, model.optimizer],
                                     feed_dict={model.X_center: batch_center, model.X_context: batch_context, model.X_topic: batch_topic})
            total_loss += batch_loss

            if (i+1) % skip_step == 0:
                print 'average loss epoch {0} : {1}'.format(i, total_loss/skip_step)
                total_loss = 0

        centerEmbed = sess.run(model.normalized_weightCenter)
        np.savetxt('/tmp/wikiCenterVec.txt', centerEmbed)
        topicEmbed = sess.run(model.normalized_weightTopic)
        np.savetxt('/tmp/wikiTopicVec.txt', topicEmbed)

        word_indices, topic_indices, _ = wt_indices
        multi_embed = sess.run(model.normalized_word_multi_embed, feed_dict={model.w_indices: word_indices, model.t_indices: topic_indices})
        #multi_embed = model.normalized_word_multi_embed.eval()
        np.savetxt('/tmp/newsTrainVec.txt', multi_embed)