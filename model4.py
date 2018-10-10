import tensorflow as tf
import numpy as np


#using tensorflow negative sampling softmax,w/out wt_dist 


class MVWEModel():

    def __init__(self, batch_size, vocab_size, n_topic, embed_size, n_negSample, beta, learning_rate, wTopic_dist):
        self._batchSize = batch_size
        self._vocabSize = vocab_size
        self._nTopic = n_topic
        self._embedSize = embed_size
        self._nNegSample = n_negSample
        self._beta = beta
        self._lr = learning_rate
        self._wTopicDist = wTopic_dist  # added in v2
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

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

            #added in v2
            self.w_t_dist = tf.Variable(self._wTopicDist, trainable=False, name='word_topic_dist')  # shape = [vocab_size, n_topic]
            self.weight_wordTopic = tf.matmul(self.w_t_dist, self.weight_topic, name='word_topic_embed')  # shape = [vocab_size, embed_size

            tf.summary.histogram('centerW', self.weight_center)
            tf.summary.histogram('contextW', self.weight_context)
            tf.summary.histogram('topicW', self.weight_topic)


    def create_bias(self):
        with tf.name_scope('bias'):
            self.bias_center = tf.Variable(tf.zeros([self._embedSize]), name='center_bias')
            self.bias_context = tf.Variable(tf.zeros([self._embedSize]), name='context_bias')
            self.bias_topic = tf.Variable(tf.zeros([self._embedSize]), name='topic_bias')
            self.bias = tf.Variable(tf.zeros([self._vocabSize]), name='topic_bias')

            tf.summary.histogram('totalbias', self.bias)




    def inference(self):
        # shape = [batch_size * n_neg+1 , embed_size]
        self.center_embed = tf.nn.embedding_lookup(self.weight_center, self.X_center, name='center_embed') + self.bias_center
        self.context_embed = tf.nn.embedding_lookup(self.weight_context, self.X_context, name='context_embed') + self.bias_context

        #v1:
        #self.topic_embed = tf.nn.embedding_lookup(self.weight_topic, self.X_topic, name='topic_embed') + self.bias_topic

        # v2:
        self.topic_embed = tf.nn.embedding_lookup(self.weight_wordTopic, self.X_center, name='topic_embed') + self.bias_topic


    def create_loss(self):

        # shape = [batch_size * n_neg+1 , embed_size]
        word_topic = tf.mul(self.center_embed, self.topic_embed)
        train_labels = tf.reshape(tf.cast(self.X_context, dtype=tf.int64), shape=[self._batchSize, 1])

        sample_vals = tf.nn.learned_unigram_candidate_sampler(true_classes=train_labels,num_true=1,
                                                              num_sampled=self._nNegSample, unique=True,range_max=self._vocabSize)

        loss_batch = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=self.weight_context, biases=self.bias, inputs=word_topic,
                                       labels=train_labels, num_sampled=self._nNegSample, num_classes=self._vocabSize, sampled_values=sample_vals))


        regularizer = tf.nn.l2_loss(self.weight_center) + tf.nn.l2_loss(self.weight_context) + tf.nn.l2_loss(
            self.weight_topic)
        self.loss = tf.reduce_mean(loss_batch + self._beta * regularizer)

        tf.summary.scalar('loss', self.loss)


    def create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self._lr).minimize(self.loss, global_step=self.global_step)


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
    initial_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('tmp/mvwe_demo/1')
        writer.add_graph(sess.graph)
        total_loss = 0.0
        initial_step = model.global_step.eval()
        for i in range(initial_step, initial_step + n_epochs):
            batch_center, batch_context, batch_topic = next(batch_generation)

            if i%5 == 0:
                s = sess.run(merged_summary,
                             feed_dict={model.X_center: batch_center, model.X_context: batch_context, model.X_topic: batch_topic})
                writer.add_summary(s,i)
            batch_loss, _ = sess.run([model.loss, model.optimizer],
                                     feed_dict={model.X_center: batch_center, model.X_context: batch_context, model.X_topic: batch_topic})
            total_loss += batch_loss

            if (i+1) % skip_step == 0:
                print 'average loss epoch {0} : {1}'.format(i, total_loss/skip_step)
                total_loss = 0.0

       
        word_indices, topic_indices, _ = wt_indices
        multi_embed = sess.run(model.normalized_word_multi_embed, feed_dict={model.w_indices: word_indices, model.t_indices: topic_indices})
        
