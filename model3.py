import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import os


#using gensim table for negative sampling, with/out wt_dist 
 
class MVWEModel():

    def __init__(self, batch_size, vocab_size, n_topic, embed_size, n_negSample, beta, wTopic_dist, learnin_rate):
        self._batchSize = batch_size
        self._vocabSize = vocab_size
        self._nTopic = n_topic
        self._embedSize = embed_size
        self._nNegSample = n_negSample
        self._beta = beta
        self._wTopicDist = wTopic_dist #added in v2
        self._lr = learnin_rate
        #self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def create_placeholders(self):
        with tf.name_scope('input_data'):
            self.X_center = tf.placeholder(tf.int32, shape=[self._batchSize * (self._nNegSample+1)], name='center_words')
            self.X_context = tf.placeholder(tf.int32, shape=[self._batchSize * (self._nNegSample+1)], name='context_words')
            self.X_topic = tf.placeholder(tf.int32, shape=[self._batchSize * (self._nNegSample+1)],name='topic_words')
            self.Neg_idx = tf.placeholder(tf.int32, shape = [self._batchSize, self._nNegSample+1], name = 'neg_samples')
            #self.lr = tf.placeholder(tf.float32, (), name= 'learning_rate')

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




    def create_bias(self):
        with tf.name_scope('bias'):
            self.bias_center = tf.Variable(tf.zeros([self._embedSize]), name='center_bias')
            self.bias_context = tf.Variable(tf.zeros([self._embedSize]), name='context_bias')
            self.bias_topic = tf.Variable(tf.zeros([self._embedSize]), name='topic_bias')





    def inference(self):
        # shape = [batch_size * n_neg+1 , embed_size]
        self.center_embed = tf.nn.embedding_lookup(self.weight_center, self.X_center, name='center_embed') + self.bias_center
        # v1:
        # self.topic_embed = tf.nn.embedding_lookup(self.weight_topic, self.X_topic, name='topic_embed') + self.bias_topic

        #v2:
        self.topic_embed = tf.nn.embedding_lookup(self.weight_wordTopic, self.X_center, name='topic_embed') + self.bias_topic


    def create_loss(self):

        # shape = [batch_size * n_neg+1 , embed_size]
        word_topic = tf.mul(self.center_embed, self.topic_embed)

        #flatten the batch matrix of negative ids of shape batch_size*n_neg+1
        neg_Ids = tf.reshape(self.Neg_idx, [-1])

        #shape = [batch_size * n_neg+1 , embed_size]
        neg_samples = tf.nn.embedding_lookup(self.weight_context, neg_Ids, name = 'nois_samples')

        #shape = [batchsize* n_neg+1, 1]
        tmp = tf.reduce_sum(tf.mul(word_topic, neg_samples), axis=1)

        true_corrupt_score = tf.reshape(tmp, [self._batchSize, self._nNegSample + 1])

        activate = tf.sigmoid(true_corrupt_score, name = 'activation')

        sign_vec = np.ones(self._nNegSample + 1)
        sign_vec[0] = -1

        max_min = activate * sign_vec
        loss_matrix = tf.reduce_sum(max_min, axis=1)
        loss_batch = tf.reduce_sum(loss_matrix)

        regularizer = tf.nn.l2_loss(self.weight_center) + tf.nn.l2_loss(self.weight_context) + tf.nn.l2_loss(
            self.weight_topic)
        self.loss = tf.reduce_mean(loss_batch + self._beta * regularizer)



    def create_optimizer(self):
        '''
         with tf.name_scope('learning_rate'):
            decay_step = 100000
            decay_rate = 0.95
            self.learning_rate = tf.train.exponential_decay(self._initLR, self.global_step, decay_step, decay_rate, staircase=True)

        '''


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self.loss)


    def create_summary(self):
        tf.summary.histogram('centerWeight', self.weight_center)
        tf.summary.histogram('contextWeight', self.weight_context)
        tf.summary.histogram('topicWeight', self.weight_topic)

        tf.summary.histogram('centerBias', self.bias_center)
        tf.summary.histogram('contextBias', self.bias_context)
        tf.summary.histogram('topicBias', self.bias_topic)

        tf.summary.scalar('loss', self.loss)
        self.merged_summary = tf.summary.merge_all()

    def build_graph(self):
        self.create_placeholders()
        self.create_factorMatrix()
        self.create_bias()
        self.inference()
        self.create_loss()
        self.create_optimizer()
        #self.create_summary()


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
    #initial_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #writer = tf.summary.FileWriter('/tmp/mvwe_demo5/1')
        #writer.add_graph(sess.graph)
        total_loss = 0.0
        #initial_step = model.global_step.eval()
        for i in range(n_epochs):
            batch_center, batch_context, batch_topic, batch_negSample = next(batch_generation)
            #learning_rate = 0.01

            batch_loss, _ = sess.run([model.loss, model.optimizer],
                                     feed_dict={model.X_center: batch_center, model.X_context: batch_context, model.X_topic: batch_topic, model.Neg_idx: batch_negSample})

            #writer.add_summary(summary)
            total_loss += batch_loss

            if (i+1) % skip_step == 0:
                print ('average loss epoch {0} : {1}'.format(i, total_loss/skip_step))
                total_loss = 0.0


        word_indices, topic_indices, _ = wt_indices
        multi_embed = sess.run(model.normalized_word_multi_embed, feed_dict={model.w_indices: word_indices, model.t_indices: topic_indices})

       

        '''
        #visualziation
        embedding_var = tf.Variable(multi_embed[:1000], name='embedding')
        sess.run(embedding_var.initializer)
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('/tmp/embed_visual', sess.graph)
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        os.path.join('/tmp/embed_visual/', 'news_metadata.tsv')
        embedding.metadata_path = os.path.join('/tmp/embed_visual/', 'news_metadata.tsv')
        projector.visualize_embeddings(summary_writer,config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, '/tmp/embed_visual/model3.ckpt', 1)
        '''
