
from __future__ import division

import tensorflow as tf
import numpy as np

class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16,
                 prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self._end = word_to_idx['<END>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-0.5, maxval=0.5)
        self.senti_initializer = tf.random_uniform_initializer(minval=-0.5, maxval=0.5)

        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.sample_caption = tf.placeholder(tf.int32, [None, self.T])

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [512, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [512, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')
            return x

    def _int_embedding(self, inputs, reuse=False):
        with tf.variable_scope('int_embedding', reuse=reuse):
            w = tf.get_variable('w', [3, 256], initializer=self.senti_initializer, trainable=True)
            x = tf.nn.embedding_lookup(w, inputs, name='int_vector')
            x = x[:,0,:]
            return x

    def _ext_embedding(self, inputs, reuse=False):
        with tf.variable_scope('ext_embedding', reuse=reuse):
            w = tf.get_variable('w', [3, 256], initializer=self.senti_initializer, trainable=True)
            x = tf.nn.embedding_lookup(w, inputs, name='ext_vector')
            x = x[:,0,:]
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [512, 512], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, 512])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, 512])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, 512], initializer=self.weight_initializer)
            b = tf.get_variable('b', [512], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [512, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, 512]), w_att), [-1, self.L])
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')  # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, features_senti, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):

            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [512, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)

            out_logits = tf.matmul(h_logits, w_out) + b_out

            w_ctx2out_2 = tf.get_variable('w_ctx2out_2', [256, self.V], initializer=self.weight_initializer)
            out_logits += tf.matmul(features_senti, w_ctx2out_2)

            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, 4:self.T]
        captions_out = captions[:, 5:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        features_category = tf.cast(self.captions[:, 3:4], tf.int32)

        features_int = self._int_embedding(inputs=features_category)
        features_ext = self._ext_embedding(inputs=features_category)

        features = self._batch_norm(features[:, :, 0:512], mode='train', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)

        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T-4):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            context = tf.nn.dropout(context, 0.5)
		
            context_lstm = tf.concat([features_int, context], 1)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x[:, t, :], context_lstm], 1), state=[c, h])

            logits = self._decode_lstm(x[:, t, :], h, context, features_ext, dropout=self.dropout, reuse=(t != 0))
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels=captions_out[:, t]) * mask[:, t])

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))
            alphas_all = tf.reduce_sum(alphas, 1)
            alpha_reg = self.alpha_c * tf.reduce_sum((16. / 196 - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features

        features_category = tf.cast(features[:, 1, 515:516], tf.int32)
        features_int = self._int_embedding(inputs=features_category)
        features_ext = self._ext_embedding(inputs=features_category)

        features = self._batch_norm(features[:, :, 0:512], mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        sampled_word = self._start
        for t in range(max_len):

            if sampled_word == self._end:
                break

            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            context_lstm = tf.concat([features_int, context], 1)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context_lstm], 1), state=[c, h])

            logits = self._decode_lstm(x, h, context, features_ext, reuse=(t != 0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))
        return alphas, betas, sampled_captions

    def build_loss(self):
        features = self.features

        features_category = tf.cast(self.sample_caption[:, 3:4], tf.int32)
        features_int = self._int_embedding(inputs=features_category)
        features_ext = self._ext_embedding(inputs=features_category)

        captions = self.sample_caption[:, 4:self.T]
        mask = tf.to_float(tf.not_equal(captions, self._null))

        features = self._batch_norm(features[:, :, 0:512], mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions)
        features_proj = self._project_features(features=features)

        loss = []
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T-4):
            if t == 0:
                word = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                word= x[:, t -1, :]

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            context_lstm = tf.concat([features_int, context], 1)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([word, context_lstm], 1), state=[c, h])

            logits = self._decode_lstm(word, h, context, features_ext, reuse=(t != 0))

            softmax = tf.nn.softmax(logits, dim=-1, name=None)

            loss.append( tf.transpose(tf.multiply(tf.transpose(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(captions[:, t], self.V), [1, 0]),  mask[:, t]), [1, 0]))

        loss_out = tf.transpose(tf.stack(loss), (1, 0, 2))  # (N, T, max_len)

        return loss_out


    def build_multinomial_sampler(self, max_len=16):
        features = self.features

        features_category = tf.cast(features[:, 1, 515:516], tf.int32)
        features_int = self._int_embedding(inputs=features_category )#, reuse=True)
        features_ext = self._ext_embedding(inputs=features_category) #, reuse=True)

        features = self._batch_norm(features[:, :, 0:512], mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        loss = []
        for t in range(self.T-4):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)


            context_lstm = tf.concat([features_int, context], 1)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context_lstm], 1), state=[c, h])

            logits = self._decode_lstm(x, h, context, features_ext, reuse=(t != 0))
            softmax = tf.nn.softmax(logits, dim=-1, name=None)
            sampled_word = tf.multinomial(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)), 1)

            sampled_word = tf.reshape(sampled_word, [-1])
            loss.append(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(tf.identity(sampled_word), self.V))

            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))
        loss_out = tf.transpose(tf.stack(loss), (1, 0, 2))
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))
        return alphas, betas, sampled_captions,loss_out

