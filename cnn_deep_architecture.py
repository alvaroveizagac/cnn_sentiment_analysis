import tensorflow as tf
import utils



class CnnDeepArch(object):

    def __init__(self, words_per_document, num_classes, vocabulary_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda, train_size, batch_size, strides_filters):
        # Placeholders
        self.emb_place_holder = tf.placeholder(tf.float32, [None, embedding_size], name="emb_place_holder")
        self.x_place_holder = tf.placeholder(tf.int32, [None, words_per_document], name="x")
        self.y_place_holder = tf.placeholder(tf.float32, [None, num_classes], name="labels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.decay_rate = tf.placeholder(tf.float32, name="decay_rate")
        l2_loss = tf.constant(0.0, name="l2_loss")

        # First Layer Embeddings [MiniBatchSize, Largest Size, embSize]
        with tf.device('/cpu:0'):
            self.W = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, embedding_size]), trainable=True, name="W")
            embedding_init = self.W.assign(value=self.emb_place_holder)
            self.embedded_chars = tf.nn.embedding_lookup(embedding_init, self.x_place_holder)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Second Layer Convolutional
        x_conv_layer = self.embedded_chars_expanded
        filter_shape_second = embedding_size
        filter_shape_third = 1
        for filter_size in filter_sizes.split(","):
            if filter_size in strides_filters:
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [int(filter_size), filter_shape_second, filter_shape_third, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        x_conv_layer,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="h")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, int(strides_filters[filter_size][0]), 1, 1],
                        strides=[1, int(strides_filters[filter_size][1]), int(strides_filters[filter_size][1]), 1],
                        padding="VALID",
                        name="pool")
                    tf.summary.histogram("weights", W)
                    tf.summary.histogram("biases", b)
                    tf.summary.histogram("activations", h)
                    x_conv_layer = pooled
                    filter_shape_second = 1
                    filter_shape_third = num_filters
            else:
                last_filter = utils.calculate_filter_layer(words_per_document, filter_sizes, strides_filters)
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [int(filter_size), filter_shape_second, filter_shape_third, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        x_conv_layer,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="h")
                    pooled_last = tf.nn.max_pool(
                        h,
                        ksize=[1, last_filter, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    tf.summary.histogram("weights", W)
                    tf.summary.histogram("biases", b)
                    tf.summary.histogram("activations", h)
                    x_conv_layer = pooled_last
            num_filters_total = num_filters * 1
            self.h_pool_flat = tf.reshape(x_conv_layer, [-1, num_filters_total])

        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            tf.summary.histogram("dropout", self.h_drop)

        # Last layer # xw_plus_b: compute  matmul(x, weights) + biases
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            l2_loss += tf.nn.l2_loss(W, name="l2_loss")
            l2_loss += tf.nn.l2_loss(b, name="l2_loss")
            tf.summary.histogram("l2", l2_loss)
            tf.summary.histogram("weigths", W)
            tf.summary.histogram("biases", b)

        # Loss function # cross entropy between what we got and our labels
        with tf.name_scope("loss"):
            cross_entropy_r = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                      labels=self.y_place_holder)
            self.cross_entropy = tf.reduce_mean(cross_entropy_r, name="cross_entropy") + (l2_reg_lambda * l2_loss)
            self.stream_loss, self.stream_loss_update = tf.contrib.metrics.streaming_mean(self.cross_entropy)
            tf.summary.scalar("loss_tr", self.stream_loss)
            # tf.summary.scalar("loss_tr", self.cross_entropy)

        # Train Step # using learning rate 0.0004
        # minimize is it the same as compute_gradients and apply gradients
        with tf.name_scope("train"):
            self.global_step_ = tf.Variable(0, name="global_step", trainable=False)
            lr = tf.train.exponential_decay(
                self.learning_rate,
                self.global_step_ * batch_size,
                train_size,
                self.decay_rate)
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.cross_entropy, global_step=self.global_step_,
                                                                  name="train_oper")

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.predictions, tf.argmax(self.y_place_holder, 1),
                                          name="correct_prediction")
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
            self.stream_accuracy, self.stream_accuracy_update = tf.contrib.metrics.streaming_mean(self.accuracy)
            tf.summary.scalar("accuracy", self.stream_accuracy)
            # tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("confussion_matrix"):
            labels_max = tf.argmax(self.y_place_holder, 1, name="label_max")
            # self.matrix = tf.contrib.metrics.confusion_matrix(labels_max, self.predictions, num_classes=2, name="matrix")
            self.matrix = tf.confusion_matrix(labels_max, self.predictions, num_classes=2, name="matrix")
            true_positive = self.matrix[1, 1]
            true_negative = self.matrix[0, 0]
            false_positive = self.matrix[0, 1]
            false_negative = self.matrix[1, 0]
            self.precision_mini_batch = utils.calculate_precision(true_positive, false_positive)
            self.recall_mini_batch = utils.calculate_recall(true_positive, false_negative)
            self.f1_score_min_batch = utils.calculate_f1(self.precision_mini_batch, self.recall_mini_batch)

            self.stream_precision, self.stream_precision_update = tf.contrib.metrics.streaming_mean(
                self.precision_mini_batch)
            self.stream_recall, self.stream_recall_update = tf.contrib.metrics.streaming_mean(
                self.recall_mini_batch)
            self.stream_f1, self.stream_f1_update = tf.contrib.metrics.streaming_mean(
                self.f1_score_min_batch)
            tf.summary.scalar("Precision", self.stream_precision)
            tf.summary.scalar("Recall", self.stream_recall)
            tf.summary.scalar("F1", self.stream_f1)

            # tf.summary.scalar("Precision", self.precision_mini_batch)
            # tf.summary.scalar("Recall", self.recall_mini_batch)
            # tf.summary.scalar("F1", self.f1_score_min_batch)

        # if should_load:
        #     log("Data processing ok load network...")
        #     saver = tf.train.Saver()
        #     try:
        #         saver.restore(sess, checkpoint_file_path)
        #     except Exception as e:
        #         log("Not able to load file")

        # summaries
        self.summary = tf.summary.merge_all()

        self.accuracy_validation_placeholder = tf.placeholder(tf.float32, name="acc_val_placeholder")
        self.loss_validation_placeholder = tf.placeholder(tf.float32, name="loss_val_placeholder")
        self.precision_validation_placeholder = tf.placeholder(tf.float32, name="prec_val_placeholder")
        self.recall_validation_placeholder = tf.placeholder(tf.float32, name="recall_val_placeholder")
        self.f1_validation_placeholder = tf.placeholder(tf.float32, name="f1_val_placeholder")

        with tf.name_scope("validation"):
            self.acc_validation_mean = tf.reduce_mean(self.accuracy_validation_placeholder)
            self.loss_validation_mean = tf.reduce_mean(self.loss_validation_placeholder)
            self.prec_validation_mean = tf.reduce_mean(self.precision_validation_placeholder)
            self.recall_validation_mean = tf.reduce_mean(self.recall_validation_placeholder)
            self.f1_validation_mean = tf.reduce_mean(self.f1_validation_placeholder)

            loss_val = tf.summary.scalar("loss_val", self.loss_validation_mean)
            accuracy_val = tf.summary.scalar("accuracy_val", self.acc_validation_mean)
            precission_val = tf.summary.scalar("Precision_val", self.prec_validation_mean)
            recall_val = tf.summary.scalar("Recall_val", self.recall_validation_mean)
            f1_val = tf.summary.scalar("F1_val", self.f1_validation_mean)
            self.summary_val = tf.summary.merge([loss_val, accuracy_val, precission_val, recall_val, f1_val])

