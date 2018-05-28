#!usr/bin/env python
import tensorflow as tf
import utils
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.data import Dataset, Iterator


def log(output, log_file):
    # output = ' '.join(string)
    print(output)
    log_file.write(''.join(['\n', output]))


def testing_step(checkpoint_dir, batch_size, x_test, y_test, vocab_inv_emb_dset, LOG_FILE, parameters, early_stop_log):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            # all_vars = graph.get_operations()
            x_placeholder = graph.get_operation_by_name("x").outputs[0]
            y_placeholder = graph.get_operation_by_name("labels").outputs[0]
            embedding_placeholder = graph.get_operation_by_name("emb_place_holder").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            accuracies = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
            loss = graph.get_operation_by_name("loss/cross_entropy").outputs[0]
            label_match = graph.get_operation_by_name("confussion_matrix/label_max").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            l2_loss = graph.get_operation_by_name("l2_loss").outputs[0]

            testing_data = (Dataset.from_tensor_slices((x_test, y_test))
                            .shuffle(buffer_size=10)
                            .batch(parameters.batch_size)
                            .make_initializable_iterator())
            next_element_testing = testing_data.get_next()

            epochs = 1
            step = 0
            accuracies_test = []
            losses_test = []
            precisions_test = []
            recalls_test = []
            f1_scores_test = []
            for actual_epoch in range(epochs):
                sess.run(testing_data.initializer)
                while True:
                    try:
                        batch_testing = sess.run(next_element_testing)

                        feed_dict = {x_placeholder: batch_testing[0], y_placeholder: batch_testing[1],
                                     embedding_placeholder: vocab_inv_emb_dset, dropout_keep_prob: 1.0}

                        acc_batch, loss_batch, label_op, pred_op, l2_loss_op = \
                            sess.run([accuracies, loss, label_match, predictions, l2_loss], feed_dict)

                        matrix_batch = confusion_matrix(label_op, pred_op)

                        true_positive = matrix_batch[1, 1]
                        true_negative = matrix_batch[0, 0]
                        false_positive = matrix_batch[0, 1]
                        false_negative = matrix_batch[1, 0]
                        precision_mini_batch = utils.calculate_precision(true_positive, false_positive)
                        recall_mini_batch = utils.calculate_recall(true_positive, false_negative)
                        f1_score_min_batch = utils.calculate_f1(precision_mini_batch, recall_mini_batch)

                        accuracies_test.append(acc_batch)
                        losses_test.append(loss_batch)
                        precisions_test.append(precision_mini_batch)
                        recalls_test.append(recall_mini_batch)
                        f1_scores_test.append(f1_score_min_batch)

                        log("Step " + str(step) + "(epoch " + str(epochs) + ")" + "Test accuracy: " + str(
                            acc_batch) +
                            " test loss: " + str(loss_batch) + " test precission: " + str(precision_mini_batch) +
                            " test recall: " +   str(recall_mini_batch) + "test F1: " + str(f1_score_min_batch), LOG_FILE)

                    except tf.errors.OutOfRangeError:
                        avg_accuracy = np.mean(accuracies_test)
                        avg_losses = np.mean(losses_test)
                        avg_precision = np.mean(precisions_test)
                        avg_recall = np.mean(recalls_test)
                        avg_f1 = np.mean(f1_scores_test)
                        log(str(parameters), LOG_FILE)
                        log("Final results, test accuracy: " + str(avg_accuracy) + " test loss: " + str(avg_losses) +
                            " test precission: " + str(avg_precision) + " test recall: " + str(
                            avg_recall) + " test f1: "
                            + str(avg_f1), LOG_FILE)
                        log("End training dataset epoch: " + str(early_stop_log), LOG_FILE)
                        break
