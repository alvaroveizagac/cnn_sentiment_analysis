#!usr/bin/env python
import tensorflow as tf
import os, utils
import numpy as np
import argparse
import time
# import tf.data.Dataset
from cnn_shallow_architecture import cnn_shallow_arch
from cnn_test import testing_step
from tensorflow.contrib.data import Dataset, Iterator

def splitDataset(split_ratio, dataset):
    split_index = int(len(x) * (split_ratio / 100.0))
    part_a, part_b = dataset[split_index:], dataset[:split_index]
    return part_a, part_b


def shuffleDataset(dataset_x, dataset_y):
    shuffle_index_dataset = np.random.permutation(np.arange(len(dataset_x)))
    dataset_shuffle_x = dataset_x[shuffle_index_dataset]
    dataset_shuffle_y = dataset_y[shuffle_index_dataset]
    return dataset_shuffle_x, dataset_shuffle_y


def log(output, log_file):
    # output = ' '.join(string)
    print(output)
    log_file.write(''.join(['\n', output]))


def create_log_file(log_file_path):
    try:
        log_file = open(log_file_path, 'a')
    except:
        print("Failed to open file.")
        quit()
    return log_file


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_full_path_directory(directory):
    return os.path.abspath(os.path.join(os.path.curdir, directory))


def get_size(dataset):
    return dataset.shape[1]


def calculate_every_frequency(dataset, batch_size, frequency):
    every_frequency = int(len(dataset) / batch_size)
    every_frequency = (every_frequency + 1 if len(dataset) % batch_size != 0.0 else every_frequency) * frequency
    return every_frequency


def training_step(x_batch, y_batch, batches_in_epoch):
    feed_dict = {cnn.x_place_holder: x_batch, cnn.y_place_holder: y_batch,
                 cnn.emb_place_holder: vocab_inv_emb_dset,
                 cnn.dropout_keep_prob: args.dropout_rate, cnn.learning_rate: args.learning_rate_constant,
                 cnn.decay_rate: args.decay_rate}
    # _, summary_training, accuracy_train, globalStep, loss_train, matrix_conf, prec_batch, recall_batch, f1_batch, scor \
    #     = sess.run([cnn.train_step, cnn.summary, cnn.accuracy, cnn.global_step_, cnn.cross_entropy,
    #                 cnn.matrix, cnn.precision_mini_batch, cnn.recall_mini_batch,
    #                 cnn.f1_score_min_batch, cnn.scores], feed_dict)

    _, accuracy_train, globalStep, loss_train, matrix_conf, prec_batch, recall_batch, f1_batch, scor, los_step, \
    acu_step, prec_step, rec_step, f1_step = sess.run([cnn.train_step, cnn.accuracy, cnn.global_step_,
                                                       cnn.cross_entropy, cnn.matrix, cnn.precision_mini_batch,
                                                       cnn.recall_mini_batch, cnn.f1_score_min_batch, cnn.scores,
                                                       cnn.stream_loss_update, cnn.stream_accuracy_update,
                                                       cnn.stream_precision_update, cnn.stream_recall_update,
                                                       cnn.stream_f1_update],
                                                      feed_dict)
    # print(str(los_step)+" "+str(acu_step)+" "+str(prec_step)+" "+str(rec_step)+" "+str(f1_step))

    current_epoch = globalStep / batches_in_epoch if globalStep % batches_in_epoch == 0 else 1 + (
        globalStep / batches_in_epoch)
    # losses_accum_per_epoch.append(loss_train)
    # summary_training_writer.add_summary(summary_training, globalStep)

    log("Step %d of %d (epoch %d), training accuracy: %g, precision: %g, recall: %g, f1: %g,  loss: %g" % (
        globalStep, total_num_step, current_epoch, accuracy_train, prec_batch, recall_batch,
        f1_batch, loss_train), LOG_FILE)
    return loss_train, current_epoch, globalStep


def validation_step(x_val_batch, y_val_batch, current_epoch):
    feed_dict = {cnn.x_place_holder: x_val_batch, cnn.y_place_holder: y_val_batch,
                 cnn.emb_place_holder: vocab_inv_emb_dset, cnn.dropout_keep_prob: 1.0,
                    cnn.learning_rate: args.learning_rate_constant, cnn.decay_rate: args.decay_rate}
    accuracy_validation, loss_validation, matrix_conf_val, precision_validation, recall_validation,\
    f1_validation, step_global = sess.run([cnn.accuracy, cnn.cross_entropy, cnn.matrix, cnn.precision_mini_batch,
                                           cnn.recall_mini_batch, cnn.f1_score_min_batch, cnn.global_step_], feed_dict)

    current_epoch = step_global / batches_in_epoch if step_global % batches_in_epoch == 0 else 1 + (
        step_global / batches_in_epoch)
    # summary_val_writer.add_summary(summary_validation, current_epoch)
    log("Step %d of %d (epoch %d), Validation accuracy: %g, loss: %g, precission: %g, recall: %g, F1: %g" % (
            global_step, total_num_step, current_epoch, accuracy_validation, loss_validation, precision_validation,
            recall_validation, f1_validation), LOG_FILE)
    return loss_validation, accuracy_validation, precision_validation, recall_validation, f1_validation


def calculate_number_batches_dataset(instances, batch_size, epochs):
    batches_epoch = calculate_number_batches_epoch(instances, batch_size)
    number_batches_dataset = epochs * batches_epoch
    return number_batches_dataset, batches_epoch


def calculate_number_batches_epoch(instances, batch_size):
    batches_in_epoch = len(instances) / batch_size
    if len(instances) % batch_size != 0:
        batches_in_epoch = batches_in_epoch + 1
    else:
        batches_in_epoch = batches_in_epoch
    return batches_in_epoch


np.set_printoptions(threshold=np.nan)

parser = argparse.ArgumentParser(description="Hyper-parameters")
parser.add_argument("--embedding_size", help="Vector embedding dimension", type=int, default=100)
parser.add_argument("--filter_sizes", help="CNN filter sizes", type=str, default="2,3,4")
parser.add_argument("--num_filters", help="Number of filters", type=int, default=128)
parser.add_argument("--batch_size", help="Batch size", type=int, default=50)
parser.add_argument("--epochs", help="Number of epochs", type=int, default=4)
parser.add_argument("--dropout_rate", help="Dropout rate value", type=float, default=0.50)
parser.add_argument("--l2_reg_lambda", help="L2 regularization parameter", type=float, default=0.005)
parser.add_argument("--learning_rate_constant", help="Learning rate constant", type=float, default=0.01)
parser.add_argument("--decay_rate", help="Learning rate decay", type=float, default=0.95)
parser.add_argument("--activation_func", help="Activation function", type=str, default="relu")
parser.add_argument("--valid_freq", help="Every how many epochs the validation is executed", type=int, default=1)
parser.add_argument("--checkpoint_freq", help="Every how many epochs the model is saved", type=int, default=1)
parser.add_argument("--pad_to_tweets", help="The number of words that a document should have", type=int, default=42)
parser.add_argument("--strip_stop_earlier", help="Parameter to check the early stop", type=int, default=5)
parser.add_argument("--threshold_estop", help="Threshold for early stop", type=float, default=0.75)
parser.add_argument("--early_stop_freq", help="Early stop frequency", type=int, default=2)
parser.add_argument("--results_directory", help="Directory to save all the results", type=str, default="output/output_301")
load = ''
args = parser.parse_args()

create_directory(args.results_directory)
output_directory_path = get_full_path_directory(args.results_directory)

url_pretrain_emb = "/home/alvaro/Desktop/NNets/thesis/cnn1_6M/smallDataSet/dataset/glove.twitter.27B.100d_smalldset.txt"

url_directory = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/oversample2017_test_train/"
url_dataset = url_directory+"semeval_over_preproc_training.csv"
url_dataset_vocab = url_directory+"vocab_train_semeval2017_preproc_37words.csv"
url_dataset_vocab_inv = url_directory+"vocab_inv_train_semeval2017_preproc_37words.csv"


log_file_path = get_full_path_directory(args.results_directory + "/log_drop.log")
checkpoint_file_path = get_full_path_directory(args.results_directory + "/checkpoint.ckpt")
summary_dir = get_full_path_directory(args.results_directory + "/summaries")
validation_dir = get_full_path_directory(args.results_directory + "/summaries_validation")
run_dir = get_full_path_directory(args.results_directory + "/" + load)
should_load = os.path.exists(run_dir)

LOG_FILE = create_log_file(log_file_path)
log(str(args), LOG_FILE)
log("== START  ==", LOG_FILE)
log("Preprocessing..", LOG_FILE)

# loading dataset training and test
x, y, vocab_dset, vocab_inv_dset = utils.load_data_network(
    url_dataset,
    url_dataset_vocab,
    url_dataset_vocab_inv, args.pad_to_tweets)
# Creating a subset of the embedding
vocab_inv_emb_dset = utils.getEmbDictAndEmbDataset(vocab_inv_dset, url_pretrain_emb, args.embedding_size)
print(np.shape(vocab_inv_emb_dset))

np.random.seed(123)
x_shuffled, y_shuffled = shuffleDataset(x, y)


# Split 60 / 40
split_index = int(len(x_shuffled) * (40 / 100.0))
x_train, x_val_test = x_shuffled[split_index:], x_shuffled[:split_index]
y_train, y_val_test = y_shuffled[split_index:], y_shuffled[:split_index]

# Split 40 into dev and test sets
split_index_val_test = int(len(x_val_test) * (50 / 100.0))
x_val, x_test = x_val_test[split_index_val_test:], x_val_test[:split_index_val_test]
y_val, y_test = y_val_test[split_index_val_test:], y_val_test[:split_index_val_test]


training_data = (Dataset.from_tensor_slices((x_train, y_train))
                 .shuffle(buffer_size=10)
                 .batch(args.batch_size)
                 .make_initializable_iterator())
validation_data = (Dataset.from_tensor_slices((x_val, y_val))
                   .shuffle(buffer_size=10)
                   .batch(args.batch_size)
                   .make_initializable_iterator())

next_element_training = training_data.get_next()
next_element_validation = validation_data.get_next()

del x, y

words_per_document = get_size(x_train)
num_classes = get_size(y_train)
vocabulary_size = len(vocab_dset)
filter_sizes = list(map(int, args.filter_sizes.split(",")))

validate_every = calculate_every_frequency(y_train, args.batch_size, args.valid_freq)
early_stop_every = calculate_every_frequency(y_train, args.batch_size, args.early_stop_freq)
checkpoint_every = calculate_every_frequency(y_train, args.batch_size, args.checkpoint_freq)

log("\nDataset:", LOG_FILE)
log("\tTrain set size = %d\n\tValidation  set size = %d\n\tTest set size = %d\n\t"
    "Input layer size(largest tweet) = %d\n\tNumber of classes = %d" % (len(y_train), len(y_val), len(y_test),
                                                                        words_per_document, num_classes), LOG_FILE)
start = time.time()
session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

with tf.Session(config=session_conf) as sess:
    log("Starting training....", LOG_FILE)
    cnn = cnn_shallow_arch(
        words_per_document=words_per_document, num_classes=num_classes, vocabulary_size=vocabulary_size,
        embedding_size=args.embedding_size, filter_sizes=filter_sizes, num_filters=args.num_filters,
        l2_reg_lambda=args.l2_reg_lambda, train_size=len(x_train), batch_size=args.batch_size)
    # sess.run(init_op)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # batches = utils.batch_iter(list(zip(x_train, y_train)), args.batch_size, args.epochs)
    # # when the dataset is big then we have to split the validation/development set
    # validation_batches = list(utils.batch_iter(list(zip(x_val, y_val)), args.batch_size, 1))

    #global_step = 0
    total_num_step, batches_in_epoch = calculate_number_batches_dataset(y_train, args.batch_size, args.epochs)
    summary_training_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    summary_training_writer.add_graph(sess.graph)
    summary_val_writer = tf.summary.FileWriter(validation_dir, sess.graph)
    summary_val_writer.add_graph(sess.graph)

    early_stop = False
    counter_bad_loss = 0

    # Training loop
    counterbatch = 0
    losses_epoch = []
    losses_epoch_validation = []
    losses_epoch_training = []
    losses_accum_per_epoch = []

    # Optimization
    for actual_epoch in range(args.epochs):
        sess.run(training_data.initializer)
        while True:
            try:
                batch = sess.run(next_element_training)
                loss_train_per_batch, current_epoch, global_step = training_step(batch[0], batch[1], batches_in_epoch)
                losses_accum_per_epoch.append(loss_train_per_batch)

            except tf.errors.OutOfRangeError:
                if (actual_epoch + 1) % args.valid_freq == 0:
                    avg_loss_per_eppoch = np.mean(losses_accum_per_epoch)
                    losses_epoch_training.append(avg_loss_per_eppoch)
                    losses_accum_per_epoch = []

                    feed_dict = {cnn.x_place_holder: batch[0], cnn.y_place_holder: batch[1],
                                 cnn.emb_place_holder: vocab_inv_emb_dset, cnn.dropout_keep_prob: 1.0}
                    summary_epoch, step_epoch = sess.run([cnn.summary, cnn.global_step_], feed_dict)
                    summary_training_writer.add_summary(summary_epoch, (actual_epoch+1))

                    # it is not a good idea to run this every epoch
                    # sess.run(tf.local_variables_initializer())

                print("End training dataset epoch: "+str(actual_epoch + 1))
                break

        if (actual_epoch + 1) % args.valid_freq == 0:
            validation_total_accuracy = []
            validation_total_error = []
            validation_total_precision = []
            validation_total_recall = []
            validation_total_f1 = []
            sess.run(validation_data.initializer)
            # Validations step
            log("Step" + str(global_step) + "  - Validation...", LOG_FILE)
            while True:
                try:
                    batch_val = sess.run(next_element_validation)
                    val_batch = sess.run(next_element_validation)
                    loss_val, acc_val, prec_val, recall_val, f1_val = validation_step(val_batch[0], val_batch[1]
                                                                                      , (actual_epoch + 1))
                    validation_total_error.append(loss_val)
                    validation_total_accuracy.append(acc_val)
                    validation_total_precision.append(prec_val)
                    validation_total_recall.append(recall_val)
                    validation_total_f1.append(f1_val)

                    # losses_epoch_validation.append(loss_val)

                except tf.errors.OutOfRangeError:
                    feed_dict_validation = {cnn.accuracy_validation_placeholder: validation_total_accuracy,
                                            cnn.loss_validation_placeholder: validation_total_error,
                                            cnn.precision_validation_placeholder: validation_total_precision,
                                            cnn.recall_validation_placeholder: validation_total_recall,
                                            cnn.f1_validation_placeholder: validation_total_f1}

                    acc_val_mean, loss_val_mean, prec_val_mean, recall_val_mean, f1_val_mean, val_summ = \
                        sess.run([cnn.acc_validation_mean, cnn.loss_validation_mean, cnn.prec_validation_mean,
                                  cnn.recall_validation_mean, cnn.f1_validation_mean, cnn.summary_val],
                                 feed_dict_validation)
                    losses_epoch_validation.append(loss_val_mean)
                    summary_val_writer.add_summary(val_summ, (actual_epoch+1))
                    log("Final Step %d of %d (epoch %d), Validation accuracy: %g,  loss: %g, precission: %g, "
                        "recall: %g, F1: %g" % (global_step, total_num_step, current_epoch, acc_val_mean,
                                                loss_val_mean, prec_val_mean, recall_val_mean, f1_val_mean), LOG_FILE)
                    break
        # Early Stop
        early_stop_log = "No E_Stop "
        if (actual_epoch+1) % args.early_stop_freq == 0:
            log("Earlystop checking: ", LOG_FILE)
            early_stop_value = utils.calc_earlystop_pq(losses_epoch_training, losses_epoch_validation,
                                                       (actual_epoch+1), args.strip_stop_earlier)
            if early_stop_value > args.threshold_estop:
                early_stop = True
                log("Early Stop the current model will not be saved.", LOG_FILE)
                log(("Early stop value %d" % early_stop_value), LOG_FILE)
                early_stop_log = "E_Stop: epoch "+str(actual_epoch+1)+" value "+str(early_stop_value)
                break

        if early_stop:
            break
        if (actual_epoch+1) % args.checkpoint_freq == 0:
            log("Saving checkpoint...", LOG_FILE)
            saver = tf.train.Saver()
            saver.save(sess, checkpoint_file_path)

testing_step(args.results_directory + "/", args.batch_size, x_test, y_test, vocab_inv_emb_dset, LOG_FILE, args,
             early_stop_log)
end = time.time()
log("Elapsed time: "+str(end - start), LOG_FILE)

