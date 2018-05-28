#!usr/bin/env python
from __future__ import division
import io
import csv
import re
import os
import numpy as np


# Method for calculating the precision
def calculate_precision(true_positive, false_positive):
    if (true_positive + false_positive) == 0:
        return 0
    return true_positive / (true_positive + false_positive)


# Method for calculating the recall
def calculate_recall(true_positive, false_negative):
    if(true_positive + false_negative) == 0:
        return 0
    return true_positive/(true_positive + false_negative)


# Method for calculating the F1
def calculate_f1(precision, recall):
    if(precision + recall) == 0:
        return 0
    return 2 * ((precision * recall)/(precision + recall))


# The input file should be separated by "\t" and it should only contain 2 fields: 0 -> sentiment and 1 -> the tweet
# Method that splits a text-line using the delimiter "\t" and returns an specific position. The returned element
# can be also split using delimiter blank space " " when the split_yes variable is set to true
def splitAndstrip(line_text, position, split_yes):
    line_text = line_text.split("\t")[position]
    # line_text = line_text.split(":::")[position]
    line_text = line_text.lower()
    if split_yes:
        line_text = re.sub(r"(\r\n|\r|\n)", "", line_text)
        line_text = line_text.strip()
        line_text = line_text.split(" ")
    return line_text


# The input file should be separated by "\t" and it should only contain 2 fields: 0 -> sentiment and 1 -> the tweet
# This method read the dataset  and split the labels and tweets.
# Returns x_text (tweets/text) and y ( labels.)
def load_data_labels(file_in):
    print("\tdata_helpers: loading dataset positive and negatives")
    x_text = list(io.open(file_in, 'r', buffering=200000, encoding='cp1252').readlines())
    x_text = [splitAndstrip(tweet, 1, True) for tweet in x_text]
    y = list(open(file_in).readlines())
    y = [eval(splitAndstrip(tweet, 0, False)) for tweet in y]
    return [x_text, y]


# Method that pads all sentences to the same length. The length is defined by the longest sentence.
# It uses a padding word "<PAD/>"
# Returns padded sentences.
def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    print("Max lenght of a sentence in number of words is: " + str(sequence_length))
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


# Method that pads all sentences to the same length given by "pad_to" input variable.
# It uses a padding word "<PAD/>"
# Returns padded sentences.
def pad_sentences_to(sentences, pad_to, padding_word="<PAD/>"):
    sequence_length = pad_to
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        # consider both increment and decrement
        num_padding = sequence_length - len(sentence)
        if num_padding > 0:
            new_sentence = sentence + [padding_word] * num_padding
        elif num_padding < 0:
            words_to_remov = num_padding*(-1)
            for i in range(words_to_remov):
                sentence.pop()
            new_sentence = sentence
        else:
            new_sentence = sentence
        padded_sentences.append(new_sentence)
    return padded_sentences


# Reads and load the vocabulary and its inverse mapping from the csv file.
# Returns a list with the vocabulary and the inverse mapping.
def build_vocab(vocIn, vocInvIn):
    voc = csv.reader(open(vocIn))
    voc_inv = csv.reader(open(vocInvIn))
    # Mapping from index to word
    vocabulary_inv = [x for x in voc_inv]
    # Mapping from word to index
    vocabulary = {x: i for x, i in voc}
    return [vocabulary, vocabulary_inv]


# Method that matches the words of the datasets to the indexes from the vocabulary
# returns x -> tweets represented as numbers that represent them the match to the vocabulary
# return  y -> labels
def build_input_data(tweets, labels, vocabulary):
    x = np.array([[int(vocabulary[word]) if word in vocabulary else 0 for word in tweet]
                  for tweet in tweets]).astype(dtype="int32")
    y = np.array(labels)
    return [x, y]


# Method used in the Cnn network.
# Given the dataset(already preprocessed, formatted into two fields separated by the "\t" delimiter), the vocabularies
# files and the number of word to pad. It returns x -> the texts represented as numbers that point to the vocabulary
# indexes, y -> the labels, and load into memory both vocabularies
def load_data_network(file_in, voc_in, voc_inv_in, words_to_pad):
    sentences, labels = load_data_labels(file_in)
    print("Padding strings... ")
    sentences_padded = pad_sentences_to(sentences, words_to_pad)
    print("Creating vocab files")
    vocabulary, vocabulary_inv = build_vocab(voc_in, voc_inv_in)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


# Load pre-trained embedding file into a dictionary key -> word, value -> vector
# return the dictionary
def load_pretrain_emb(emb_in):
    f = open(emb_in)
    embedding_vocab = {}
    for line in f:
        values = line.split(" ")
        word = values[0]
        emb_values = np.asarray(values[1:], dtype="float32")
        embedding_vocab[word] = emb_values
    f.close()
    return embedding_vocab




# method that makes a subset from the pre-trained embeddings. It contains only the words that are in the vocabulary
# dataset
# return the subset of the pre-trained embeddings
def match_emb_dataset(embedding_dict, vocabulary_inv, emb_size):
    x = np.array(
        [embedding_dict[join_array(word)] if join_array(word) in embedding_dict else zero_vector(emb_size) for word in
         vocabulary_inv]).astype(dtype="float32")
    return x


# method that prints how many words from the dataset vocabulary were found in the pre-trained embeddings
# creates a subset of the dataset
def vocab_to_emb(embDir, vocabInv, outFile):
    LOG_FILE_PATH = os.path.abspath(outFile)
    f = open(LOG_FILE_PATH, 'w')
    emb_dict = load_pretrain_emb(embDir)
    counter = 0
    for word in vocabInv:
        if join_array(word) in emb_dict:
            counter += 1
            word_tweet = join_array(word)
            line_word = word_tweet+' '+' '.join(map(str, emb_dict[word_tweet]))
            if line_word.rstrip() != "":
                f.write(''.join([ line_word, '\n']))
    f.close()
    print(counter)


# Load offline thesaurus into a dictionary
# returns the dictionary
def load_thesaurus(file_dict):
    thesaurus_file = open(file_dict)
    thesaurus_dict = {}
    for line in thesaurus_file:
        array_line = line.split(",")
        word = array_line[0]
        value = array_line[1]
        thesaurus_dict[word] = value
    return thesaurus_dict


# Method that deals with elongated words. It receives an elongated word and to replace the repeated word it checks
# against the offline thesaurus to replace using one or two words. E.g. cheeeeese -> cheese, Hiiiii -> hi
# this method is used when glove embeddings are used. It adds the <elong> string.
def replace_elongated_words_glove(tweet_in, thesaurus):
    array_tweet = tweet_in.split(" ")
    new_tweet = ""
    for word in array_tweet:
        has_repeated = re.search(r"([a-z])\1{2,}", word)
        if has_repeated is not None:
            if word in thesaurus:
                print("in thesaurus"+ word)
                new_tweet += word+" "
            else:
                word1 = re.sub(r"([a-z])\1{2,}", r'\1 <elong>', word)
                word2 = re.sub(r"([a-z])\1{2,}", r'\1\1 <elong>', word)

                if (word1 in thesaurus) and (word2 in thesaurus):
                    rankw1 = thesaurus[word1]
                    rankw2 = thesaurus[word2]
                    if rankw1 == rankw2:
                        new_tweet += word1+" "
                    elif rankw1 > rankw2:
                        new_tweet += word1+" "
                    else:
                        new_tweet += word2+" "
                elif word1 in thesaurus:
                    new_tweet += word1+" "
                elif word2 in thesaurus:
                    new_tweet += word2+" "
                else:
                    new_tweet += word2 +" "
        else:
            new_tweet += word + " "
    return new_tweet.strip()


# Method that deals with elongated words. It receives an elongated word and to replace the repeated word it checks
# against the offline thesaurus to replace using one or two words. E.g. cheeeeese -> cheese, Hiiiii -> hi
def replace_elongated_words(tweet_in, thesaurus):
    arrayTweet = tweet_in.split(" ")
    newTweet = ""

    for word in arrayTweet:
        hasrepeated = re.search(r"([a-z])\1{2,}", word)
        if hasrepeated is not None:
            if word in thesaurus:
                print("in thesaurus"+ word)
                newTweet += word+" "
            else:
                word1 = re.sub(r"([a-z])\1{2,}", r'\1', word)
                word2 = re.sub(r"([a-z])\1{2,}", r'\1\1', word)

                if (word1 in thesaurus) and (word2 in thesaurus):
                    rankw1 = thesaurus[word1]
                    rankw2 = thesaurus[word2]
                    if rankw1 == rankw2:
                        newTweet += word1+" "
                    elif rankw1 > rankw2:
                        newTweet += word1+" "
                    else:
                        newTweet += word2+" "
                elif word1 in thesaurus:
                    newTweet += word1+" "
                elif word2 in thesaurus:
                    newTweet += word2+" "
                else:
                    newTweet += word2 +" "
        else:
            newTweet += word + " "
    return newTweet.strip()


# Given a dimension returns a vector with random values
def random_vector(emb_dim):
    emb_rand_vect = np.random.random([emb_dim])
    return emb_rand_vect


# Given a dimension returns a vector with zero values
def zero_vector(emb_dim):
    embZeroVect = np.zeros([emb_dim])
    return embZeroVect


# Method call other methods that build a subset of the pre-trained embeddings. This subset contains only the words that
# are in the vocabulary
def getEmbDictAndEmbDataset(vocabulary_inv, emb_dataset, emb_size):
    embedding_dict = load_pretrain_emb(emb_dataset)
    emb_dataset = match_emb_dataset(embedding_dict, vocabulary_inv, emb_size)
    return emb_dataset


# Method that returns a string given an array
def join_array(array_in):
    array_str = "".join(array_in)
    return (array_str)


#
def create_emb_bigset(sentences, emb_dict, vocab_inv):
    x = np.array([[emb_dict[join_array(vocab_inv[int(word)])] if join_array(
        vocab_inv[int(word)]) in emb_dict else random_vector(100) for word in tweet] for tweet in sentences])
    return x


# This method generates the minibatches for the cnn. It takes the number of epochs and the batch size
# returns the minibatches
def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    print("len data")
    print(len(data))
    num_batches_per_epoch = int(len(data) / batch_size)
    if len(data) % batch_size != 0:
        num_batches_per_epoch += 1
    print("number batches per epoch")
    print(num_batches_per_epoch)
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# Methods for early stop
def calc_early_stop_pk(strip_len, looses_training, current_epoch):
    subset_strip = looses_training[((current_epoch - strip_len + 1) - 1):current_epoch]
    cpk_epoch = 1000 * ((sum(subset_strip) / (strip_len * min(subset_strip))) - 1)
    return cpk_epoch


# Methods for early stop
def calc_early_stop_gl(losses_validation, current_epoch):
    subset_losses_val = losses_validation[:current_epoch]
    error_opt_val = min(subset_losses_val)
    current_error_val = losses_validation[current_epoch - 1]
    glEpoch = 100 * ( (current_error_val/error_opt_val) - 1)
    return glEpoch


# Methods for early stop
def calc_earlystop_pq(losses_training, losses_validation, current_epoch, strip_len):
    glEpoch = calc_early_stop_gl(losses_validation, current_epoch)
    pkEpoch = calc_early_stop_pk(strip_len, losses_training, current_epoch)
    return glEpoch/pkEpoch


def calculate_filter_layer(initial_size, filters, strides_filters):
    for filt in filters.split(","):
        conv_output = conv_oper(initial_size, int(filt), 1)
        if filt in strides_filters:
            filter_maxpol = int(strides_filters[filt][0])
            stride_maxpol = int(strides_filters[filt][1])
            maxpol_out = conv_oper(conv_output, filter_maxpol, stride_maxpol)
            initial_size = maxpol_out
        else:
            last_filter_maxpol = conv_output
            return last_filter_maxpol


def conv_oper(initial_size, filter, stride):
    return ((initial_size - filter)/stride) + 1


############

def vocab_to_emb(embDir, vocabInv, outFile):
    LOG_FILE_PATH = os.path.abspath(outFile)
    f = open(LOG_FILE_PATH, 'w')
    emb_dict = load_pretrain_emb(embDir)
    counter = 0
    for word in vocabInv:
        if join_array(word) in emb_dict:
            counter += 1
            word_tweet = join_array(word)
            line_word = word_tweet+' '+' '.join(map(str, emb_dict[word_tweet]))
            if line_word.rstrip() != "":
                f.write(''.join([ line_word, '\n']))
    f.close()
    print(counter)



def vocabToEmb(embDir, vocabInv, outFile):
    LOG_FILE_PATH = os.path.abspath(outFile)
    f = open(LOG_FILE_PATH, 'w')
    emb_dict = loadPretrainedEmb(embDir)
    counter = 0
    for word in vocabInv:
        if joinArray(word) in emb_dict:
            counter += 1
            word_tweet = joinArray(word)
            line_word = word_tweet+' '+' '.join(map(str, emb_dict[word_tweet]))
            if line_word.rstrip() != "":
                f.write(''.join([ line_word, '\n']))
    f.close()
    print(counter)

#def load_pretrain_emb(emb_in):
def loadPretrainedEmb(embIn):
    f = open(embIn)
    embedding_vocab = {}
    for line in f:
        values = line.split(" ")
        word = values[0]
        emb_values = np.asarray(values[1:], dtype="float32")
        embedding_vocab[word] = emb_values
    f.close()
    return embedding_vocab







