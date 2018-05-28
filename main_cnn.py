#!usr/bin/env python
import tensorflow as tf
import os
import preproc, utils, vocab_builder


def main():
    # Preprocessing dataset(blankspaces, remove urls, authors, delete # char, replacing repeated chars, only numbers  )
    # in raw data set ->  <polarity>,<tweet id>,<date>,<query>,<user>,<tweet>
    # out pos (separator \t) : [1, 0]   <tweet>
    # out neg (separator \t) : [0, 1]   <tweet>
    url = "/home/alvaro/Desktop/NNets/thesis/cnn1_6M/smallDataSet/"
    url_thesaurus = "external/thesaurus.txt"
    pad_words = 42
    preproc.preproc_twitter_ultradense(url+'distantSupervisionSet.csv', url+'alvaro_test_oct.csv', url_thesaurus)
    vocab_builder.create_vocabularies(url+'alvaro_test_oct.csv', None, pad_words, url)
    #
    # print("creating sub set pretrained word embeddings")
    #
    # x, y, vocabulary, vocabulary_inv = utils.load_data_network(url + "train_set.csv", url + "vocab_100k.csv",
    #                                                            url + "vocab_inv_100k.csv")



    # """/home/alvaro/Desktop/NNets/thesis/cnn1_6M/smallDataSet/vocab_inv_alvaro_test_june_words.csv
    # /home/alvaro/Desktop/NNets/thesis/cnn1_6M/smallDataSet/vocab_alvaro_test_june_words.csv
    # /home/alvaro/Desktop/NNets/thesis/cnn1_6M/smallDataSet/alvaro_test_june.csv"""


if  __name__ =='__main__':main()

