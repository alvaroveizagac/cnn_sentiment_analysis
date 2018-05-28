#!usr/bin/env python
import itertools
from utils import *
from collections import Counter


# This method creates the vocabularies given the text. As an option, it can generate dictionaries containing
# the most "k" frequent terms
def build_vocabulary(tweets, num_vocab):
    word_counts = Counter(itertools.chain(*tweets))
    vocabulary_inv = [x[0] for x in word_counts.most_common(num_vocab)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


# method that calls other methods to create the vocabulary
# file in -> file that contains the documents and labels delimiter "\t" and two fields 0 -> sentiment,
#  1 -> document/tweet
# num_vocab -> To consider only a subset of the vocabulary, when "None" is given the it considers all words
# max_words_tweet -> the number to padd all the documents or tweets
def create_vocabularies(file_in, num_vocab, max_words_tweet, folder_out):
    file_name = file_in.split(".")[0]
    file_name = file_name.split("/")[-1]
    numb_voc = ""
    if num_vocab is not None:
        numb_voc = str(num_vocab)+"k_"
    print("Vocabulary loading data...")
    sentences, labels = load_data_labels(file_in)

    print("Vocab builder: padding...")
    sentences_padded = pad_sentences_to(sentences, max_words_tweet)
    print("Loading vocabularies...")
    vocabulary, vocabulary_inv = build_vocabulary(sentences_padded, num_vocab)

    print(len(vocabulary))
    print(len(vocabulary_inv))
    print("vocab builder: writing csv...")
    voc = csv.writer(open(folder_out+"vocab_"+(str(numb_voc))+file_name+"_"+str(max_words_tweet)+"_words.csv", "w"))
    voc_inv = csv.writer(open(folder_out+"vocab_inv_"+(str(numb_voc))+file_name+"_"+str(max_words_tweet)
                              +"_words.csv", "w"))
    for key, val in vocabulary.items():
        voc.writerow([key, val])
    for val in vocabulary_inv:
        voc_inv.writerow([val])


# url = "/home/alvaro/Desktop/NNets/Datasets/largeMovieReviewDSet/aclImdb/"


# create_vocabularies(url+"mreview_train_figure.csv", None, None, url)





