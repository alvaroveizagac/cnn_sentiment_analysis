#!/usr/bin/python
# -*- coding: latin-1 -*-
import io
import re
import os
import itertools
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from utils import load_thesaurus
from utils import replace_elongated_words
from utils import replace_elongated_words_glove


# Method that applies pre-processing steps for the movie reviews dataset
# cp1252, latin-1
# read a file that has the delimiter "\t", position 3 it is the label and position 2 it is the review
def read_dataset_rotten(file_in, file_out):
    outFile = open(file_out, 'a')
    with io.open(file_in, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                dataset_label = line.split('\t')[3]
                dataset_label = dataset_label.replace('"', '')
                dataset_label = dataset_label.strip()
                if (dataset_label == '0') or (dataset_label == '1'):
                    dataset_label = [1, 0]
                elif (dataset_label == '3') or (dataset_label == '4'):
                    dataset_label = [0, 1]
                tweet = line.split('\t')[2]
                tweet = tweet.replace('"', '')
                tweet = tweet.lower()
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                # delete urls
                tweet = re.sub(
                    r"(https?://(?:www.|(?!www))[^\s.]+.[^\s]{2,}|www.[^\s]+.[^\s]{2,})+",
                    " ", tweet)
                tweet = re.sub(r"(\B@\S+)", " ", tweet)  # delete the @uthors
                tweet = re.sub(r"(\B#)", " ", tweet)  # delete only the # character not the following word

                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # Second delete blank spaces
                tweet = re.sub(r"([a-z])\1{2,}", r'\1',
                               tweet)  # Replacing repeated characters \1\1 it is for looking 2 repeated characters
                tweet = re.sub(r"[\[\](){}<>:;,‒–—―\‐\-….!«»?¿“”\"/⁄]+", " ", tweet)  # ff
                tweet = re.sub(r"(\b[0-9]+\b)", "", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet = tweet.lstrip(' ')
                tweet = tweet.rstrip(' ')
                if (len(tweet.strip()) > 0):
                    if (str(dataset_label) != '2'):
                        tweet = tweet.encode('cp1252', errors='ignore')  # cp1252 utf-8
                        newTweet = str(dataset_label) + '\t' + tweet+'\n'
                        outFile.write(newTweet)
        except Exception as e:
            print(str(e))
    outFile.close()


# Method that reads twitter 1.6M dataset and applies preprocessing steps. The steps were performed to improve the match
# against the pretrained set embeddings ultradense.
# The dataset file has a delimiter ",".
# 0 -> contains the label, last field contains the tweet or document
def preproc_twitter_ultradense(file_in, file_out, file_dict):
    outFile = open(file_out, 'a')
    tokenizer_reg = RegexpTokenizer('\w+\'+\w+|\w+|[^\r\t\n ]+')
    thesaurus = load_thesaurus(file_dict)
    with io.open(file_in, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                # line = line.replace('"','')
                dataset_label = line.split('","')[0]
                dataset_label = dataset_label.replace('"', '')
                if dataset_label == '0':
                    dataset_label = [1, 0]
                elif dataset_label == '4':
                    dataset_label = [0, 1]
                tweet = line.split('","')[-1]
                tweet = tweet.lstrip('\"')
                tweet = tweet.rstrip('\n')
                tweet = tweet.rstrip('\"')
                tweet = tweet.lower()
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                # delete urls
                tweet = re.sub(
                    r"(https?:\/\/(?:www.|(?!www))[^\s.]+.[^\s]{2,}|www.[^\s]+.[^\s]{2,})+",
                    "<web>", tweet)
                tweet = re.sub(r"(\/)", " / ", tweet)
                tweet = re.sub(r"(\B@\S+)", "<user>", tweet)  # delete the @uthors
                tweet = re.sub(r"&(?:[A-Za-z]+[;])", " ", tweet) # delete html chars
                tweet = re.sub(r"[,](\w+)", r", \1", tweet)
                tweet = re.sub(r"[.](\w+)", r". \1", tweet)
                tweet = re.sub(r"[?](\w+)", r"? \1", tweet)
                tweet = re.sub(r"[!](\w+)", r"! \1", tweet)
                tweet = re.sub(r"[;](\w+)", r"; \1", tweet)
                tweet = re.sub(r"[:](\w+)", r": \1", tweet)
                tweet = re.sub(r"[(]", " ( ", tweet)
                tweet = re.sub(r"[)]", " ) ", tweet)
                tweet = re.sub(r"[\[]", " [ ", tweet)
                tweet = re.sub(r"[\]]", " ] ", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # Second delete blank spaces
                tweet = replace_elongated_words(tweet, thesaurus)
                tweet = re.sub(r"([!?.])\1{1,}", r'\1', tweet)
                tweet = re.sub(r"((\S+[^\x0a\x20-\x7f]+\S+)|([^\x0a\x20-\x7f]+\S+)|(\S+[^\x0a\x20-\x7f]+\b)|([^\x0a\x20-\x7f])+)", " ", tweet)
                tweet = re.sub(r"(\b[0-9]+\b)", "", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet_list = tokenizer_reg.tokenize(tweet)
                tweet = " ".join(tweet_list)
                tweet = tweet.lstrip(' ')
                tweet = tweet.rstrip(' ')
                if len(tweet.strip()) > 0:
                    if str(dataset_label) != '2':
                        tweet = tweet.encode('utf-8', errors='ignore')  # cp1252 utf-8
                        new_tweet = str(dataset_label) + '\t' + tweet+'\n'
                        outFile.write(new_tweet)
        except Exception as e:
            print(str(e))
    outFile.close()


def preproc_twitter_ultradense_image(file_in, file_out):
    outFile = io.open(file_out, 'a', buffering=200000, encoding='cp1252')

    with io.open(file_in, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                # line = line.replace('"','')
                dataset_label = line.split("\t")[0]
                dataset_label = dataset_label.replace('"', '')
                if dataset_label == 'positive':
                    dataset_label = [1, 0]
                elif dataset_label == 'negative':
                    dataset_label = [0, 1]
                tweet = line.split("\t")[-1]
                tweet = tweet.lstrip('\"')
                tweet = tweet.rstrip('\n')
                tweet = tweet.rstrip('\"')
                if len(tweet.strip()) > 0:
                    if str(dataset_label) != '2':
                        #tweet = tweet.encode('utf-8', errors='ignore')  # cp1252 utf-8
                        new_tweet = str(dataset_label) + '\t' + tweet+'\n'
                        outFile.write(new_tweet)
        except Exception as e:
            print(str(e))
    outFile.close()

# url_image = "/home/alvaro/Desktop/NNets/Datasets/largeMovieReviewDSet/aclImdb/"
# preproc_twitter_ultradense_image(url_image+"mreview_train.csv", url_image+"mreview_train_figure.csv")

# Method that reads twitter augmented dataset and applies pre-processing steps. The steps were performed to improve
# the match against the pre-trained set embeddings.
# 627152870375313408	negative	Courtney Love doesn't want photos that could
def preproc_twitter_aug_ultradense(fileIn, fileOut, file_dict):
    outFile = open(fileOut, 'a')
    tokenizer_reg = RegexpTokenizer('\w+\'+\w+|\w+|[^\r\t\n ]+')
    thesaurus = load_thesaurus(file_dict)
    with io.open(fileIn, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                dataset_id = line.split('\t')[0]
                dataset_label = line.split('\t')[1]
                dataset_label = dataset_label.replace('"', '')
                #if dataset_label == '0':
                #    dataset_label = [1, 0]
                #elif dataset_label == '4':
                #    dataset_label = [0, 1]
                tweet = line.split('","')[-1]  # the last element of the tweet
                tweet = tweet.lstrip('\"')
                tweet = tweet.rstrip('\n')
                tweet = tweet.rstrip('\"')
                tweet = tweet.lower()
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet = re.sub(
                    r"(https?:\/\/(?:www.|(?!www))[^\s.]+.[^\s]{2,}|www.[^\s]+.[^\s]{2,})+",
                    "<web>", tweet)  # delete urls
                tweet = re.sub(r"(\/)", " / ", tweet)
                tweet = re.sub(r"(\B@\S+)", "<user>", tweet)  # delete the @uthors
                tweet = re.sub(r"&(?:[A-Za-z]+[;])", " ", tweet) # delete html chars
                tweet = re.sub(r"[,]", " ", tweet)
                tweet = re.sub(r"[.](\w+)", r". \1", tweet)
                tweet = re.sub(r"[?](\w+)", r"? \1", tweet)
                tweet = re.sub(r"[!](\w+)", r"! \1", tweet)
                tweet = re.sub(r"[;](\w+)", r"; \1", tweet)
                tweet = re.sub(r"[:](\w+)", r": \1", tweet)
                tweet = re.sub(r"[(]", " ( ", tweet)
                tweet = re.sub(r"[)]", " ) ", tweet)
                tweet = re.sub(r"[\[]", " [ ", tweet)
                tweet = re.sub(r"[\]]", " ] ", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # Second delete blank spaces
                tweet = replace_elongated_words(tweet, thesaurus)
                tweet = re.sub(r"([!?.])\1{1,}", r'\1', tweet)
                tweet = re.sub(r"((\S+[^\x0a\x20-\x7f]+\S+)|([^\x0a\x20-\x7f]+\S+)|(\S+[^\x0a\x20-\x7f]+\b)|([^\x0a\x20-\x7f])+)", " ", tweet)
                tweet = re.sub(r"(\b[0-9]+\b)", "", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet_list = tokenizer_reg.tokenize(tweet)
                tweet = " ".join(tweet_list)
                tweet = tweet.lstrip(' ')
                tweet = tweet.rstrip(' ')
                if len(tweet.strip()) > 0:
                    if str(dataset_label) != '2':
                        tweet = tweet.encode('utf-8', errors='ignore')  # cp1252 utf-8
                        newTweet = dataset_id+','+ dataset_label + ',' + tweet+'\n'
                        outFile.write(newTweet)
        except Exception as e:
            print(str(e))
    outFile.close()

# url_train = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/semeval_train.csv"
# url_train_aug = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/semeval_train_for_aug.csv"

# url_test = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/semeval_test.csv"
# url_test_aug = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/semeval_test_for_aug.csv"
# file_dict = "/media/alvaro/TI106320W0C1/ResearchProjectSource/twitterSpark/ExternalFiless/thesaurus.txt"
# preproc_twitter_aug_ultradense(url_train, url_train_aug, file_dict)
# preproc_twitter_aug_ultradense(url_test, url_test_aug, file_dict)
# preproc_twitter_aug_ultradense(fileIn, fileOut)

# Method that reads twitter 1.6M dataset and applies preprocessing steps. The steps were performed to improve the match
# against the pretrained set embeddings ultradense.
# The dataset file has a delimiter ",".
# 0 -> contains the label, last field contains the tweet or document
def preproc_twitter_glove(fileIn, fileOut,file_dict):
    outFile = open(fileOut, 'a')
    tokenizer_reg = RegexpTokenizer('\w+\'+\w+|\w+|[^\r\t\n ]+')
    thesaurus = load_thesaurus(file_dict)
    with io.open(fileIn, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                dataset_label = line.split('","')[0]
                dataset_label = dataset_label.replace('"', '')
                if dataset_label == '0':
                    dataset_label = [1, 0]
                elif dataset_label == '4':
                    dataset_label = [0, 1]
                tweet = line.split('","')[-1]  # the last element of the tweet
                tweet = tweet.lstrip('\"')
                tweet = tweet.rstrip('\n')
                tweet = tweet.rstrip('\"')
                tweet = tweet.lower()
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet = re.sub(
                    r"(https?:\/\/(?:www.|(?!www))[^\s.]+.[^\s]{2,}|www.[^\s]+.[^\s]{2,})+",
                    "<url>", tweet)
                tweet = re.sub(r"(\/)", " / ", tweet)
                tweet = re.sub(r"(\B@\S+)", "<user>", tweet)  # delete the @uthors
                tweet = re.sub(r"(\B#)", "# <hashtag> ", tweet)
                tweet = re.sub(r"&(?:[A-Za-z]+[;])", " ", tweet) # delete html chars
                tweet = re.sub(r"[,](\w+)", r", \1", tweet)
                tweet = re.sub(r"[.](\w+)", r". \1", tweet)
                tweet = re.sub(r"[?](\w+)", r"? \1", tweet)
                tweet = re.sub(r"[!](\w+)", r"! \1", tweet)
                tweet = re.sub(r"[;](\w+)", r"; \1", tweet)
                tweet = re.sub(r"[:](\w+)", r": \1", tweet)
                tweet = re.sub(r"[(]", " ( ", tweet)
                tweet = re.sub(r"[)]", " ) ", tweet)
                tweet = re.sub(r"[\[]", " [ ", tweet)
                tweet = re.sub(r"[\]]", " ] ", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # Second delete blank spaces
                tweet = replace_elongated_words_glove(tweet, thesaurus)
                tweet = re.sub(r"([!?.])\1{1,}", r'\1 <repeat>', tweet)
                tweet = re.sub(r"((\S+[^\x0a\x20-\x7f]+\S+)|([^\x0a\x20-\x7f]+\S+)|(\S+[^\x0a\x20-\x7f]+\b)|([^\x0a\x20-\x7f])+)", " ", tweet)
                tweet = re.sub(r"(\b[0-9]+\b)", "<number> ", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet_list = tokenizer_reg.tokenize(tweet)
                tweet = " ".join(tweet_list)
                tweet = tweet.lstrip(' ')
                tweet = tweet.rstrip(' ')
                if len(tweet.strip()) > 0:
                    if str(dataset_label) != '2':
                        tweet = tweet.encode('utf-8', errors='ignore')  # cp1252 utf-8
                        newTweet = str(dataset_label) + '\t' + tweet+'\n'
                        outFile.write(newTweet)
        except Exception as e:
            print(str(e))
    outFile.close()


# Method that reads twitter SemEval dataset and applies preprocessing steps.
# The dataset file has a delimiter "\t".
# 1 -> contains the label, last field contains the tweet or document
def preproc_semeval_twitter(fileIn, fileOut,file_dict):
    outFile = open(fileOut, 'a')
    tokenizer_reg = RegexpTokenizer('\w+\'+\w+|\w+|[^\r\t\n ]+')
    thesaurus = load_thesaurus(file_dict)
    with io.open(fileIn, 'r', buffering=200000) as f:
        try:
            for line in f:
                dataset_label = line.split('\t')[1]
                dataset_label = dataset_label.replace('"', '')
                if dataset_label == 'negative':
                    dataset_label = [1, 0]
                elif dataset_label == 'positive':
                    dataset_label = [0, 1]
                tweet = line.split('\t')[-1]  # the last element of the tweet
                tweet = tweet.lstrip('\"')
                tweet = tweet.rstrip('\n')
                tweet = tweet.rstrip('\"')
                tweet = tweet.lower()
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet = re.sub(
                    r"(https?:\/\/(?:www.|(?!www))[^\s.]+.[^\s]{2,}|www.[^\s]+.[^\s]{2,})+",
                    "<web>", tweet)  # delete urls
                tweet = re.sub(r"(\/)", " / ", tweet)
                tweet = re.sub(r"(\B@\S+)", "<user>", tweet)  # delete the @uthors
                tweet = re.sub(r"&(?:[A-Za-z]+[;])", " ", tweet) # delete html chars
                tweet = re.sub(r"[,](\w+)", r", \1", tweet)
                tweet = re.sub(r"[.](\w+)", r". \1", tweet)
                tweet = re.sub(r"[?](\w+)", r"? \1", tweet)
                tweet = re.sub(r"[!](\w+)", r"! \1", tweet)
                tweet = re.sub(r"[;](\w+)", r"; \1", tweet)
                tweet = re.sub(r"[:](\w+)", r": \1", tweet)
                tweet = re.sub(r"[(]", " ( ", tweet)
                tweet = re.sub(r"[)]", " ) ", tweet)
                tweet = re.sub(r"[\[]", " [ ", tweet)
                tweet = re.sub(r"[\]]", " ] ", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # Second delete blank spaces
                tweet = replace_elongated_words(tweet, thesaurus)
                tweet = re.sub(r"([!?.])\1{1,}", r'\1', tweet)
                tweet = re.sub(r"((\S+[^\x0a\x20-\x7f]+\S+)|([^\x0a\x20-\x7f]+\S+)|(\S+[^\x0a\x20-\x7f]+\b)|([^\x0a\x20-\x7f])+)", " ", tweet)
                tweet = re.sub(r"(\b[0-9]+\b)", "", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet_list = tokenizer_reg.tokenize(tweet)
                tweet = " ".join(tweet_list)
                tweet = tweet.lstrip(' ')
                tweet = tweet.rstrip(' ')
                if len(tweet.strip()) > 0:
                    if str(dataset_label) != '2':
                        tweet = tweet.encode('cp1252', errors='ignore')  # cp1252 utf-8 , encoding='cp1252'
                        newTweet = str(dataset_label) + '\t' + tweet+'\n'
                        outFile.write(newTweet)
        except Exception as e:
            print(str(e))
    outFile.close()

# url_training = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/oversample2017_test_train/"
# url_testing = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/oversample2017_test_train/"
# url_dict = "/media/alvaro/TI106320W0C1/ResearchProjectSource/twitterSpark/ExternalFiless/thesaurus.txt"
# preproc_semeval_twitter(url_training+"semeval_training.csv", url_training+"semeval_over_preproc_training.csv", url_dict)
# preproc_semeval_twitter(url_testing+"semeval_testing.csv", url_testing+"semeval_over_prepr_testing.csv",url_dict)


# Method that reads the movie reviews dataset and applies preprocessing steps.
# The dataset file has a delimiter "\t".
# 0 -> contains the label, last field contains the tweet or document
def preproc_movie_reviews(fileIn, fileOut,file_dict):
    outFile = open(fileOut, 'a')
    tokenizer_reg = RegexpTokenizer('\w+\'+\w+|\w+|[^\r\t\n ]+')
    thesaurus = load_thesaurus(file_dict)
    with io.open(fileIn, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                datasetLabel = line.split('\t')[0]
                datasetLabel = datasetLabel.replace('"', '')
                if datasetLabel == 'negative':
                    datasetLabel = [1, 0]
                elif datasetLabel == 'positive':
                    datasetLabel = [0, 1]
                tweet = line.split('\t')[-1]  # the last element of the tweet
                tweet = tweet.lstrip('\"')
                tweet = tweet.rstrip('\n')
                tweet = tweet.rstrip('\"')
                tweet = tweet.lower()
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet = re.sub(
                    r"(https?:\/\/(?:www.|(?!www))[^\s.]+.[^\s]{2,}|www.[^\s]+.[^\s]{2,})+",
                    "<web>", tweet) # delete urls
                tweet = re.sub(r"(\/)", " / ", tweet)
                tweet = re.sub(r"(\B@\S+)", "<user>", tweet)  # delete the @uthors
                tweet = re.sub(r"&(?:[A-Za-z]+[;])", " ", tweet) # delete html chars
                tweet = re.sub(r"[\"]", "", tweet)
                tweet = re.sub(r"[,](\w+)", r", \1", tweet)
                tweet = re.sub(r"[.](\w+)", r". \1", tweet)
                tweet = re.sub(r"[?](\w+)", r"? \1", tweet)
                tweet = re.sub(r"[!](\w+)", r"! \1", tweet)
                tweet = re.sub(r"[;](\w+)", r"; \1", tweet)
                tweet = re.sub(r"[:](\w+)", r": \1", tweet)
                tweet = re.sub(r"[(]", " ( ", tweet)
                tweet = re.sub(r"[)]", " ) ", tweet)
                tweet = re.sub(r"[\[]", " [ ", tweet)
                tweet = re.sub(r"[\]]", " ] ", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # Second delete blank spaces
                tweet = replace_elongated_words(tweet, thesaurus)
                tweet = re.sub(r"([!?.])\1{1,}", r'\1', tweet)
                tweet = re.sub(r"((\S+[^\x0a\x20-\x7f]+\S+)|([^\x0a\x20-\x7f]+\S+)|(\S+[^\x0a\x20-\x7f]+\b)|([^\x0a\x20-\x7f])+)", " ", tweet)
                tweet = re.sub(r"(\b[0-9]+\b)", "", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet_list = tokenizer_reg.tokenize(tweet)
                tweet = " ".join(tweet_list)
                tweet = tweet.lstrip(' ')
                tweet = tweet.rstrip(' ')
                if len(tweet.strip()) > 0:
                    if str(datasetLabel) != '2':
                        tweet = tweet.encode('cp1252', errors='ignore')  # cp1252 utf-8 , encoding='cp1252'
                        newTweet = str(datasetLabel) + '\t' + tweet+'\n'
                        outFile.write(newTweet)
        except Exception as e:
            print(str(e))
    outFile.close()


# Method that reads twitter augmented dataset and applies pre-processing steps. The steps were performed to improve
# the match against the pre-trained set embeddings.
def preproc_augmented_post(fileIn, fileOut, file_dict):
    outFile = open(fileOut, 'a')
    tokenizer_reg = RegexpTokenizer('\w+\'+\w+|\w+|[^\r\t\n ]+')
    thesaurus = load_thesaurus(file_dict)
    with io.open(fileIn, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                datasetLabel = line.split('\t')[0]
                datasetLabel = datasetLabel.replace('"', '')
                tweet = line.split('\t')[-1]  # the last element of the tweet
                tweet = tweet.lstrip('\"')
                tweet = tweet.rstrip('\n')
                tweet = tweet.rstrip('\"')
                tweet = tweet.lower()
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet = re.sub(
                    r"(https?:\/\/(?:www.|(?!www))[^\s.]+.[^\s]{2,}|www.[^\s]+.[^\s]{2,})+",
                    "<web>", tweet)
                tweet = re.sub(r"(\/)", " / ", tweet)
                tweet = re.sub(r"(\B@\S+)", "<user>", tweet)  # delete the @uthors
                tweet = re.sub(r"&(?:[A-Za-z]+[;])", " ", tweet) # delete html chars
                tweet = re.sub(r"[,](\w+)", r", \1", tweet)
                tweet = re.sub(r"[.](\w+)", r". \1", tweet)
                tweet = re.sub(r"[?](\w+)", r"? \1", tweet)
                tweet = re.sub(r"[!](\w+)", r"! \1", tweet)
                tweet = re.sub(r"[;](\w+)", r"; \1", tweet)
                tweet = re.sub(r"[:](\w+)", r": \1", tweet)
                tweet = re.sub(r"[(]", " ( ", tweet)
                tweet = re.sub(r"[)]", " ) ", tweet)
                tweet = re.sub(r"[\[]", " [ ", tweet)
                tweet = re.sub(r"[\]]", " ] ", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # Second delete blank spaces
                tweet = replace_elongated_words(tweet, thesaurus)
                tweet = re.sub(r"([!?.])\1{1,}", r'\1', tweet)
                tweet = re.sub(r"((\S+[^\x0a\x20-\x7f]+\S+)|([^\x0a\x20-\x7f]+\S+)|(\S+[^\x0a\x20-\x7f]+\b)|([^\x0a\x20-\x7f])+)", " ", tweet)
                tweet = re.sub(r"(\b[0-9]+\b)", "", tweet)
                tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                tweet_list = tokenizer_reg.tokenize(tweet)
                tweet = " ".join(tweet_list)
                tweet = tweet.lstrip(' ')
                tweet = tweet.rstrip(' ')
                if (len(tweet.strip()) > 0):
                    if (str(datasetLabel) != '2'):
                        tweet = tweet.encode('utf-8', errors='ignore')  # cp1252 utf-8
                        newTweet = str(datasetLabel) + '\t' + tweet+'\n'
                        outFile.write(newTweet)
        except Exception as e:
            print(str(e))
    outFile.close()


# url_file_in = ""
# url_file_out = ""
# url_file_dict = ""
# preproc_augmented_post(url_file_in , url_file_out, url_file_dict)

# Method that splits augmented dataset into no agumented, positive and negative files
def split_data_aug_sets(file_in, file_no_aug, file_pos, file_neg):
    out_positive = io.open(file_pos, 'a', buffering=200000, encoding='cp1252')
    out_negative = io.open(file_neg, 'a', buffering=200000, encoding='cp1252')
    out_no_aug = io.open(file_no_aug, 'a', buffering=200000, encoding='cp1252')
    counter = 0
    with io.open(file_in, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                counter = counter + 1
                line_dataset = line.strip("\n")
                datasetLabel = line_dataset.split('\t')[0]
                tweet = line_dataset.split('\t')[-1]  # the last element of the tweet
                line_post = str(datasetLabel)+"\t"+tweet+"\n"
                if (counter <= 1599978 ):
                    out_no_aug.write(line_post)
                elif counter >= 1599979 and counter <= 2372160:
                    out_negative.write(line_post)
                elif counter >= 2372161:
                    out_positive.write(line_post)
        except Exception as e:
            print(str(e))
    out_no_aug.close()
    out_negative.close()
    out_positive.close()
    print("end split ")


# Method that splits dataset into positive and negative files
def split_data_pos_neg(file_in, file_pos, file_neg):
    out_positive = io.open(file_pos, 'a', buffering=200000, encoding='cp1252')
    out_negative = io.open(file_neg, 'a', buffering=200000, encoding='cp1252')
    counter = 0
    with io.open(file_in, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                counter = counter + 1
                line_dataset = line.strip("\n")
                datasetLabel = line_dataset.split('\t')[0]
                if datasetLabel == "[0, 1]":
                    out_positive.write(line_dataset+"\n")
                elif datasetLabel== "[1, 0]":
                    out_negative.write(line_dataset+"\n")
        except Exception as e:
            print(str(e))
    #out_no_aug.close()
    out_negative.close()
    out_positive.close()
    print("end split ")


# url_file_in = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/Preprocess/twitter_generic/semeval_only_aug_twitter_generic.csv"
# url_file_pos = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/Preprocess/twitter_generic/twitter_gen_positive.csv"
# url_file_neg = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/Preprocess/twitter_generic/twitter_gen_negative.csv"
# split_data_pos_neg(url_file_in, url_file_pos, url_file_neg)


# Method that returns in a file random instances. The variable numInst denotes how many random will be given
# 640198209281880064	positive	barcastuff: Messi is
def get_random_inst_semeval(fileIn, fileOut, numInst):
    out = open(fileOut, 'a')
    counter = 10000
    negatives = []
    with io.open(fileIn, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                # counter = counter + 1
                id_dataset = line.split("\t")[0]
                datasetLabel = line.split("\t")[1]
                datasetLabel = datasetLabel.replace('"', '')
                #if datasetLabel == '0':
                # id = counter
                # datasetLabel = "negative"
                tweet = line.split("\t")[-1]  # the last element of the tweet
                tweet = tweet.replace('"', '')
                tweet = tweet.strip("\n")
                line_dict = id_dataset+"\t"+datasetLabel+"\t"+tweet+"\n"
                negatives.append(line_dict)
            print("end dict creation")
        except Exception as e:
            print(str(e))
    print("len dict: "+ str(len(negatives)))
    np.random.seed(123)
    shuffle_index = np.random.permutation(np.arange(len(negatives)))
    for i in range(numInst):
        index_random = shuffle_index[i]
        tweet = negatives[index_random].encode('utf-8', errors='ignore')
        out.write(tweet)
    out.close()

# fileIn = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/semeval_positive.csv"
# fileOut = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/semeval_positive_7323.csv"
# numInst = 7323
# get_random_inst_semeval(fileIn, fileOut, numInst)

# Method that returns augmented instances according to the variable percentage
def get_random_inst_augmented(fileIn, fileOut, numInst, percentage):
    num_instances = (numInst * percentage) / 100
    out = io.open(fileOut, 'a', buffering=200000, encoding='cp1252')
    counter = 10000
    instances = []
    with io.open(fileIn, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                counter = counter + 1
                datasetLabel = line.split('\t')[0]
                datasetLabel = datasetLabel.replace('"', '')
                id = counter
                tweet = line.split('\t')[-1]  # the last element of the tweet
                tweet = tweet.replace('"', '')
                tweet = tweet.strip("\n")
                line_dict = datasetLabel+"\t"+tweet+"\n"
                instances.append(line_dict)
            print("end dict creation")
        except Exception as e:
            print(str(e))
    print("len dict: "+ str(len(instances)))
    np.random.seed(123)
    shuffle_index = np.random.permutation(np.arange(len(instances)))
    for i in range(num_instances):
        index_random = shuffle_index[i]
        tweet = instances[index_random]
        out.write(tweet)
    out.close()

# Method that merges 2 files and returns one.
def mergeFiles(file_in, file_neg, file_pos, file_out):
    out = io.open(file_out, 'a', buffering=200000, encoding='cp1252')
    counter = 0
    instances = []
    with io.open(file_in, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                if(len(line.strip()) > 0 ):
                    counter = counter + 1
                    out.write(line)
        except Exception as e:
            print(str(e))
    with io.open(file_neg, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                if (len(line.strip()) > 0):
                    counter = counter + 1
                    out.write(line)
        except Exception as e:
            print(str(e))
    with io.open(file_pos, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                if (len(line.strip()) > 0):
                    counter = counter + 1
                    out.write(line)
        except Exception as e:
            print(str(e))
    out.close()
    print("len dict: " + str(counter))


#url = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/Preprocess/twitter_generic/"
#mergeFiles(url+"semeval_train_prepr_noaug.csv", url+"positive_10.csv", url+"negative_10.csv", url+"semeval_train_prepr_aug_10.csv")
#mergeFiles(url+"semeval_train_prepr_noaug.csv", url+"positive_20.csv", url+"negative_20.csv", url+"semeval_train_prepr_aug_20.csv")
#mergeFiles(url+"semeval_train_prepr_noaug.csv", url+"positive_30.csv", url+"negative_30.csv", url+"semeval_train_prepr_aug_30.csv")
#mergeFiles(url+"semeval_train_prepr_noaug.csv", url+"positive_40.csv", url+"negative_40.csv", url+"semeval_train_prepr_aug_40.csv")
#mergeFiles(url+"semeval_train_prepr_noaug.csv", url+"positive_50.csv", url+"negative_50.csv", url+"semeval_train_prepr_aug_50.csv")
#mergeFiles(url+"semeval_train_prepr_noaug.csv", url+"positive_60.csv", url+"negative_60.csv", url+"semeval_train_prepr_aug_60.csv")
#mergeFiles(url+"semeval_train_prepr_noaug.csv", url+"positive_70.csv", url+"negative_70.csv", url+"semeval_train_prepr_aug_70.csv")
#mergeFiles(url+"semeval_train_prepr_noaug.csv", url+"positive_80.csv", url+"negative_80.csv", url+"semeval_train_prepr_aug_80.csv")
#mergeFiles(url+"semeval_train_prepr_noaug.csv", url+"positive_90.csv", url+"negative_90.csv", url+"semeval_train_prepr_aug_90.csv")


# Method that prints the word frequency per tweet
def word_freq_tweet(fileIn):
    with io.open(fileIn, 'r', buffering=200000, encoding='cp1252') as f:
        num_words = []
        try:
            for line in f:

                tweet = line.split("\t")[1]
                array_word = tweet.split(" ")
                num_words.append(len(array_word))

        except Exception as e:
            print(str(e))
    total_number = 0
    for i in num_words:
        total_number += i
    print(total_number)
    # print(num_words)
    words_count = Counter(itertools.chain(num_words))
    print(words_count)


# Method that read and change the format of the augmented dataset
def read_aug_no_preproc(fileIn, fileOut):
    out_file = io.open(fileOut, 'a', buffering=200000, encoding='cp1252')
    with io.open(fileIn, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                if (len(line.strip())) > 0:
                    num_comp = len(line.split(','))
                    dataset_label = ""
                    tweet = ""
                    if num_comp == 3:
                        dataset_label = line.split(',')[1]
                        tweet = line.split(',')[2]
                    elif num_comp == 2:
                        dataset_label = line.split(',')[0]
                        tweet = line.split(',')[1]
                    dataset_label = dataset_label.strip()
                    # negative
                    if dataset_label == 'negative':
                        dataset_label = [1, 0]
                    # positive
                    elif dataset_label == 'positive':
                        dataset_label = [0, 1]
                    tweet = tweet.lower()
                    tweet = re.sub(r"[\x20]{2,}", " ", tweet)  # delete blank spaces
                    tweet = tweet.lstrip(' ')
                    tweet = tweet.rstrip(' ')
                    if (len(tweet.strip()) > 0):
                        new_tweet = str(dataset_label) + '\t' + tweet
                        out_file.write(new_tweet)
        except Exception as e:
            print("Error"+str(e))
    out_file.close()

"""
url_ant = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_blan = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_goo_gen = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_goo_sent = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_twit_gen = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_twit_sen = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
read_aug_no_preproc(url_ant+"semeval_antonymous.csv", url_ant+"semeval_antonymous_prepr.csv")
read_aug_no_preproc(url_blan+"semeval_blankout.csv", url_blan+"semeval_blankout_prepr.csv")
read_aug_no_preproc(url_goo_gen +"semeval_google_generic.csv", url_goo_gen+ "semeval_google_generic_prepr.csv")
read_aug_no_preproc(url_goo_sent +"semeval_google_sent.csv", url_goo_sent+"semeval_google_sent_prepr.csv")
read_aug_no_preproc(url_twit_gen+"semeval_twitter_gener.csv", url_twit_gen+"semeval_twitter_gener_prepr.csv")
read_aug_no_preproc(url_twit_sen+"semeval_twitter_sent.csv", url_twit_sen+"semeval_twitter_sent_prepr.csv")"""
#
# url_test = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/"
# read_aug_no_preproc(url_test+"semeval_train_for_aug.csv", url_test+"semeval_train_for_aug_prepr.csv")

# Method that counts the positive and negative tweets
def count_pos_neg_dataset(file_in):
    pos_tweets = 0
    neg_tweets = 0
    neu_tweets = 0
    with io.open(file_in, 'r', buffering=200000, encoding='cp1252') as f:
        try:
            for line in f:
                datasetLabel = str(line.split('\t')[1])
                # print(dataset_label)
                # dataset_label = str(line.split(',')[1])
                # semeval
                if datasetLabel == 'negative':
                    neg_tweets += 1
                elif datasetLabel == 'positive':
                    pos_tweets += 1
                elif datasetLabel == 'neutral':
                    neu_tweets += 1
                # 1.6 M Twitter dataset
                #if datasetLabel == '[1, 0]':
                #    neg_tweets += 1
                #elif datasetLabel == '[0, 1]':
                #    pos_tweets += 1
                #else:
                #   print("error "+ datasetLabel)
        except Exception as e:
            print(str(e))
    print("Positives" + str(pos_tweets))
    print("Negatives" + str(neg_tweets))
    print("NEutral"+str(neu_tweets))
#print("dev")
#count_pos_neg_dataset("/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_task_4A/dev_uniques.csv")
#print("test")
#count_pos_neg_dataset("/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_task_4A/test_uniques.csv")
#print("train")
#count_pos_neg_dataset("/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_task_4A/train_uniques.csv")

"""
url_ant = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_blan = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_goo_gen = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_goo_sent = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_twit_gen = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
url_twit_sen = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/Augmented/"
print("semeval_antonymous_prepr")
count_pos_neg_dataset(url_ant +"semeval_antonymous_prepr.csv")
print("semeval_blankout_prepr")
count_pos_neg_dataset(url_blan + "semeval_blankout_prepr.csv")semeval_twitter_sent_prepr
print("semeval_google_generic_prepr")
count_pos_neg_dataset(url_goo_gen + "semeval_google_generic_prepr.csv")
print("semeval_google_sent_prepr")
count_pos_neg_dataset(url_goo_sent + "semeval_google_sent_prepr.csv")
print("semeval_twitter_gener_prepr")
count_pos_neg_dataset(url_twit_gen + "semeval_twitter_gener_prepr.csv")
print("semeval_twitter_sent_prepr")
count_pos_neg_dataset(url_twit_sen + "semeval_twitter_sent_prepr.csv")"""




# print("test")
# count_pos_neg_dataset("/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/semeval_test_for_aug.csv")
# print("train")
# count_pos_neg_dataset("/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_uniq_2class/augmentedBill/semeval_train_for_aug.csv")
# print("dev")
# url_dev = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_gold_set/dev_gold_uniques.csv"
# count_pos_neg_dataset(url_dev)
# print("dev")
# url_test = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_gold_set/test_gold_uniques.csv"
# count_pos_neg_dataset(url_test)
# print("train")
# url_train = "/home/alvaro/Desktop/NNets/Datasets/Semeval2017/semeval_gold_set/train_gold_uniques.csv"
# count_pos_neg_dataset(url_train)


# Method that merges semeval datasets
def merge_semeval_datasets(folderIn, fileOut):
    out = open(fileOut,'a')
    document_dict = {}
    for document in os.listdir(folderIn):
        print("len dict")
        print(len(document_dict))
        url_doc = os.path.abspath(os.path.join(folderIn, document))
        print(url_doc)
        counter = 0
        with io.open(url_doc, 'r', buffering=200000, encoding='cp1252') as f:
            try:
                for line in f:
                    parts = line.strip("\n")
                    parts = parts.split("\t")
                    id = parts[0]
                    sentiment = parts[1]
                    tweet = parts[2]
                    if id not in document_dict:
                        document_dict[id] = tweet
                        if(tweet != "Not Available") and (sentiment!= "neutral"):
                            tweet = id+"\t"+sentiment+"\t"+tweet+"\n"
                            counter += 1
                            out.write(tweet)
                print("salio del "+str(counter))
            except Exception as e:
                print(str(e))

    out.close()


# Method that merges the movie review datasets
def merge_movie_review_dataset(folderIn, label, fileOut):
    out = io.open(fileOut,'a', buffering=200000, encoding='cp1252')
    counter = 0
    for document in os.listdir(folderIn):
        url_doc = os.path.abspath(os.path.join(folderIn, document))
        print(url_doc)

        with io.open(url_doc, 'r', buffering=200000, encoding='cp1252') as f:
            try:
                for line in f:
                    review = line.strip("\n")
                    if review != "":
                        review_out = review + "\n"
                        counter += 1
                        out.write(review_out)
            except Exception as e:
                print(str(e))
    print("salio del "+str(counter))
    out.close()



# Method that counts the unique tweets
def countUniquesTweets(fileIn):
    with io.open(fileIn, 'r', buffering=200000, encoding='cp1252') as f:
        counter = 0
        for line in f:
            counter += 1
        print(counter)


