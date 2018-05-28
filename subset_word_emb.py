#!usr/bin/env python
import utils

url = "/home/alvaro/Desktop/NNets/thesis/cnn1_6M/smallDataSet/"
x, y, vocabulary, vocabulary_inv = utils.load_data_network(url + "train_set.csv", url + "vocab_100k.csv",
                                                               url + "vocab_inv_100k.csv")

word2vec = url + "twitter_lower_cw1_sg400_transformed.txt"
outFile = url + "twitter_lower_cw1_sg400_transformed_subset.txt"
print("creating subset - start")
utils.vocab_to_emb(word2vec, vocabulary_inv, outFile)
print("creating subset - finish")