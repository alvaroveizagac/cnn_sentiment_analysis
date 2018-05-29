# Convolutional Neural Networks applied to Sentiment Analysis
Convolutional Neural Networks applied to Sentiment Analysis
The project applies sentiment analysis using convolutional neural networks.

The dataset should follow the following format using as a separator "," (Sentiment 140):
SENTIMENT_OF_TEXT (positive=4/negative=0/neutral=2)
TWEET-ID(long)
POST_DATE
TWEET-QUERY
USER -TWEET (string)

Example:
"0","1550724000","Sat Apr 18 07:04:07 PDT 2009","NO_QUERY","trwiles","Wishing I was going to play golf today. Yard work instead. YAY "

The project was built using Python 2.7 and TensorFlow 1.3. In order to run the code, you might execute the following files in chronological order.
1. Data preprocessing and creation of dictionaries: "main_cnn.py"  takes the dataset and applies few preprocessing steps. The output will be:
  [1, 0]   <tweet>  (positive tweet, separator \t)
  [0, 1]   <tweet>  (negative tweet, separator \t)

Then it will create 2 different dictionaries (normal and inverted). (Please update the paths accordingly)

2. WordEmbedding subset. "subset_word_emb.py" We have to use the three files generated in step 1 (the preprocess dataset, and the 2 dictionaries) in order to upload to memory the dictionaries. Then we have to reference the pretrained word embedding text in the variable "word2vec" and also set the file path for the subset "outFile". The output will be a subset of the pretrained word embeddings.

3. Cnn: "cnn_shallow.py" or "cnn_deep.py". Update the paths of the files generated in the step 1 and 2. 
The default hyperparameters are under the "parser" ArgumentParser variable as well as the output directories and log directories and files. Moreover, the implementation splits the dataset into 60:20:20 (Training/validation and test)

The results of the sentiment analysis tasks are saved under the logfile. The model and the summaries are also saved under the default directories.

