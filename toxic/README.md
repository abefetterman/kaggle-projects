# TGS Salt Identification Challenge

[Link to competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

Identify and classify toxic online comments.

## Model and Training

Used a 2-Layer BiLSTM (or BiGRU) with various pre-trained embeddings. Training 
a single model with ADAM took only a few epochs, but better results were obtained by 
ensembling among multiple embeddings and models (including linear models).

