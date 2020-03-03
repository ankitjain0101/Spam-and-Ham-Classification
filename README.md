# Spam-and-Ham-Classification
  Mobile phone has become essential alongside the development of wireless communication techniques. Numerous public institutions and private enterprises use the SMSs (Short Message Service) for informing or notifying their clients. This surge of SMS goes through the issue of spam SMS that are produced by various users. Spam can be characterized as unsolicited (undesirable, junk) email for a beneficiary or any email that the users don't had any desire to have in their inboxes. Spam filtering is a special problem in the field of document classification and machine learning.

The dataset I have used is adapted from the SMS Spam Collection to create a Spam classifier at http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/. This data includes SMS messages and labels indicating if the message is spam or ham. We will look at classifying SMS messages using the Naive Bayes Machine Learning model

## Data Cleaning and Feature Engineering
SMS messages are strings of text composed of words, spaces, numbers, and punctuation. Dealing this kind of complex information requires a lot of thought and effort.
First step is to clean up the data by removing numbers, converting to lowercase, removing punctuation, removing stop words and applying stemming. Then, next create a Document Term Matrix (DTM) to split these text messages into individual words through tokenization, single component of words. A DTM is a data structure where each record is represented in its own row and each word is represented in its own column. It is additionally a sparse matrix, where most of its entries are populated with zeros.

## Naive Bayes
Naive Bayes is a simple, yet effective and commonly-used, machine learning classifier. It is a probabilistic classifier that makes classifications using the Maximum A Posteriori decision rule in a Bayesian setting.It's popular for text classification, and are a traditional solution for problems such as spam detection.

First, split the data into train and test datasets, then transform the sparse matrix into something the Naive Bayes model can train. Next, filter the DTM sparse matrix to only contain words that are most frequent to reduce features in the DTM.
Finally, tuned Laplace estimator parameter to improve model performance.
