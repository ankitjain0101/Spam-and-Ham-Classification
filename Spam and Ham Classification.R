dev.off() # close plots
rm(list=ls()) # wipe environment
library(DataExplorer)
library(ggplot2)
library(psych)
library(corrplot)
library(gmodels)
library(caret)
library(reshape2)
library(e1071)
library(tm)
library(wordcloud)
library(SnowballC)

sms_raw <- read.csv("E:/College/Analytics/Predictive/Assignment 2/sms_spam.csv", stringsAsFactors = FALSE)
str(sms_raw)

#Data Preparation
sms_raw$type<-factor(sms_raw$type)
table(sms_raw$type)

sms_corpus<-Corpus(VectorSource(sms_raw$text))
print(sms_corpus)
inspect(sms_corpus[5])

summary(sms_raw)
describe(sms_raw)
summary(is.na(sms_raw))
plot_missing(sms_raw)

plot_correlation(sms_raw)

#Remove numbers, capitalization, common words, punctuation, and otherwise prepare your texts for analysis.

corpus_clean <- tm_map(sms_corpus, tolower)

for (j in seq(corpus_clean)) {
  corpus_clean[[j]] <- gsub("/", " ", corpus_clean[[j]])
  corpus_clean[[j]] <- gsub("@", " ", corpus_clean[[j]])
  corpus_clean[[j]] <- gsub("\\|", " ", corpus_clean[[j]])
  corpus_clean[[j]] <- gsub("\u2028", " ", corpus_clean[[j]])
}

corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stemDocument)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

inspect(corpus_clean[5])

sms_dtm <- DocumentTermMatrix(corpus_clean)

#Training and Test datasets
sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test <- sms_raw[4170:5559, ]
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]
sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5559]

prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))


wordcloud(sms_corpus_train, min.freq = 40,colors = brewer.pal(7, "Paired"), random.order = FALSE)

spam <- subset(sms_raw_train, type == "spam")
ham <- subset(sms_raw_train, type == "ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 1.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 1.5))

sms_dict <- c(findFreqTerms(sms_dtm_train, 5))

sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))

convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}

sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_test, MARGIN = 2, convert_counts)

summary(sms_train[, 1:5])

# Naive Bayes
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)

sms_classifier$tables[1:2]

# Evaluating model performance 
sms_test_pred <- predict(sms_classifier, sms_test)

CrossTable(sms_test_pred, sms_raw_test$type,prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))
confusionMatrix(sms_test_pred,sms_raw_test$type)

# Improving model performance
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type,laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_test_pred2, sms_raw_test$type,prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))
confusionMatrix(sms_test_pred2,sms_raw_test$type)


