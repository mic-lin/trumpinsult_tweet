# Trump Insulting Tweets ----- 
library(foreign)
library(psych) 
library(lsa)
sessionInfo() 

trumpinsult_tweet <- read.csv(file.choose(), header = TRUE, encoding="UTF-8", stringsAsFactors=FALSE)  # file.choose()
# trump_insult_tweets_new.csv

dim(trumpinsult_tweet)
head(trumpinsult_tweet) 

class(trumpinsult_tweet$date) 
trumpinsult_tweet$date <- as.Date(trumpinsult_tweet$date,"%d/%m/%Y")

names(trumpinsult_tweet)
names(trumpinsult_tweet)[names(trumpinsult_tweet) == 'X'] <- 'doc_id'
names(trumpinsult_tweet)[names(trumpinsult_tweet) == 'tweet'] <- 'text'

library(NLP)
library(tm)

tweets_trump_docs <- subset(trumpinsult_tweet, select = c("doc_id", "text"))
tweets_trump_VCorpus <- VCorpus(DataframeSource(tweets_trump_docs)) 

# to inspect the contents 
tweets_trump_VCorpus[[1]]$content

# to convert to all lower cases
tweets_trump_VCorpus <- tm_map(tweets_trump_VCorpus, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
tweets_trump_VCorpus[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets_trump_VCorpus <- tm_map(tweets_trump_VCorpus, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
tweets_trump_VCorpus <- tm_map(tweets_trump_VCorpus, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
tweets_trump_VCorpus <- tm_map(tweets_trump_VCorpus, removeWords, stopwords("english"))

# to remove extra whitespaces 
tweets_trump_VCorpus <- tm_map(tweets_trump_VCorpus, stripWhitespace) 

# to remove punctuations 
tweets_trump_VCorpus <- tm_map(tweets_trump_VCorpus, removePunctuation)
tweets_trump_VCorpus[[1]]$content

# Stemming: to remove plurals and action suffixes (please use it with caution: some hypertextual elements such as @mentions, #hashtags, and URLs are removed)
library(SnowballC)
tweets_trump_VCorpus <- tm_map(tweets_trump_VCorpus, stemDocument)
tweets_trump_VCorpus[[1]]$content

# TF and TF-IDF
# converting to Document-term matrix (TDM)
tweets_trump_dtm <- DocumentTermMatrix(tweets_trump_VCorpus, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
tweets_trump_dtm
# A high sparsity means terms are not repeated often among different documents.
inspect(tweets_trump_dtm) # a sample of the matrix 

# TF
term_freq_trump <- colSums(as.matrix(tweets_trump_dtm)) 
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_trump)), decreasing=TRUE)), file="tweets_trump_dtm_tf.csv")

# TF-IDF
tweets_trump_dtm_tfidf <- DocumentTermMatrix(tweets_trump_VCorpus, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(tweets_trump_dtm_tfidf) 
tweets_trump_dtm_tfidf2 = removeSparseTerms(tweets_trump_dtm_tfidf, 0.99)
print(tweets_trump_dtm_tfidf2) 
write.csv(as.data.frame(sort(colSums(as.matrix(tweets_trump_dtm_tfidf2)), decreasing=TRUE)), file="tweets_trump_dtm_tfidf.csv")

#topic modeling with LDA
#install.packages("topicmodels")
library(topicmodels)

# clean the empty (non-zero entry) 
rowTotals_trump <- apply(tweets_trump_dtm , 1, sum) #Find the sum of words in each Document
tweets_trump_dtm_nonzero <- tweets_trump_dtm[rowTotals_trump> 0, ]

library(ldatuning)
library(slam)

result <- FindTopicsNumber(
  tweets_trump_dtm_nonzero,
  topics = seq(from = 2, to = 15, by = 1),
  metrics = c("CaoJuan2009", "Arun2010", "Deveaud2014",'Griffiths2004'),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(result)
  
# after finding "the optimal-K" topics, then redo the above analysis
tweets_trump_dtm_6topics <- LDA(tweets_trump_dtm_nonzero, k = 6, method = "Gibbs", control = list(iter=2000, seed = 2000)) 
tweets_trump_dtm_6topics_10words <- terms(tweets_trump_dtm_6topics, 10) # get top 10 words of every topic
tweets_trump_dtm_6topics_10words

# repeated keywords in 2015 -----
library(lubridate)
trumpinsult_tweet2015 <- trumpinsult_tweet[year(trumpinsult_tweet$date)==2015,]

tweets_trump_docs2015 <- subset(trumpinsult_tweet2015, select = c("doc_id", "text"))
tweets_trump_VCorpus2015 <- VCorpus(DataframeSource(tweets_trump_docs2015)) 

# to inspect the contents 
tweets_trump_VCorpus2015[[1]]$content

# to convert to all lower cases
tweets_trump_VCorpus2015 <- tm_map(tweets_trump_VCorpus2015, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
tweets_trump_VCorpus2015[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets_trump_VCorpus2015 <- tm_map(tweets_trump_VCorpus2015, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
tweets_trump_VCorpus2015 <- tm_map(tweets_trump_VCorpus2015, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
tweets_trump_VCorpus2015 <- tm_map(tweets_trump_VCorpus2015, removeWords, stopwords("english"))

# to remove extra whitespaces 
tweets_trump_VCorpus2015 <- tm_map(tweets_trump_VCorpus2015, stripWhitespace) 

# to remove punctuations 
tweets_trump_VCorpus2015 <- tm_map(tweets_trump_VCorpus2015, removePunctuation)
tweets_trump_VCorpus2015[[1]]$content

# Stemming: to remove plurals and action suffixes (please use it with caution: some hypertextual elements such as @mentions, #hashtags, and URLs are removed)
library(SnowballC)
tweets_trump_VCorpus2015 <- tm_map(tweets_trump_VCorpus2015, stemDocument)
tweets_trump_VCorpus2015[[1]]$content

# TF and TF-IDF
# converting to Document-term matrix (TDM)
tweets_trump_dtm2015 <- DocumentTermMatrix(tweets_trump_VCorpus2015, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
tweets_trump_dtm2015
# A high sparsity means terms are not repeated often among different documents.
inspect(tweets_trump_dtm2015) # a sample of the matrix 

# TF
term_freq_trump2015 <- colSums(as.matrix(tweets_trump_dtm2015)) 
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_trump2015)), decreasing=TRUE)), file="tweets_trump_dtm_tf2015.csv")

# TF-IDF
tweets_trump_dtm_tfidf2015 <- DocumentTermMatrix(tweets_trump_VCorpus2015, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(tweets_trump_dtm_tfidf2015) 
tweets_trump_dtm_tfidf22015 = removeSparseTerms(tweets_trump_dtm_tfidf2015, 0.99)
print(tweets_trump_dtm_tfidf22015) 
write.csv(as.data.frame(sort(colSums(as.matrix(tweets_trump_dtm_tfidf22015)), decreasing=TRUE)), file="tweets_trump_dtm_tfidf2015.csv")

#topic modeling with LDA
library(topicmodels)

# clean the empty (non-zero entry) 
rowTotals_trump2015 <- apply(tweets_trump_dtm2015 , 1, sum) #Find the sum of words in each Document
tweets_trump_dtm_nonzero2015 <- tweets_trump_dtm2015[rowTotals_trump2015> 0, ]

result <- FindTopicsNumber(
  tweets_trump_dtm_nonzero2015,
  topics = seq(from = 2, to = 15, by = 1),
  metrics = c("CaoJuan2009", "Arun2010", "Deveaud2014",'Griffiths2004'),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(result)


tweets_trump_dtm_topics2015 <- LDA(tweets_trump_dtm_nonzero2015, k = 5, method = "Gibbs", control = list(iter=2000, seed = 2000)) 
tweets_trump_dtm_topics2015_10words <- terms(tweets_trump_dtm_topics2015, 10) # get top 10 words of every topic
tweets_trump_dtm_topics2015_10words


# repeated keywords in 2016 -----

trumpinsult_tweet2016 <- trumpinsult_tweet[year(trumpinsult_tweet$date)==2016,]
tweets_trump_docs2016 <- subset(trumpinsult_tweet2016, select = c("doc_id", "text"))
tweets_trump_VCorpus2016 <- VCorpus(DataframeSource(tweets_trump_docs2016)) 

# to inspect the contents 
tweets_trump_VCorpus2016[[1]]$content

# to convert to all lower cases
tweets_trump_VCorpus2016 <- tm_map(tweets_trump_VCorpus2016, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
tweets_trump_VCorpus2016[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets_trump_VCorpus2016 <- tm_map(tweets_trump_VCorpus2016, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
tweets_trump_VCorpus2016 <- tm_map(tweets_trump_VCorpus2016, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
tweets_trump_VCorpus2016 <- tm_map(tweets_trump_VCorpus2016, removeWords, stopwords("english"))

# to remove extra whitespaces 
tweets_trump_VCorpus2016 <- tm_map(tweets_trump_VCorpus2016, stripWhitespace) 

# to remove punctuations 
tweets_trump_VCorpus2016 <- tm_map(tweets_trump_VCorpus2016, removePunctuation)
tweets_trump_VCorpus2016[[1]]$content

# Stemming: to remove plurals and action suffixes (please use it with caution: some hypertextual elements such as @mentions, #hashtags, and URLs are removed)
library(SnowballC)
tweets_trump_VCorpus2016 <- tm_map(tweets_trump_VCorpus2016, stemDocument)
tweets_trump_VCorpus2016[[1]]$content

# TF and TF-IDF
# converting to Document-term matrix (TDM)
tweets_trump_dtm2016 <- DocumentTermMatrix(tweets_trump_VCorpus2016, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
tweets_trump_dtm2016
# A high sparsity means terms are not repeated often among different documents.
inspect(tweets_trump_dtm2016) # a sample of the matrix 

# TF
term_freq_trump2016 <- colSums(as.matrix(tweets_trump_dtm2016)) 
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_trump2016)), decreasing=TRUE)), file="tweets_trump_dtm_tf2016.csv")

# TF-IDF
tweets_trump_dtm_tfidf2016 <- DocumentTermMatrix(tweets_trump_VCorpus2016, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(tweets_trump_dtm_tfidf2016) 
tweets_trump_dtm_tfidf22016 = removeSparseTerms(tweets_trump_dtm_tfidf2016, 0.99)
print(tweets_trump_dtm_tfidf22016) 
write.csv(as.data.frame(sort(colSums(as.matrix(tweets_trump_dtm_tfidf22016)), decreasing=TRUE)), file="tweets_trump_dtm_tfidf2016.csv")

#topic modeling with LDA
library(topicmodels)

# clean the empty (non-zero entry) 
rowTotals_trump2016 <- apply(tweets_trump_dtm2016 , 1, sum) #Find the sum of words in each Document
tweets_trump_dtm_nonzero2016 <- tweets_trump_dtm2016[rowTotals_trump2016> 0, ]

tweets_trump_dtm_topics2016 <- LDA(tweets_trump_dtm_nonzero2016, k = 5, method = "Gibbs", control = list(iter=2000, seed = 2000)) 
tweets_trump_dtm_topics2016_10words <- terms(tweets_trump_dtm_topics2016, 10) # get top 10 words of every topic
tweets_trump_dtm_topics2016_10words


# repeated keywords in 2017 -----
trumpinsult_tweet2017 <- trumpinsult_tweet[year(trumpinsult_tweet$date)==2017,]
tweets_trump_docs2017 <- subset(trumpinsult_tweet2017, select = c("doc_id", "text"))
tweets_trump_VCorpus2017 <- VCorpus(DataframeSource(tweets_trump_docs2017)) 

# to inspect the contents 
tweets_trump_VCorpus2017[[1]]$content

# to convert to all lower cases
tweets_trump_VCorpus2017 <- tm_map(tweets_trump_VCorpus2017, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
tweets_trump_VCorpus2017[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets_trump_VCorpus2017 <- tm_map(tweets_trump_VCorpus2017, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
tweets_trump_VCorpus2017 <- tm_map(tweets_trump_VCorpus2017, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
tweets_trump_VCorpus2017 <- tm_map(tweets_trump_VCorpus2017, removeWords, stopwords("english"))

# to remove extra whitespaces 
tweets_trump_VCorpus2017 <- tm_map(tweets_trump_VCorpus2017, stripWhitespace) 

# to remove punctuations 
tweets_trump_VCorpus2017 <- tm_map(tweets_trump_VCorpus2017, removePunctuation)
tweets_trump_VCorpus2017[[1]]$content

# Stemming: to remove plurals and action suffixes (please use it with caution: some hypertextual elements such as @mentions, #hashtags, and URLs are removed)
library(SnowballC)
tweets_trump_VCorpus2017 <- tm_map(tweets_trump_VCorpus2017, stemDocument)
tweets_trump_VCorpus2017[[1]]$content

# TF and TF-IDF
# converting to Document-term matrix (TDM)
tweets_trump_dtm2017 <- DocumentTermMatrix(tweets_trump_VCorpus2017, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
tweets_trump_dtm2017
# A high sparsity means terms are not repeated often among different documents.
inspect(tweets_trump_dtm2017) # a sample of the matrix 

# TF
term_freq_trump2017 <- colSums(as.matrix(tweets_trump_dtm2017)) 
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_trump2017)), decreasing=TRUE)), file="tweets_trump_dtm_tf2017.csv")

# TF-IDF
tweets_trump_dtm_tfidf2017 <- DocumentTermMatrix(tweets_trump_VCorpus2017, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(tweets_trump_dtm_tfidf2017) 
tweets_trump_dtm_tfidf22017 = removeSparseTerms(tweets_trump_dtm_tfidf2017, 0.99)
print(tweets_trump_dtm_tfidf22017) 
write.csv(as.data.frame(sort(colSums(as.matrix(tweets_trump_dtm_tfidf22017)), decreasing=TRUE)), file="tweets_trump_dtm_tfidf2017.csv")

#topic modeling with LDA
library(topicmodels)

# clean the empty (non-zero entry) 
rowTotals_trump2017 <- apply(tweets_trump_dtm2017 , 1, sum) #Find the sum of words in each Document
tweets_trump_dtm_nonzero2017 <- tweets_trump_dtm2017[rowTotals_trump2017> 0, ]

tweets_trump_dtm_topics2017 <- LDA(tweets_trump_dtm_nonzero2017, k = 5, method = "Gibbs", control = list(iter=2000, seed = 2000)) 
tweets_trump_dtm_topics2017_10words <- terms(tweets_trump_dtm_topics2017, 10) # get top 10 words of every topic
tweets_trump_dtm_topics2017_10words


# repeated keywords in 2018 -----
trumpinsult_tweet2018 <- trumpinsult_tweet[year(trumpinsult_tweet$date)==2018,]
tweets_trump_docs2018 <- subset(trumpinsult_tweet2018, select = c("doc_id", "text"))
tweets_trump_VCorpus2018 <- VCorpus(DataframeSource(tweets_trump_docs2018)) 

# to inspect the contents 
tweets_trump_VCorpus2018[[1]]$content

# to convert to all lower cases
tweets_trump_VCorpus2018 <- tm_map(tweets_trump_VCorpus2018, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
tweets_trump_VCorpus2018[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets_trump_VCorpus2018 <- tm_map(tweets_trump_VCorpus2018, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
tweets_trump_VCorpus2018 <- tm_map(tweets_trump_VCorpus2018, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
tweets_trump_VCorpus2018 <- tm_map(tweets_trump_VCorpus2018, removeWords, stopwords("english"))

# to remove extra whitespaces 
tweets_trump_VCorpus2018 <- tm_map(tweets_trump_VCorpus2018, stripWhitespace) 

# to remove punctuations 
tweets_trump_VCorpus2018 <- tm_map(tweets_trump_VCorpus2018, removePunctuation)
tweets_trump_VCorpus2018[[1]]$content

# Stemming: to remove plurals and action suffixes (please use it with caution: some hypertextual elements such as @mentions, #hashtags, and URLs are removed)
library(SnowballC)
tweets_trump_VCorpus2018 <- tm_map(tweets_trump_VCorpus2018, stemDocument)
tweets_trump_VCorpus2018[[1]]$content

# TF and TF-IDF
# converting to Document-term matrix (TDM)
tweets_trump_dtm2018 <- DocumentTermMatrix(tweets_trump_VCorpus2018, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
tweets_trump_dtm2018
# A high sparsity means terms are not repeated often among different documents.
inspect(tweets_trump_dtm2018) # a sample of the matrix 

# TF
term_freq_trump2018 <- colSums(as.matrix(tweets_trump_dtm2018)) 
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_trump2018)), decreasing=TRUE)), file="tweets_trump_dtm_tf2018.csv")

# TF-IDF
tweets_trump_dtm_tfidf2018 <- DocumentTermMatrix(tweets_trump_VCorpus2018, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(tweets_trump_dtm_tfidf2018) 
tweets_trump_dtm_tfidf22018 = removeSparseTerms(tweets_trump_dtm_tfidf2018, 0.99)
print(tweets_trump_dtm_tfidf22018) 
write.csv(as.data.frame(sort(colSums(as.matrix(tweets_trump_dtm_tfidf22018)), decreasing=TRUE)), file="tweets_trump_dtm_tfidf2018.csv")

#topic modeling with LDA
library(topicmodels)

# clean the empty (non-zero entry) 
rowTotals_trump2018 <- apply(tweets_trump_dtm2018 , 1, sum) #Find the sum of words in each Document
tweets_trump_dtm_nonzero2018 <- tweets_trump_dtm2018[rowTotals_trump2018> 0, ]

tweets_trump_dtm_topics2018 <- LDA(tweets_trump_dtm_nonzero2018, k = 5, method = "Gibbs", control = list(iter=2000, seed = 2000)) 
tweets_trump_dtm_topics2018_10words <- terms(tweets_trump_dtm_topics2018, 10) # get top 10 words of every topic
tweets_trump_dtm_topics2018_10words


# repeated keywords in 2019 -----
trumpinsult_tweet2019 <- trumpinsult_tweet[year(trumpinsult_tweet$date)==2019,]
tweets_trump_docs2019 <- subset(trumpinsult_tweet2019, select = c("doc_id", "text"))
tweets_trump_VCorpus2019 <- VCorpus(DataframeSource(tweets_trump_docs2019)) 

# to inspect the contents 
tweets_trump_VCorpus2019[[1]]$content

# to convert to all lower cases
tweets_trump_VCorpus2019 <- tm_map(tweets_trump_VCorpus2019, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
tweets_trump_VCorpus2019[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets_trump_VCorpus2019 <- tm_map(tweets_trump_VCorpus2019, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
tweets_trump_VCorpus2019 <- tm_map(tweets_trump_VCorpus2019, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
tweets_trump_VCorpus2019 <- tm_map(tweets_trump_VCorpus2019, removeWords, stopwords("english"))

# to remove extra whitespaces 
tweets_trump_VCorpus2019 <- tm_map(tweets_trump_VCorpus2019, stripWhitespace) 

# to remove punctuations 
tweets_trump_VCorpus2019 <- tm_map(tweets_trump_VCorpus2019, removePunctuation)
tweets_trump_VCorpus2019[[1]]$content

# Stemming: to remove plurals and action suffixes (please use it with caution: some hypertextual elements such as @mentions, #hashtags, and URLs are removed)
library(SnowballC)
tweets_trump_VCorpus2019 <- tm_map(tweets_trump_VCorpus2019, stemDocument)
tweets_trump_VCorpus2019[[1]]$content

# TF and TF-IDF
# converting to Document-term matrix (TDM)
tweets_trump_dtm2019 <- DocumentTermMatrix(tweets_trump_VCorpus2019, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
tweets_trump_dtm2019
# A high sparsity means terms are not repeated often among different documents.
inspect(tweets_trump_dtm2019) # a sample of the matrix 

# TF
term_freq_trump2019 <- colSums(as.matrix(tweets_trump_dtm2019)) 
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_trump2019)), decreasing=TRUE)), file="tweets_trump_dtm_tf2019.csv")

# TF-IDF
tweets_trump_dtm_tfidf2019 <- DocumentTermMatrix(tweets_trump_VCorpus2019, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(tweets_trump_dtm_tfidf2019) 
tweets_trump_dtm_tfidf22019 = removeSparseTerms(tweets_trump_dtm_tfidf2019, 0.99)
print(tweets_trump_dtm_tfidf22019) 
write.csv(as.data.frame(sort(colSums(as.matrix(tweets_trump_dtm_tfidf22019)), decreasing=TRUE)), file="tweets_trump_dtm_tfidf2019.csv")

#topic modeling with LDA
library(topicmodels)

# clean the empty (non-zero entry) 
rowTotals_trump2019 <- apply(tweets_trump_dtm2019 , 1, sum) #Find the sum of words in each Document
tweets_trump_dtm_nonzero2019 <- tweets_trump_dtm2019[rowTotals_trump2019 > 0, ]

tweets_trump_dtm_topics2019 <- LDA(tweets_trump_dtm_nonzero2019, k = 5, method = "Gibbs", control = list(iter=2000, seed = 2000)) 
tweets_trump_dtm_topics2019_10words <- terms(tweets_trump_dtm_topics2019, 10) # get top 10 words of every topic
tweets_trump_dtm_topics2019_10words


# repeated keywords in 2020 -----
trumpinsult_tweet2020 <- trumpinsult_tweet[year(trumpinsult_tweet$date)==2020,]
tweets_trump_docs2020 <- subset(trumpinsult_tweet2020, select = c("doc_id", "text"))
tweets_trump_VCorpus2020 <- VCorpus(DataframeSource(tweets_trump_docs2020)) 

# to inspect the contents 
tweets_trump_VCorpus2020[[1]]$content

# to convert to all lower cases
tweets_trump_VCorpus2020 <- tm_map(tweets_trump_VCorpus2020, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
tweets_trump_VCorpus2020[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets_trump_VCorpus2020 <- tm_map(tweets_trump_VCorpus2020, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
tweets_trump_VCorpus2020 <- tm_map(tweets_trump_VCorpus2020, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
tweets_trump_VCorpus2020 <- tm_map(tweets_trump_VCorpus2020, removeWords, stopwords("english"))

# to remove extra whitespaces 
tweets_trump_VCorpus2020 <- tm_map(tweets_trump_VCorpus2020, stripWhitespace) 

# to remove punctuations 
tweets_trump_VCorpus2020 <- tm_map(tweets_trump_VCorpus2020, removePunctuation)
tweets_trump_VCorpus2020[[1]]$content

# Stemming: to remove plurals and action suffixes (please use it with caution: some hypertextual elements such as @mentions, #hashtags, and URLs are removed)
library(SnowballC)
tweets_trump_VCorpus2020 <- tm_map(tweets_trump_VCorpus2020, stemDocument)
tweets_trump_VCorpus2020[[1]]$content

# TF and TF-IDF
# converting to Document-term matrix (TDM)
tweets_trump_dtm2020 <- DocumentTermMatrix(tweets_trump_VCorpus2020, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
tweets_trump_dtm2020
# A high sparsity means terms are not repeated often among different documents.
inspect(tweets_trump_dtm2020) # a sample of the matrix 

# TF
term_freq_trump2020 <- colSums(as.matrix(tweets_trump_dtm2020)) 
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_trump2020)), decreasing=TRUE)), file="tweets_trump_dtm_tf2020.csv")

# TF-IDF
tweets_trump_dtm_tfidf2020 <- DocumentTermMatrix(tweets_trump_VCorpus2020, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(tweets_trump_dtm_tfidf2020) 
tweets_trump_dtm_tfidf22020 = removeSparseTerms(tweets_trump_dtm_tfidf2020, 0.99)
print(tweets_trump_dtm_tfidf22020) 
write.csv(as.data.frame(sort(colSums(as.matrix(tweets_trump_dtm_tfidf22020)), decreasing=TRUE)), file="tweets_trump_dtm_tfidf2020.csv")

#topic modeling with LDA
library(topicmodels)

# clean the empty (non-zero entry) 
rowTotals_trump2020 <- apply(tweets_trump_dtm2020 , 1, sum) #Find the sum of words in each Document
tweets_trump_dtm_nonzero2020 <- tweets_trump_dtm2020[rowTotals_trump2020> 0, ]

tweets_trump_dtm_topics2020 <- LDA(tweets_trump_dtm_nonzero2020, k = 5, method = "Gibbs", control = list(iter=2000, seed = 2000)) 
tweets_trump_dtm_topics2020_10words <- terms(tweets_trump_dtm_topics2020, 10) # get top 10 words of every topic
tweets_trump_dtm_topics2020_10words



# repeated keywords in 2021 -----
trumpinsult_tweet2021 <- trumpinsult_tweet[year(trumpinsult_tweet$date)==2021,]
tweets_trump_docs2021 <- subset(trumpinsult_tweet2021, select = c("doc_id", "text"))
tweets_trump_VCorpus2021 <- VCorpus(DataframeSource(tweets_trump_docs2021)) 

# to inspect the contents 
tweets_trump_VCorpus2021[[1]]$content

# to convert to all lower cases
tweets_trump_VCorpus2021 <- tm_map(tweets_trump_VCorpus2021, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
tweets_trump_VCorpus2021[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
tweets_trump_VCorpus2021 <- tm_map(tweets_trump_VCorpus2021, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
tweets_trump_VCorpus2021 <- tm_map(tweets_trump_VCorpus2021, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
tweets_trump_VCorpus2021 <- tm_map(tweets_trump_VCorpus2021, removeWords, stopwords("english"))

# to remove extra whitespaces 
tweets_trump_VCorpus2021 <- tm_map(tweets_trump_VCorpus2021, stripWhitespace) 

# to remove punctuations 
tweets_trump_VCorpus2021 <- tm_map(tweets_trump_VCorpus2021, removePunctuation)
tweets_trump_VCorpus2021[[1]]$content

# Stemming: to remove plurals and action suffixes (please use it with caution: some hypertextual elements such as @mentions, #hashtags, and URLs are removed)
library(SnowballC)
tweets_trump_VCorpus2021 <- tm_map(tweets_trump_VCorpus2021, stemDocument)
tweets_trump_VCorpus2021[[1]]$content

# TF and TF-IDF
# converting to Document-term matrix (TDM)
tweets_trump_dtm2021 <- DocumentTermMatrix(tweets_trump_VCorpus2021, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
tweets_trump_dtm2021
# A high sparsity means terms are not repeated often among different documents.
inspect(tweets_trump_dtm2021) # a sample of the matrix 

# TF
term_freq_trump2021 <- colSums(as.matrix(tweets_trump_dtm2021)) 
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_trump2021)), decreasing=TRUE)), file="tweets_trump_dtm_tf2021.csv")

# TF-IDF
tweets_trump_dtm_tfidf2021 <- DocumentTermMatrix(tweets_trump_VCorpus2021, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(tweets_trump_dtm_tfidf2021) 
tweets_trump_dtm_tfidf22021 = removeSparseTerms(tweets_trump_dtm_tfidf2021, 0.99)
print(tweets_trump_dtm_tfidf22021) 
write.csv(as.data.frame(sort(colSums(as.matrix(tweets_trump_dtm_tfidf22021)), decreasing=TRUE)), file="tweets_trump_dtm_tfidf2021.csv")

#topic modeling with LDA
library(topicmodels)

# clean the empty (non-zero entry) 
rowTotals_trump2021 <- apply(tweets_trump_dtm2021 , 1, sum) #Find the sum of words in each Document
tweets_trump_dtm_nonzero2021 <- tweets_trump_dtm2021[rowTotals_trump2021> 0, ]

tweets_trump_dtm_topics2021 <- LDA(tweets_trump_dtm_nonzero2021, k = 5, method = "Gibbs", control = list(iter=2000, seed = 2000)) 
tweets_trump_dtm_topics2021_10words <- terms(tweets_trump_dtm_topics2021, 10) # get top 10 words of every topic
tweets_trump_dtm_topics2021_10words
###################################################### the end of the codes ---



