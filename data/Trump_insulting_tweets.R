# Trump Insulting Tweets ----- 
library(foreign)
library(psych) 
library(lsa)

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


###################################################### the end of the codes ---



