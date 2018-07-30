# SET PATH

packages <- c("jsonlite", "dplyr", "purrr", "ggplot2")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)
library(tm)
library(SnowballC)
library(stringr)
library(slam)
library(tidytext)
library(tidyr)
library(topicmodels)
set.seed(123)


df <- fromJSON("reddit_jokes.json")

df$num_id <- strtoi(df$id, base = 36) #the id's are actually base 36


###################################################### - Remove duplicates based on body (not ideal)
dups<- df %>%
  select(body)%>%
  duplicated()
dups <- df[dups, c("body", "num_id", "score")]
nrow(dups) # number of exact duplicates

uniq_dups <- unique(dups$body)

uniq.df <- data_frame(body = unique(df$body), unq_id = as.factor(1:length(unique(df$body))))
df <- df %>%
  left_join(uniq.df) %>%
  group_by(unq_id) %>%
  slice(which.max(score))

df$document <- as.character(1:nrow(df))

###################################################### - Preprocessing and creating corpus

df$fullbody <- paste(df$title, df$body)                   # combine title and body
df$fullbody <- gsub("\r?\n|\r", " ", df$fullbody)         # remove new lines
df$fullbody <- gsub("\\.", " ", df$fullbody)              # remove periods
df$fullbody <- gsub("-", " ", df$fullbody, fixed = TRUE)  # remove "-"
df$fullbody <- gsub("[^[:alnum:]///' ]", "", df$fullbody) # remove anything that isn't alphanumeric

doc.vec <- VectorSource(df$fullbody)
doc.corpus <- Corpus(doc.vec)

doc.corpus <- tm_map(doc.corpus, content_transformer(tolower))
doc.corpus <- tm_map(doc.corpus, removeNumbers)
doc.corpus <- tm_map(doc.corpus, removePunctuation)
doc.corpus <- tm_map(doc.corpus, removeWords, stopwords('english'))
doc.corpus <- tm_map(doc.corpus, stripWhitespace)


###################################################### - Creating and pre-processing document-term matrix
dtm <- DocumentTermMatrix(doc.corpus)

# Remove words that occur less than 5 times
a<-col_sums(dtm, na.rm = TRUE)                            
sum(as.numeric(a<5))
dtm <- dtm[,!(a<5)]

# Remove documents that now have no words (aka, only had words that occurred less than 5 times)
a<-row_sums(dtm, na.rm = TRUE)==0
dtm <- dtm[!a,] # 28k words

###################################################### - Fitting LDA model

burnin <- 1500 
iter <- 1500
keep <- 100 

fitted <- LDA(dtm, k = 50, method = "Gibbs",
              control = list(burnin = burnin, iter = iter, keep = keep))


ldaTopics <- topics(fitted) # joke topic assignments
topicProbs <- as.data.frame(fitted@gamma)

###################################################### - Get corresponding dataframe
# Since we removed some rows in the dtm we need to do 
# the same to the original dataset

remDocs <- as.character(which(a))
df_f <- df[!(df$document %in% remDocs),] %>% ungroup()
df_f$topic <- ldaTopics

###################################################### - Getting dataset ready 
# Log-transform the score
df_f$score <- log(df_f$score + 1)

# Create dataframe of term probabilities
termProbs <- as.data.frame(t(posterior(fitted)$terms))
termProbs$term <- rownames(termProbs)

# We need to create some datasets that'll be useful
# word_df -> words and their topic probs
# tidys -> a tidy dataset that connect words to jokes

tidys <- tidy(t(dtm))

temp <- df %>%
  ungroup()%>%
  select(score, document)

tidys <- tidys %>%
  left_join(temp, by = "document")%>%
  bind_tf_idf(term, document, count)


# Create full word dataset
wordNum <- 1:ncol(dtm)
words <- dtm$dimnames[[2]]

word_df <- data.frame(wordNum = wordNum, term = words) %>%
  left_join(termProbs, by = "term")

names(word_df) <- c("wordNum", "term", 1:50)

temp <- tidys %>%
  group_by(term)%>%
  summarise(count = sum(count))

word_df <- word_df %>%
  left_join(temp, by = 'term')


# gather topics
word_df <- word_df %>%
  gather(topic, prob, -count, -wordNum, -term)


# Besides structuring, we also care about the size
# of the file, so we'll be doing some filtering

# Filter to max 1000 words per topic.
num_words <- 500
word_df_filt <- word_df %>%
  group_by(topic)%>%
  arrange(-prob)%>%
  top_n(num_words, wt = prob)%>%
  ungroup()


# current version tidys_filt will give us the documents that contain the words we want
tidys_filt <- tidys %>%
  select(term, document)%>%
  semi_join(word_df_filt, by = "term")

# final version of tidys filt contains the intersection between top 10k jokes and our wordslist
tidys_filt <- df_f %>%
  arrange(-score)%>%
  select(document, score)%>%
  head(10000)%>%
  inner_join(tidys_filt, by = "document")


# finally, we want to select the top 30 jokes associated with each word

word_df_filt <- tidys_filt %>%
  group_by(term)%>%
  top_n(30, score)%>%
  summarise(joke_ids = list(unique(document)))%>%
  right_join(word_df_filt, by = "term")

# we good here. word_df_filt has it all.

# we also need a filtered joke df as a second json object

filtered_df <- tidys_filt%>%
  select(document) %>%
  distinct()%>%
  inner_join(df_f, by = "document")

# Now we want to add the following to the word_df_filtered file as of 12-2-17
# - vector of 50 probabilities per word. one for each topic
# - count of time word used in jokes of this topic

k = df_f %>% select(document, topic)%>%
  left_join(tidys, by = "document")

k = k %>% group_by(term, topic)%>%
  summarise(topic_count = n())

k$topic <- as.character(k$topic)

k <- k[k$term %in% unique(word_df_filt$term),]

k2 <- word_df_filt %>%
  left_join(k, by = c("term", "topic")) # NA's introduced

k2[is.na(k2)] <- 0
word_df_filt <- k2


# Get a distribution of topic probabilities for each word
termProbVector <- function(word){
  word <- word[1]
  t <- termProbs %>% filter(term == word) %>% select(-term)
  t <- list(as.numeric(t))
  return(t)
}

wordDists <- termProbs %>% select(term) %>% apply(1, function(x) termProbVector(x))
k <- data.frame(term = termProbs$term)
k$topic_dist <- wordDists
wordDists <- k
wordDists <- wordDists[wordDists$term %in% unique(word_df_filt$term),]
word_df_filt <- word_df_filt %>%
  left_join(wordDists, by = "term")

###################################################### - Creating json objects for visualization
# Creating Topics
library(RJSONIO)
num_topics = 50
createTopic <- function(topic_i, df_f, num_words){
  dat <- df_f %>% filter(topic == topic_i)%>%select(score) # scores should be from whole dataset, not just 
  
  scores <- dat$score
  
  temp <- word_df_filt %>% filter(topic == topic_i)%>%
    arrange(desc(prob))%>%select(term)
  n <- min(num_words, nrow(temp))
  word_list <- temp$term[1:n]
  indx_w <- 1
  wordlist <- list()
  for(word_i in word_list){
    dat <- word_df_filt %>% filter(topic == topic_i, term == word_i)
    wordlist[[indx_w]] <- list(name = dat$term, prob = dat$prob, 
                               count = dat$count, count_topic = dat$topic_count, 
                               topic_dist = unlist(dat$topic_dist),
                               joke_ids = unlist(dat$joke_ids))
    indx_w <- indx_w + 1
  }
  topiclist <- list(number = topic_i, words = wordlist, scores = scores)
  topicfile <- toJSON(topiclist)
  write(topicfile, paste("topic_", topic_i, ".json", sep = ""))
  print(topic_i)
}


for(i in 1:num_topics){
  i <- as.character(i)
  createTopic(i, df_f, num_words = num_words)
}

# Creating joke files

filtered_df <- filtered_df %>%
  arrange(-score)

jokeVec <- 1:nrow(filtered_df) # this assumes filtered_df is sorted by joke

jokelist <- list()
i = 1
for(joke_i in jokeVec){
  dat <- filtered_df[joke_i, ]
  jokelist[[i]] <- list(id = dat$document, title = dat$title, topic = as.character(dat$topic),
                        body = dat$body, score = dat$score)
  i = i + 1
}
jokefile <- toJSON(jokelist)
write(jokefile, "jokefile.json")

# Topic file with score distribution

topic_scores = df_f %>% group_by(topic)%>%
  summarise(topic_score = mean(score))%>%
  arrange(topic)
topic_joke_count = df_f %>% group_by(topic)%>%
  summarise(joke_count = n())%>%
  arrange(topic)

topic_scores = topic_scores$topic_score
topic_joke_count = topic_joke_count$joke_count
topic_scores_list <- list()

for(i in 1:50){
  topic_scores_list[[i]] = list(topic_scores[i], topic_joke_count[i])
}

topic_score_file = toJSON(topic_scores_list)
write(topic_score_file, file = "topic_scores.json")

