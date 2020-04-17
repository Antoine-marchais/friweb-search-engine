# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: 'Python 3.6.9 64-bit (''friweb'': venv)'
#     name: python36964bitfriwebvenv039b39cfcf0545998426d5bebb76417f
# ---

# # Processing test

# ## Imports

# +
import sys
import os
import pickle as pkl
sys.path.append(os.path.abspath("../src"))

import preprocess
import config
from tqdm import tqdm
# -

# loading corpus :

with open("../data/corpus.pkl","rb") as f:
    corpus = pkl.load(f)

# ## Vocabulary and token count

# Preprocessing corpus :

# +
corpus_without_stp = preprocess.remove_stop_words_collection(corpus,"../data/stop_words.txt")
corpus_lemmatized = preprocess.lemmatize_collection(corpus_without_stp)

with open("../data/corpus_without_stp", "wb") as f:
    pkl.dump(corpus_without_stp, f)

with open("../data/corpus_lemmatized", "wb") as f:
    pkl.dump(corpus_lemmatized, f)


# -

# First we check the influence of our operations on the number of tokens in the collection :

def count_tokens(corpus):
    n_tokens = 0
    for key in corpus:
        n_tokens += len(corpus[key])
    return n_tokens


# +
init_count = count_tokens(corpus)
without_stp_count = count_tokens(corpus_without_stp)
lemmatized_count = count_tokens(corpus_lemmatized)

print("token count : \n" +
    f"  initial : {init_count} \n" +
    f"  without stp : {without_stp_count} \n" +
    f"  after lemmatization : {lemmatized_count}")

# +
from collections import Counter

def get_vocabulary(corpus):
    vocabulary = set()
    for key in corpus:
        vocabulary.update(corpus[key])
    return vocabulary

def get_frequencies(corpus):
    freqs = Counter()
    for key in corpus:
        freqs.update(corpus[key])
    return freqs
        


# -

# Now we check the influence of these operations on the length of our vocabulary :

# +
init_vocab = get_vocabulary(corpus)
without_stp_vocab = get_vocabulary(corpus_without_stp)
lemmatized_vocab = get_vocabulary(corpus_lemmatized)

print("vocabulary size : \n" +
    f"  initial : {len(init_vocab)} \n" +
    f"  without stp : {len(without_stp_vocab)} \n" +
    f"  after lemmatization : {len(lemmatized_vocab)}")
# -

# ## Most frequent words

# Here we check the most frequent words we got from our preprocessing operations, in order to make adjustments to our stop words file :

# +
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

most_frequents = get_frequencies(corpus_lemmatized).most_common(30)

corpus_common_counts = [word[1] for word in most_frequents]
corpus_common_words = [word[0] for word in most_frequents]

plt.style.use('dark_background')
plt.figure(figsize=(15, 15))


sns.barplot(x=corpus_common_counts,y=corpus_common_words)
plt.title('Most Common Tokens in our Corpus')

# -

# Some special cases, like numbers, acronyms or special caracters seem to have passed the initial stop words removal. let's try to find out which type of tokens we are dealing with : 

# +
freqs = get_frequencies(corpus_lemmatized)
small_freqs = [item for item in freqs.items() if len(item[0]) <= 4 and item[0].isalpha()]
most_frequents_small = sorted(small_freqs, key=lambda item: item[1], reverse=True)[:30]

corpus_common_counts = [word[1] for word in most_frequents_small]
corpus_common_words = [word[0] for word in most_frequents_small]

plt.style.use('dark_background')
plt.figure(figsize=(15, 15))


sns.barplot(x=corpus_common_counts,y=corpus_common_words)
plt.title('Most Common Tokens in our Corpus')
# -

# From this analysis we can suggest a removal of number, special characters and more stop words expected in web documents (such as extensions)

# ## Final results

# We apply these additional treatments and obtain the following results :

corpus_without_stp = preprocess.remove_stop_words_collection(corpus,"../data/stop_words_extended.txt")
corpus_filtered = preprocess.filter_collection(corpus_without_stp)
corpus_lemmatized = preprocess.lemmatize_collection(corpus_filtered)
final_corpus = preprocess.remove_stop_words_collection(corpus_lemmatized,"../data/stop_words_extended.txt")

# +
most_frequents = get_frequencies(final_corpus).most_common(50)

corpus_common_counts = [word[1] for word in most_frequents]
corpus_common_words = [word[0] for word in most_frequents]

plt.style.use('dark_background')
plt.figure(figsize=(15, 25))


sns.barplot(x=corpus_common_counts,y=corpus_common_words)
plt.title('Most Common Tokens in our Corpus')

# +
init_count = count_tokens(corpus)
without_stp_count = count_tokens(corpus_without_stp)
filtered_count = count_tokens(corpus_filtered)
final_count = count_tokens(final_corpus)

print("token count : \n" +
    f"  initial : {init_count} \n" +
    f"  without stp : {without_stp_count} \n" +
    f"  filtered : {filtered_count} \n" +
    f"  after lemmatization : {final_count}")

# +
init_vocab = get_vocabulary(corpus)
without_stp_vocab = get_vocabulary(corpus_without_stp)
filtered_vocab = get_vocabulary(corpus_filtered)
final_vocab = get_vocabulary(final_corpus)

print("vocabulary size : \n" +
    f"  initial : {len(init_vocab)} \n" +
    f"  without stp : {len(without_stp_vocab)} \n" +
    f"  filtered : {len(filtered_vocab)} \n" +
    f"  after lemmatization : {len(final_vocab)}")
# -

# ## Speed concerns

# In our first version of the program, we used Part-Of-Speach tagging to find the proper lemmatization for each token. However, this is really time consuming, as the lemmatization process is the slowest in our preprocessing chain. For faster preprocessing, we used the snowball stemmer alone which has remarkable results in itself. The results use that stemmer : 

corpus_without_stp = preprocess.remove_stop_words_collection(corpus,"../data/stop_words_extended.txt")
corpus_filtered = preprocess.filter_collection(corpus_without_stp)
corpus_lemmatized_2 = preprocess.lemmatize_collection(corpus_filtered, pos=False)
final_corpus_2 = preprocess.remove_stop_words_collection(corpus_lemmatized_2,"../data/stop_words_extended.txt")

# +
most_frequents = get_frequencies(final_corpus_2).most_common(50)

corpus_common_counts = [word[1] for word in most_frequents]
corpus_common_words = [word[0] for word in most_frequents]

plt.style.use('dark_background')
plt.figure(figsize=(15, 25))


sns.barplot(x=corpus_common_counts,y=corpus_common_words)
plt.title('Most Common Tokens in our Corpus')

# +
init_count = count_tokens(corpus)
without_stp_count = count_tokens(corpus_without_stp)
filtered_count = count_tokens(corpus_filtered)
final_count = count_tokens(final_corpus_2)

print("token count : \n" +
    f"  initial : {init_count} \n" +
    f"  without stp : {without_stp_count} \n" +
    f"  filtered : {filtered_count} \n" +
    f"  after lemmatization : {final_count}")

# +
init_vocab = get_vocabulary(corpus)
without_stp_vocab = get_vocabulary(corpus_without_stp)
filtered_vocab = get_vocabulary(corpus_filtered)
final_vocab = get_vocabulary(final_corpus_2)

print("vocabulary size : \n" +
    f"  initial : {len(init_vocab)} \n" +
    f"  without stp : {len(without_stp_vocab)} \n" +
    f"  filtered : {len(filtered_vocab)} \n" +
    f"  after lemmatization : {len(final_vocab)}")
# -


