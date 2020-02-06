#!/usr/bin/env python
# coding: utf-8

# # Building a Sentiment Classifier using Scikit-Learn

# <center><img src="https://raw.githubusercontent.com/lazuxd/simple-imdb-sentiment-analysis/master/smiley.jpg"/></center>
# <center><i>Image by AbsolutVision @ <a href="https://pixabay.com/ro/photos/smiley-emoticon-furie-sup%C4%83rat-2979107/">pixabay.com</a></i></center>
# 
# > &nbsp;&nbsp;&nbsp;&nbsp;**Sentiment analysis**, an important area in Natural Language Processing, is the process of automatically detecting affective states of text. Sentiment analysis is widely applied to voice-of-customer materials such as product reviews in online shopping websites like Amazon, movie reviews or social media. It can be just a basic task of classifying the polarity of a text as being positive/negative or it can go beyond polarity, looking at emotional states such as "happy", "angry", etc.
# 
# &nbsp;&nbsp;&nbsp;&nbsp;Here we will build a classifier that is able to distinguish movie reviews as being either positive or negative. For that, we will use [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)<sup>(2)</sup> of IMDB movie reviews.
# This dataset contains 50,000 movie reviews divided evenly into 25k train and 25k test. The labels are balanced between the two classes (positive and negative). Reviews with a score <= 4 out of 10 are labeled negative and those with score >= 7 out of 10 are labeled positive. Neutral reviews are not included in the labeled data. This dataset also contains unlabeled reviews for unsupervised learning; we will not use them here. There are no more than 30 reviews for a particular movie because the ratings of the same movie tend to be correlated. All reviews for a given movie are either in train or test set but not in both, in order to avoid test accuracy gain by memorizing movie-specific terms.
# 
# 

# ## Downloading & extracting data

# In[1]:


get_ipython().system('wget "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"')


# In[2]:


get_ipython().system('tar -xzf "aclImdb_v1.tar.gz"')


# ## Data preprocessing

# &nbsp;&nbsp;&nbsp;&nbsp;After the dataset has been downloaded and extracted from archive we have to transform it into a more suitable form for feeding it into a machine learning model for training. We will start by combining all review data into 2 pandas Data Frames representing the train and test datasets, and then saving them as csv files: *imdb_train.csv* and *imdb_test.csv*.  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;The Data Frames will have the following form:  
# 
# |text       |label      |
# |:---------:|:---------:|
# |review1    |0          |
# |review2    |1          |
# |review3    |1          |
# |.......    |...        |
# |reviewN    |0          |  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;where:  
# - review1, review2, ... = the actual text of movie review  
# - 0 = negative review  
# - 1 = positive review

# &nbsp;&nbsp;&nbsp;&nbsp;But machine learnng algorithms work only with numerical values. We can't just input the text itself into a machine learning model and have it learn from that. We have to, somehow, represent the text by numbers or vectors of numbers. One way of doing this is by using the **Bag-of-words** model<sup>(3)</sup>, in which a piece of text(often called a **document**) is represented by a vector of the counts of words from a vocabulary in that document. This model doesn't take into account grammar rules or word ordering; all it considers is the frequency of words. If we use the counts of each word independently we name this representation a **unigram**. In general, in a **n-gram** we take into account the counts of each combination of n words from the vocabulary that appears in a given document.  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;For example, consider these two documents:  
# <br>  
# <div style="font-family: monospace;"><center><b>d1: "I am learning"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b></center></div>  
# <div style="font-family: monospace;"><center><b>d2: "Machine learning is cool"</b></center></div>  
# <br>
# The vocabulary of all words encountered in these two sentences is: 
# 
# <br/>  
# <div style="font-family: monospace;"><center><b>v: [ I, am, learning, machine, is, cool ]</b></center></div>   
# <br>
# &nbsp;&nbsp;&nbsp;&nbsp;The unigram representations of d1 and d2:  
# <br>  
# 
# |unigram(d1)|I       |am      |learning|machine |is      |cool    |
# |:---------:|:------:|:------:|:------:|:------:|:------:|:------:|
# |           |1       |1       |1       |0       |0       |0       |  
# 
# |unigram(d2)|I       |am      |learning|machine |is      |cool    |
# |:---------:|:------:|:------:|:------:|:------:|:------:|:------:|
# |           |0       |0       |1       |1       |1       |1       |
#   
# &nbsp;&nbsp;&nbsp;&nbsp;And, the bigrams of d1 and d2 are:
#   
# |bigram(d1) |I I     |I am    |I learning|...|machine am|machine learning|...|cool is|cool cool|
# |:---------:|:------:|:------:|:--------:|:-:|:--------:|:--------------:|:-:|:-----:|:-------:|
# |           |0       |1       |0         |...|0         |0               |...|0      |0        |  
# 
# |bigram(d2) |I I     |I am    |I learning|...|machine am|machine learning|...|cool is|cool cool|
# |:---------:|:------:|:------:|:--------:|:-:|:--------:|:--------------:|:-:|:-----:|:-------:|
# |           |0       |0       |0         |...|0         |1               |...|0      |0        |

# &nbsp;&nbsp;&nbsp;&nbsp;Often, we can achieve slightly better results if instead of counts of words we use something called **term frequency times inverse document frequency** (or **tf-idf**). Maybe it sounds complicated, but it is not. Bear with me, I will explain this. The intuition behind this is the following. So, what's the problem of using just the frequency of terms inside a document? Although some terms may have a high frequency inside documents they may not be so relevant for describing a given document in which they appear. That's because those terms may also have a high frequency across the collection of all documents. For example, a collection of movie reviews may have terms specific to movies/cinematography that are present in almost all documents(they have a high **document frequency**). So, when we encounter those terms in a document this doesn't tell much about whether it is a positive or negative review. We need a way of relating **term frequency** (how frequent a term is inside a document) to **document frequency** (how frequent a term is across the whole collection of documents). That is:  
#   
# $$\frac{term frequency}{document frequency} = term frequency \cdot \frac{1}{document frequency} = term frequency \cdot inverse document frequency = tf \cdot idf$$  
#   
# &nbsp;&nbsp;&nbsp;&nbsp;Now, there are more ways used to describe both term frequency and inverse document frequency. But the most common way is by putting them on a logarithmic scale:  
#   
# $$tf(t, d) = log(1+f_{t,d})$$  
# $$idf(t) = log(\frac{1+N}{1+n_t})$$  
#   
# &nbsp;&nbsp;&nbsp;&nbsp;where:  
# $f_{t,d}$ = count of term **t** in document **d**  
# N = total number of documents  
# $n_t$ = number of documents that contain term **t**  
#   
# &nbsp;&nbsp;&nbsp;&nbsp;We added 1 in the first logarithm to avoid getting $-\infty$ when $f_{t,d}$ is 0. In the second logarithm we added one fake document to avoid division by zero.

# &nbsp;&nbsp;&nbsp;&nbsp;Before we transform our data into vectors of counts or tf-idf values we should remove English **stopwords**<sup>(6)(7)</sup>. Stopwords are words that are very common in a language and are usually removed in the preprocessing stage of natural text-related tasks like sentiment analysis or search.

# &nbsp;&nbsp;&nbsp;&nbsp;Note that we should construct our vocabulary only based on the training set. When we will process the test data in order to make predictions we should use only the vocabulary constructed in the training phase, the rest of the words will be ignored.

# &nbsp;&nbsp;&nbsp;&nbsp;Now, let's create the data frames and save them as csv files:

# In[3]:


import pandas as pd
import re
from os import listdir
from os.path import isfile, join
from random import shuffle


# In[4]:


def create_data_frame(folder: str) -> pd.DataFrame:
    '''
    folder - the root folder of train or test dataset
    Returns: a DataFrame with the combined data from the input folder
    '''
    pos_folder = f'{folder}/pos' # positive reviews
    neg_folder = f'{folder}/neg' # negative reviews
    
    def get_files(fld: str) -> list:
        '''
        fld - positive or negative reviews folder
        Returns: a list with all files in input folder
        '''
        return [join(fld, f) for f in listdir(fld) if isfile(join(fld, f))]
    
    def append_files_data(data_list: list, files: list, label: int) -> None:
        '''
        Appends to 'data_list' tuples of form (file content, label)
        for each file in 'files' input list
        '''
        for file_path in files:
            with open(file_path, 'r') as f:
                text = f.read()
                data_list.append((text, label))
    
    pos_files = get_files(pos_folder)
    neg_files = get_files(neg_folder)
    
    data_list = []
    append_files_data(data_list, pos_files, 1)
    append_files_data(data_list, neg_files, 0)
    shuffle(data_list)
    
    text, label = tuple(zip(*data_list))
    # replacing line breaks with spaces
    text = list(map(lambda txt: re.sub('(<br\s*/?>)+', ' ', txt), text))
    
    return pd.DataFrame({'text': text, 'label': label})


# In[5]:


imdb_train = create_data_frame('aclImdb/train')
imdb_test = create_data_frame('aclImdb/test')

get_ipython().system("mkdir 'csv'")
imdb_train.to_csv('csv/imdb_train.csv', index=False)
imdb_test.to_csv('csv/imdb_test.csv', index=False)

# imdb_train = pd.read_csv('csv/imdb_train.csv')
# imdb_test = pd.read_csv('csv/imdb_test.csv')


# ### Text vectorization

# &nbsp;&nbsp;&nbsp;&nbsp;Fortunately, for the text vectorization part all the hard work is already done in the Scikit-Learn classes `CountVectorizer`<sup>(8)</sup> and `TfidfTransformer`<sup>(5)</sup>. We will use these classes to transform our csv files into unigram and bigram matrices(using both counts and tf-idf values). (It turns out that if we only use a n-gram for a large n we don't get a good accuracy, we usually use all n-grams up to some n. So, when we say here bigrams we actually refer to uni+bigrams and when we say unigrams it's just unigrams.) Each row in those matrices will represent a document (review) in our dataset, and each column will represent values associated with each word in the vocabulary (in the case of unigrams) or values associated with each combination of maximum 2 words in the vocabulary (bigrams).  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;`CountVectorizer` has a parameter `ngram_range` which expects a tuple of size 2 that controls what n-grams to include. After we constructed a `CountVectorizer` object we should call .fit() method with the actual text as a parameter, in order for it to learn the required statistics of our collection of documents. Then, by calling .transform() method with our collection of documents it returns the matrix for the n-gram range specified. As the class name suggests, this matrix will contain just the counts. To obtain the tf-idf values, the class `TfidfTransformer` should be used. It has the .fit() and .transform() methods that are used in a similar way with those of `CountVectorizer`, but they take as input the counts matrix obtained in the previous step and .transform() will return a matrix with tf-idf values. We should use .fit() only on training data and then store these objects. When we want to evaluate the test score or whenever we want to make a prediction we should use these objects to transform the data before feeding it into our classifier.  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;Note that the matrices generated for our train or test data will be huge, and if we store them as normal numpy arrays they will not even fit into RAM. But most of the entries in these matrices will be zero. So, these Scikit-Learn classes are using Scipy sparse matrices<sup>(9)</sup> (`csr_matrix`<sup>(10)</sup> to be more exactly), which store just the non-zero entries and save a LOT of space.  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;We will use a linear classifier with stochastic gradient descent, `sklearn.linear_model.SGDClassifier`<sup>(11)</sup>, as our model. First we will generate and save our data in 4 forms: unigram and bigram matrix (with both counts and tf-idf values for each). Then we will train and evaluate our model for each these 4 data representations using `SGDClassifier` with the default parameters. After that, we choose the data representation which led to the best score and we will tune the hyper-parameters of our model with this data form using cross-validation in order to obtain the best results.

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load # used for saving and loading sklearn objects
from scipy.sparse import save_npz, load_npz # used for saving and loading sparse matrices


# In[7]:


get_ipython().system("mkdir 'data_preprocessors'")
get_ipython().system("mkdir 'vectorized_data'")


# #### Unigram Counts

# In[8]:


unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_vectorizer.fit(imdb_train['text'].values)

dump(unigram_vectorizer, 'data_preprocessors/unigram_vectorizer.joblib')

# unigram_vectorizer = load('data_preprocessors/unigram_vectorizer.joblib')


# In[9]:


X_train_unigram = unigram_vectorizer.transform(imdb_train['text'].values)

save_npz('vectorized_data/X_train_unigram.npz', X_train_unigram)

# X_train_unigram = load_npz('vectorized_data/X_train_unigram.npz')


# #### Unigram Tf-Idf

# In[10]:


unigram_tf_idf_transformer = TfidfTransformer()
unigram_tf_idf_transformer.fit(X_train_unigram)

dump(unigram_tf_idf_transformer, 'data_preprocessors/unigram_tf_idf_transformer.joblib')

# unigram_tf_idf_transformer = load('data_preprocessors/unigram_tf_idf_transformer.joblib')


# In[11]:


X_train_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_train_unigram)

save_npz('vectorized_data/X_train_unigram_tf_idf.npz', X_train_unigram_tf_idf)

# X_train_unigram_tf_idf = load_npz('vectorized_data/X_train_unigram_tf_idf.npz')


# #### Bigram Counts

# In[12]:


bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_vectorizer.fit(imdb_train['text'].values)

dump(bigram_vectorizer, 'data_preprocessors/bigram_vectorizer.joblib')

# bigram_vectorizer = load('data_preprocessors/bigram_vectorizer.joblib')


# In[13]:


X_train_bigram = bigram_vectorizer.transform(imdb_train['text'].values)

save_npz('vectorized_data/X_train_bigram.npz', X_train_bigram)

# X_train_bigram = load_npz('vectorized_data/X_train_bigram.npz')


# #### Bigram Tf-Idf

# In[14]:


bigram_tf_idf_transformer = TfidfTransformer()
bigram_tf_idf_transformer.fit(X_train_bigram)

dump(bigram_tf_idf_transformer, 'data_preprocessors/bigram_tf_idf_transformer.joblib')

# bigram_tf_idf_transformer = load('data_preprocessors/bigram_tf_idf_transformer.joblib')


# In[15]:


X_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_train_bigram)

save_npz('vectorized_data/X_train_bigram_tf_idf.npz', X_train_bigram_tf_idf)

# X_train_bigram_tf_idf = load_npz('vectorized_data/X_train_bigram_tf_idf.npz')


# ### Choosing data format

# &nbsp;&nbsp;&nbsp;&nbsp;Now, for each data form we split it into train & validation sets, train a `SGDClassifier` and output the score.

# In[16]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np


# In[17]:


def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')


# In[18]:


y_train = imdb_train['label'].values


# In[19]:


train_and_show_scores(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')


# &nbsp;&nbsp;&nbsp;&nbsp;The best data form seems to be **bigram with tf-idf** as it gets the highest validation accuracy: **0.9**; we will use it next for hyper-parameter tuning.

# ### Using Cross-Validation for hyperparameter tuning

# &nbsp;&nbsp;&nbsp;&nbsp;For this part we will use `RandomizedSearchCV`<sup>(12)</sup> which chooses the parameters randomly from the list that we give, or according to the distribution that we specify from `scipy.stats` (e.g. uniform); then is estimates the test error by doing cross-validation and after all iterations we can find the best estimator, the best parameters and the best score in the variables `best_estimator_`, `best_params_` and `best_score_`.  
# 
# &nbsp;&nbsp;&nbsp;&nbsp;Because the search space for the parameters that we want to test is very big and it may need a huge number of iterations until it finds the best combination, we will split the set of parameters in 2 and do the hyper-parameter tuning process in two phases. First we will find the optimal combination of loss, learning_rate and eta0 (i.e. initial learning rate); and then for penalty and alpha.

# In[20]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# In[21]:


X_train = X_train_bigram_tf_idf


# #### Phase 1: loss, learning rate and initial learning rate

# In[22]:


clf = SGDClassifier()


# In[23]:


distributions = dict(
    loss=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    learning_rate=['optimal', 'invscaling', 'adaptive'],
    eta0=uniform(loc=1e-7, scale=1e-2)
)


# In[24]:


random_search_cv = RandomizedSearchCV(
    estimator=clf,
    param_distributions=distributions,
    cv=5,
    n_iter=50
)
random_search_cv.fit(X_train, y_train)
print(f'Best params: {random_search_cv.best_params_}')
print(f'Best score: {random_search_cv.best_score_}')


# &nbsp;&nbsp;&nbsp;&nbsp;Because we got "learning_rate = optimal" to be the best, then we will ignore the eta0 (initial learning rate) as it isn't used when learning_rate='optimal'; we got this value of eta0 just because of the randomness involved in the process.

# #### Phase 2: penalty and alpha

# In[25]:


clf = SGDClassifier()


# In[26]:


distributions = dict(
    penalty=['l1', 'l2', 'elasticnet'],
    alpha=uniform(loc=1e-6, scale=1e-4)
)


# In[27]:


random_search_cv = RandomizedSearchCV(
    estimator=clf,
    param_distributions=distributions,
    cv=5,
    n_iter=50
)
random_search_cv.fit(X_train, y_train)
print(f'Best params: {random_search_cv.best_params_}')
print(f'Best score: {random_search_cv.best_score_}')


# &nbsp;&nbsp;&nbsp;&nbsp;So, the best parameters that I got are:  
# `loss: squared_hinge  
#  learning_rate: optimal  
#  penalty: l2  
#  alpha: 1.2101013664295101e-05  `

# #### Saving the best classifier

# In[28]:


get_ipython().system("mkdir 'classifiers'")


# In[29]:


sgd_classifier = random_search_cv.best_estimator_

dump(random_search_cv.best_estimator_, 'classifiers/sgd_classifier.joblib')

# sgd_classifier = load('classifiers/sgd_classifier.joblib')


# ### Testing model

# In[30]:


X_test = bigram_vectorizer.transform(imdb_test['text'].values)
X_test = bigram_tf_idf_transformer.transform(X_test)
y_test = imdb_test['label'].values


# In[31]:


score = sgd_classifier.score(X_test, y_test)
print(score)


# &nbsp;&nbsp;&nbsp;&nbsp;And we got **90.18%** test accuracy. That's not bad for our simple linear model. There are more advanced methods that give better results. The current state-of-the-art on this dataset is **97.42%** <sup>(13)</sup>

# ## References
# 
# <sup>(1)</sup> &nbsp;[Sentiment Analysis - Wikipedia](https://en.wikipedia.org/wiki/Sentiment_analysis)  
# <sup>(2)</sup> &nbsp;[Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)  
# <sup>(3)</sup> &nbsp;[Bag-of-words model - Wikipedia](https://en.wikipedia.org/wiki/Bag-of-words_model)  
# <sup>(4)</sup> &nbsp;[Tf-idf - Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)  
# <sup>(5)</sup> &nbsp;[TfidfTransformer - Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)  
# <sup>(6)</sup> &nbsp;[Stop words - Wikipedia](https://en.wikipedia.org/wiki/Stop_words)  
# <sup>(7)</sup> &nbsp;[A list of English stopwords](https://gist.github.com/sebleier/554280)  
# <sup>(8)</sup> &nbsp;[CountVectorizer - Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)  
# <sup>(9)</sup> &nbsp;[Scipy sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html)  
# <sup>(10)</sup> [Compressed Sparse Row matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)  
# <sup>(11)</sup> [SGDClassifier - Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)  
# <sup>(12)</sup> [RandomizedSearchCV - Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)  
# <sup>(13)</sup> [Sentiment Classification using Document Embeddings trained with
# Cosine Similarity](https://www.aclweb.org/anthology/P19-2057.pdf)  
