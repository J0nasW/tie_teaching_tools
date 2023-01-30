# https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
# https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
# https://stackoverflow.com/questions/9343929/how-to-stem-words-in-python-list
# https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
# https://buhrmann.github.io/tfidf-analysis.html

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# from gensim.summarization import keywords
# from gensim.summarization.keywords import get_graph
import networkx as nx
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
import json
import os
import numpy as np
import pandas as pd

# Every Preprocessing Function can be found here:
from preprocessors import preprocessor_jw

def create_tf_idf_total(project_dir, output_name, filename_scopus, filename_arxiv, query):

    print("--- TERM FREQUENCY - INVERSE DOCUMENT FREQUENCY Scopus ---")
    ## load the data
    print("load data")
    with open(filename_scopus) as f:
      data_scopus = json.load(f)
    with open(filename_arxiv) as f:
      data_arxiv = json.load(f)

    data_in = data_scopus + data_arxiv

    tf_idf_name = os.path.join(project_dir, "tf_idf_" + output_name)

    ## convert list of dic into dataframe
    df = pd.DataFrame(data_in)

    list_of_titles = [title for title in df["title"]]
    list_of_abstracts = [abstract for abstract in df["abstract"]]

    ## PREPROCESSING:
    # Converting to lowercase, removing punctuation, removing stop words and lemmatization/stemming
    list_of_titles = preprocessor_jw.remove_other_things(list_of_titles)
    list_of_abstracts = preprocessor_jw.remove_other_things(list_of_abstracts)

    list_of_titles = preprocessor_jw.lowercase(list_of_titles)
    list_of_abstracts = preprocessor_jw.lowercase(list_of_abstracts)

    list_of_titles = preprocessor_jw.deleteNumbers(list_of_titles)
    list_of_abstracts = preprocessor_jw.deleteNumbers(list_of_abstracts)

    list_of_titles = preprocessor_jw.remove_punctuation(list_of_titles)
    list_of_abstracts = preprocessor_jw.remove_punctuation(list_of_abstracts)

    list_of_titles = preprocessor_jw.remove_stopwords(list_of_titles)
    list_of_abstracts = preprocessor_jw.remove_stopwords(list_of_abstracts)

    special_words = {
        "blockchain": "block chain",
    } 
    list_of_titles = preprocessor_jw.query_filter(list_of_titles, query, special_words)
    list_of_abstracts = preprocessor_jw.query_filter(list_of_abstracts, query, special_words) 

    list_of_titles = preprocessor_jw.remove_single_chars(list_of_titles)
    list_of_abstracts = preprocessor_jw.remove_single_chars(list_of_abstracts)

    list_of_titles = preprocessor_jw.lemmatize_words(list_of_titles)
    list_of_abstracts = preprocessor_jw.lemmatize_words(list_of_abstracts)

    ## Filter bigrams
    list_of_titles = list(nltk.bigrams(list_of_titles))
    list_of_abstracts = list(nltk.bigrams(list_of_abstracts))

    #list_of_titles = preprocessor_jw.stem_words(list_of_titles)
    #list_of_abstracts = preprocessor_jw.stem_words(list_of_abstracts)

    ## TF IDF

    # TF-IDF Formula:
    # document = body + title
    # TF-IDF(document) = TF-IDF(title) * alpha + TF-IDF(body) * (1-alpha)

    # tfidf = TfidfVectorizer()
    # tfidf_titles = tfidf.fit_transform(list_of_titles)
    # tfidf_abstracts = tfidf.fit_transform(list_of_abstracts)
    # feature_names = tfidf.get_feature_names()
    # for col in tfidf_titles.nonzero()[1]:
    #     print (feature_names[col], ' - ', tfidf_titles[0, col])

    # https://towardsdatascience.com/tf-idf-explained-and-python-sklearn-implementation-b020c5e83275


    # TF-IDF for titles
    string_of_titles = [' '.join(map(str, list_of_titles))]
    tfIdfTransformer = TfidfTransformer(use_idf=True)
    countVectorizer = CountVectorizer()
    wordCount = countVectorizer.fit_transform(string_of_titles)
    newTfIdf = tfIdfTransformer.fit_transform(wordCount)
    df = pd.DataFrame(newTfIdf[0].T.todense(), index=countVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    df.to_csv(tf_idf_name + "_keywords_titles.csv")
    print (df.head(25))

    # Keyword Plots
    # df.head(25).plot(kind = 'bar', color=(0, 0.24, 0.5, 1), title="TF-IDF: Title Keywords")
    title_dict = dict( df.head(30).iloc[:, -1] )
    plot_keywords(title_dict, "TF-IDF: Title Keywords")
    plt.savefig(tf_idf_name + "_keywords_title.png")
    plt.savefig(tf_idf_name + "_keywords_title.svg")

    # TF-IDF for abstracts
    string_of_abstracts = [' '.join(map(str, list_of_abstracts))]
    tfIdfTransformer = TfidfTransformer(use_idf=True)
    countVectorizer = CountVectorizer()
    wordCount = countVectorizer.fit_transform(string_of_abstracts)
    newTfIdf = tfIdfTransformer.fit_transform(wordCount)
    df = pd.DataFrame(newTfIdf[0].T.todense(), index=countVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    df.to_csv(tf_idf_name + "_keywords_abstracts.csv")
    print (df.head(25))

    # Keyword Plots
    # df.head(25).plot(kind = 'bar', color=(0, 0.24, 0.5, 1), title="TF-IDF: Abstract Keywords")
    abstract_dict = dict( df.head(30).iloc[:, -1] )

    plot_keywords(abstract_dict, "TF-IDF: Abstract Keywords")
    plt.savefig(tf_idf_name + "_keywords_abstracts.png")
    plt.savefig(tf_idf_name + "_keywords_abstracts.svg")

    print("show plots")
    plt.show()
    print("FINISHED")
    print()

    # Graph of titles
    # string_of_titles = ' '.join(map(str, list_of_titles))
    # displayGraph(get_graph(string_of_titles), tf_idf_name, "titles")

    # Graph of abstracts
    # string_of_abstracts = ' '.join(map(str, list_of_abstracts))
    # displayGraph(get_graph(string_of_abstracts), tf_idf_name, "abstracts")

def plot_keywords(data, title):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    keys = list(data.keys())
    values = list(data.values())
    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, align='center', color=(0, 0.24, 0.5, 1))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('importance')
    ax.set_title(title)
    plt.tight_layout()
    with open("./data/work/FINAL_CRAWL/output/tfidf_" + title + '_csvfile.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, data.keys())
        w.writeheader()
        w.writerow(data)

def plot_keywords_orig(list, title):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    keys = list.keys
    values = list.values
    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, align='center', color=(0, 0.24, 0.5, 1))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('importance')
    ax.set_title(title)
    plt.tight_layout()

def displayGraph(textGraph, tf_idf_name, title):
    graph = nx.Graph()
    for edge in textGraph.edges():
        graph.add_node(edge[0])
        graph.add_node(edge[1])
        graph.add_weighted_edges_from([(edge[0], edge[1], textGraph.edge_weight(edge))])
        textGraph.edge_weight(edge)
    pos = nx.spring_layout(graph)
    plt.figure()
    nx.draw(graph, pos, edge_color='grey', width=0.5, linewidths=0.5,
            node_size=400, node_color='seagreen', alpha=0.7,
            labels={node: node for node in graph.nodes()})
    plt.axis('off')
    plt.savefig(tf_idf_name + title + ".png")
    plt.show()

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def wordset(data):
    data = ' '.join(map(str, data)) # Remove [] if string is needed instead of list
    tokens = word_tokenize(data)
    wordset = list(dict.fromkeys(tokens))
    return wordset

def worddict(data):
    data_wordset = wordset(data)
    data_worddict = [0 for i in range(len(data))]
    for i in range(len(data_worddict)):
        data_worddict[i] = dict.fromkeys(data_wordset, 0)
        data_worddict[i] = Counter(data[i])
    return data_worddict

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict


    # Calculating DF first
    # df = calc_df(list_of_titles, list_of_abstracts)
    # print(df)

    # test = test_join(list_of_titles, list_of_abstracts)
    # print(test)

    # Calculating TF-IDF
    # tf_idf_titles = calc_tf_idf(list_of_titles, df_titles)
    # tf_idf_abstracts = calc_tf_idf(list_of_abstracts, df_abstracts)
    # print(tf_idf_titles)

    # text_tokens = calc_tf_test(list_of_titles)
    # print(text_tokens)

def calc_tf_test(data):

    data = [' '.join(map(str, data))] # Remove [] if string is needed instead of list
    # for i in range(len(data)):
    #     text_tokens = word_tokenize(data[i])
    return data


# def calc_tf(data):
#     TF = {}
#     N = len(data)

#     for i in range(len(data)):
#         ' '.join(document for document in data[i])

#     for i in range(len(data)):
#         text_tokens = word_tokenize(data[i])


#         for w in tokens:
#             try:
#                 DF[w].add(i)
#             except:
#                 DF[w] = {i}
#     return DF

def calc_df(title, abstract):
    DF = {}
    title = [' '.join(map(str, title))] # Remove [] if string is needed instead of list
    abstract = [' '.join(map(str, abstract))] # Remove [] if string is needed instead of list
    document = title + abstract
    document = ' '.join(map(str, document)) # Remove [] if string is needed instead of list

    tokens = word_tokenize(document)
    for i in range(len(tokens)):
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

    for i in DF:
        DF[i] = len(DF[i])
    return DF

def test_join(title, abstract):
    title = [' '.join(map(str, title))] # Remove [] if string is needed instead of list
    abstract = [' '.join(map(str, abstract))] # Remove [] if string is needed instead of list
    document = title + abstract
    document = ' '.join(map(str, document)) # Remove [] if string is needed instead of list
    return len(word_tokenize(document))


def calc_tf_idf(data, DF):
    doc = 0
    tf_idf = {}
    for i in range(N):
        tokens = data[i]
        counter = Counter(tokens + data[i])
        words_count = len(tokens + data[i])
        for token in np.unique(tokens):
            tf = counter[token]/words_count
            df = doc_freq(token, DF)
            idf = np.log((N+1)/(df+1))
            tf_idf[doc, token] = tf*idf
        doc += 1
    return tf_idf

def doc_freq(word, DF):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c