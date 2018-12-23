import gensim
from gensim import corpora
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
# from sklearn.metrics import pairwise


def load_data():
    """
    load the 100 sentence from neg.xls
    :return:
    """

    data = pd.read_excel("neg.xls", header=None)
    data[0] = data[0].apply(lambda x: jieba.lcut(x))
    docs = [doclist for doclist in data[0][:100]]
    docs = [" ".join(doc) for doc in docs]
    return docs


def lsa(docs):
    """
    transform the 100 sentence to a 100 vectors of 10 dimension using lsass
    :param docs:
    :return:
    """

    vectorizer = TfidfVectorizer(docs, use_idf=True)
    svd_model = TruncatedSVD(n_components=10,
                             algorithm='randomized',
                             n_iter=10)
    svd_transformer = Pipeline([('tfidf', vectorizer),
                                ('svd', svd_model)])
    svd_matrix = svd_transformer.fit_transform(docs)
    return svd_matrix


if __name__ == "__main__":
    docs = load_data()
    svd_matrix = lsa(docs)
    # sim = pairwise.cosine_similarity(svd_matrix)
