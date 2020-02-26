import logging
import joblib
import traceback
import pandas as pd
import numpy as np
from time import time
import settings

from BM25 import BM25Transformer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

def get_stop_words(stop_file_path):
        """load stop words """

        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            return frozenset(stop_set)

def text_features(df, path_stopwords):
    try:
        logging.info('Creating Text Features')
        stopwords = get_stop_words(path_stopwords)
        # Prodcut_name
        corpus_product_name = df.PRODUCT_NAME.values
        # Product_name
        cv = CountVectorizer(max_df=0.85, stop_words=stopwords)
        count_vector_product_names = cv.fit_transform(corpus_product_name)

        tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(count_vector_product_names)

        # print idf values
        df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])

        # sort ascending
        df_idf.sort_values(by=['idf_weights'])

        count_vector_keyword_searched = cv.transform(df.KEYWORD_SEARCHED.values)
        tfidf_search_terms = tfidf_transformer.transform(count_vector_keyword_searched)
        tfidf_product_name = tfidf_transformer.transform(count_vector_product_names)
        tfidf_product_name

        df["TF_IDF_PRODUCTS"] = cosine_similarity(tfidf_search_terms, tfidf_product_name).diagonal()
        
        bm25_transformer = BM25Transformer(use_idf=True)
        bm25_transformer.fit(count_vector_product_names)
        bm25_search_terms = bm25_transformer.transform(count_vector_keyword_searched)
        bm25_product_name = bm25_transformer.transform(count_vector_product_names)
        df["BM25_PRODUCTS"] = cosine_similarity(bm25_search_terms, bm25_product_name).diagonal()
    
        return(df)
    
    except:
        logging.info('Failed to create text_features')
    

def load_data(path, sep = ','):
    logging.info('Loading data')
    try:
        data = pd.read_csv(path,sep=sep)
        return data

    except:
        logging.error(traceback.print_last())

        
def model_assets(model_path, columns_path, data):
    try:
        logging.info('Building Model assets')
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
        #data = load_data(data_eval_path)
        data = data
        qid = data['QID']
        
        prediction = list(model['pipeline'].predict(df[columns['model_columns']]))
        relevance_true = data['RELEVANCE']
        
        return prediction, relevance_true, qid, model['pipeline'], df[columns['model_columns']]
    
    except:
        logging.error('Failed to create model assets')


def update_features(df):
    try:
        #Provide a custom function to transform your data
        
        return(agg_search)
    
    except:
        logging.error('Failed to create needed features')


def dcg(relevances, rank=10):
    """Discounted cumulative gain at rank (DCG)"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)
 
def ndcg(relevances, rank=10):
    """Normalized discounted cumulative gain (NDGC)"""
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg(relevances, rank) / best_dcg

def mean_ndcg(y_true, y_pred, query_ids, rank=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    query_ids = np.asarray(query_ids)
    # assume query_ids are sorted
    ndcg_scores = []
    previous_qid = query_ids[0]
    previous_loc = 0
    for loc, qid in enumerate(query_ids):
        if previous_qid != qid:
            chunk = slice(previous_loc, loc)
            ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
            ndcg_scores.append(ndcg(ranked_relevances, rank=rank))
            previous_loc = loc
        previous_qid = qid

    chunk = slice(previous_loc, loc + 1)
    ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
    ndcg_scores.append(ndcg(ranked_relevances, rank=rank))
    return np.mean(ndcg_scores)

def print_evaluation(model, X, y, qid):
    try:
        logging.info('Measuring the performance')
        tic = time()
        y_predicted = model.predict(X)
        prediction_time = time() - tic
        print("Prediction time: {:.3f}s".format(prediction_time))
        print("NDCG@5 score: {:.3f}".format(
        mean_ndcg(y, y_predicted, qid, rank=5)))
        print("NDCG@10 score: {:.3f}".format(
        mean_ndcg(y, y_predicted, qid, rank=10)))
        print("NDCG score: {:.3f}".format(
        mean_ndcg(y, y_predicted, qid, rank=None)))
        print("R2 score: {:.3f}".format(r2_score(y, y_predicted)))
    
    except:
        logging.error('Failed to measure performance')
    
def performance_hist(y_true, y_pred):
    try:
        logging.info('Plotting performance histogram')
        plt.hist(y_true, bins=5, alpha=.3, color='b', label='True relevance')
        plt.hist(y_pred, bins=5, alpha=.3, color='g', label='Predicted relevance')
        plt.legend(loc='best')

    except:
        logging.error('Failed to plot performance histogram')
    
def performance_scat(y_true, y_pred):
    try:
        logging.info('Plotting performance scatter')
        plt.title('Extra Trees Regressor predictions')
        plt.scatter(y_true, y_pred, alpha=0.3, s=100)
        plt.xlabel('True relevance')
        plt.ylabel('Predicted relevance')
        plt.ylim(-2, 5)
        plt.xlim(-2, 5)

    except:
        logging.error('Failed to plot performance scatter')

if __name__ == "__main__":

    eval_data_path = settings.EVAL_DATA_PATH
    products = load_data(eval_data_path)
    df = update_features(products)
    full = text_features(df, '../data/processed/spanish_stopwords.txt')
    y_pred, y_true, qid, model, X = model_assets('../models/ar/model.pkl', 
                                '../models/ar/model_columns.pkl', full)
    print_evaluation(model, X, y_true, qid)
    print('Mean ndcg: ', mean_ndcg(y_true, y_pred, qid))
    performance_hist(y_true, y_pred)
