from BM25 import BM25Transformer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentSimilarity:

    def __init__(self, query_col, document_col, stop_file_path):
        self.query_col = query_col
        self.document_col = document_col
        self.stop_file_path = stop_file_path
        
        self.stopwords = self.__get_stop_words()
        self.count_vector = self.__compute_count_vector(document_col)
        self.count_vector_document_col = self.count_vector.transform(document_col)
        self.count_vector_query_col = self.count_vector.transform(query_col)

    def computeBM25(self):
        bm25_transformer = BM25Transformer(use_idf=True)
        bm25_transformer.fit(self.count_vector_document_col)
        bm25_search_terms = bm25_transformer.transform(self.count_vector_query_col)
        bm25_product_name = bm25_transformer.transform(self.count_vector_document_col)
        bm25 = cosine_similarity(bm25_search_terms, bm25_product_name).diagonal()
        return bm25

    def computeTFIDF(self):
        tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(self.count_vector_document_col)
        tfidf_search_terms = tfidf_transformer.transform(self.count_vector_query_col)
        tfidf_product_name = tfidf_transformer.transform(self.count_vector_document_col)
        tfidf = cosine_similarity(tfidf_search_terms, tfidf_product_name).diagonal()
        return tfidf

    def __get_stop_words(self):
        """load stop words """
        with open(self.stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            return frozenset(stop_set)

    def __compute_count_vector(self, col):
        corpus = col.values
        cv = CountVectorizer(stop_words=self.stopwords)
        cv.fit(corpus)
        return cv

    
