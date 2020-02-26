import pandas as pd
import numpy as np
import logging, datetime

import document_similarity

def process_search_logs(path):
    '''Takes search logs files and performs some basic transformations.'''
    
    logging.info('STATUS: processing search logs')
    search = pd.read_csv(path).dropna()
    dtype = dict(PRODUCT_ID=int)
    search = search.astype(dtype)
    search["KEYWORD_SEARCHED"] = search["KEYWORD_SEARCHED"].str.lower()
    search["PRODUCT_NAME"] = search["PRODUCT_NAME"].str.lower()
    return search

def process_product_features(path):
    '''Takes the product_data file and apply correct dtypes, drops unused columns and removes some invalid values.'''

	product_id = #provide the name of the column of your product id

    logging.info('STATUS: processing product features')
    product_features = pd.read_csv(path, dtype=dict(PRODUCT_ID=object, HAS_IMAGE=object, IS_POPULAR=object, AGE_RESTRICTION=object, 
             CONTROLLED=object, ACTIVE=object, HAS_TOPPINGS=object, REQUIRES_MEDICAL_PRESCRIPTION=object, HAS_DISCOUNT=object))
    product_features = product_features.drop([#Provide what you need to drop
		], axis=1)
    product_features = product_features[product_features.product_id != "False"].drop_duplicates(subset=[#Avoid duplicates providing subset
		])
    product_features[#provide product name column
			] = product_features[#provide product name column
				].str.lower()
    
    product_features.dropna(inplace=True)
    dtype = dict(#Provide the data-types of your data
		)
    product_features = product_features.astype(dtype)
    return product_features

def generate_judgement_list(search_logs):
    '''Takes search logs and groups by search query, and performs aggregations of clicks to define relevance as CTR to a range 0-4 (4 meaning max relevance).'''
    logging.info('STATUS: generating judgement list')
    agg_search = search_logs.groupby([#Provide grouby condition
		]).agg(#Provide agg function
    			).reset_index()

    agg_search = agg_search[agg_search.RELEVANCE >= 50]

    agg_search["CTR"] = agg_search["CLICK_SUM"] / agg_search["IMPRESS"]
    ctr_by_search = agg_search.groupby(#Provide query column name
		).agg(
        MAX_CTR=pd.NamedAgg(column='CTR', aggfunc='max'),
    )

    agg_search = agg_search.merge(ctr_by_search, on=#Provide query column name
		)

    agg_search["RELEVANCE"] = (4 * (agg_search["CTR"] / agg_search["MAX_CTR"])).apply(np.ceil)
    judgement_list = agg_search.dropna().copy()
    judgement_list["QID"] = judgement_list.KEYWORD_SEARCHED.astype('category').cat.codes

    return judgement_list[[#Provide what you want to return"
			]]

def write_file_to_csv(features):
    logging.info('STATUS: writing features to csv')
    path = "features.csv"
    with open(path, 'w+') as outfile:
        features.to_csv(outfile, index=False)
    return path

if __name__ == "__main__":
    # Logging level
    logging.getLogger().setLevel(logging.INFO)

    # Load and process data
    search_logs = process_search_logs(#provide it)
    judgement_list = generate_judgement_list(search_logs)

    # Delete raw data
    del search_logs

    # Product Features
    product_features = process_product_features(#provide it)

    # Similarity features (TF-IDF, BM25)
    query_search = judgement_list[["KEYWORD_SEARCHED", "PRODUCT_NAME"]].drop_duplicates()
    query_product_name_sim = document_similarity.DocumentSimilarity(query_search.KEYWORD_SEARCHED, query_search.PRODUCT_NAME, #provide stopwords)
    
    logging.info('STATUS: computing similarity scores')
    query_search["TF_IDF_PRODUCTS"] = query_product_name_sim.computeTFIDF()
    query_search["BM25_PRODUCTS"] = query_product_name_sim.computeBM25()

    # Merge features
    logging.info('STATUS: merging features')
    full_features = judgement_list.merge(
            query_search, on=["KEYWORD_SEARCHED", "PRODUCT_NAME"]).merge(
            product_features, on=["PRODUCT_ID", "PRODUCT_NAME"], how="left")

    # Writing features to csv
    write_file_to_csv(full_features)
