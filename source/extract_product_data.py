import logging, datetime
import data_extraction
import settings

def get_product_data():
    logging.info('STATUS: retrieving product data')
    snowflake_database.connect()
    fp_products = settings.QUERIES_PATH + "product_features.sql"
    df_products = snowflake_database.get_df_from_query(fp_products,
                                                        query_last_months=settings.QUERY_LAST_MONTHS,
                                                        country_code=settings.COUNTRY_CODE)
    snowflake_database.disconnect()
    return df_products

def write_file_to_csv(df, name):
    logging.info('STATUS: writing file to path')
    big_csv_fn = f'name.csv'
    path_file = settings.RAW_PATH + big_csv_fn
    with open(path_file, 'w+') as outfile:
        df.to_csv(outfile, index=False)
    return path_file

if __name__ == "__main__":
    # Logging level
    logging.getLogger().setLevel(logging.INFO)

    # Initialize DB connection
    logging.info('STATUS: creating snowflake database connector')
    snowflake_database = data_extraction.SnowflakeDatabase(database=settings.DATABASE, 
                                                           user=settings.USER, 
                                                           password=settings.PASSWORD, 
                                                           account=settings.ACCOUNT,
                                                           authenticator=settings.AUTHENTICATOR)

    # Downloading product data
    product_data = get_product_data()
    write_file_to_csv(product_data, "product_data")
