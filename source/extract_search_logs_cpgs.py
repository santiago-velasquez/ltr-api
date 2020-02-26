import logging, datetime
import data_extraction
import settings

def get_search_logs():
    logging.info('STATUS: retrieving search logs')
    snowflake_database.connect()
    fp_search = settings.QUERIES_PATH + "local_search_cpgs.sql"
    n_searches = "" if settings.N_SEARCHES == 0 else "limit {}".format(settings.N_SEARCHES)
    df_search = snowflake_database.get_df_from_query(fp_search,
                                                        query_last_months=settings.QUERY_LAST_MONTHS,
                                                        country_code=settings.COUNTRY_CODE,
                                                        n_searches=n_searches)
    snowflake_database.disconnect()
    return df_search

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

    # Downloading search logs
    search_logs = get_search_logs()
    write_file_to_csv(search_logs, "search_logs")
