#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd
import snowflake.connector

# from snowflake.sqlalchemy import URL
# from sqlalchemy import create_engine
# import seaborn as sns
# os.path.dirname(sys.executable)

# CONSTANTS
# database
warehouse = ''
database = ''
authenticator = ''

# queries
queries_path = './queries/'
queries_extension = '.sql'
# data
data_path = './data/'
raw_path = 'raw/'
data_extension = '.csv'


class SnowflakeDatabase:

    def __init__(self, database, user, password, account, authenticator):
        self.user = user
        self.password = password
        self.database = database
        self.is_connected = False
        self.account = account
        self.authenticator = authenticator

    def connect(self):
        if self.user is None:
            print('You have to set the environment variables: source app-env-file')
            exit()

        self.con = snowflake.connector.connect(user=self.user,
                                               password=self.password,
                                               account=self.account,
                                               database=self.database,
                                               authenticator=self.authenticator
                                               )
        self.is_connected = True
        print('STATUS: connection created')

    def disconnect(self):
        if self.is_connected:
            self.con.close()
            print('STATUS: connection closed')

    def get_string_query(self, query_fn, **kwargs):
        data = ''
        with open(query_fn, 'r') as file:
            data = file.read().format(**kwargs)
        return data

    def execute_query(self, query_fn, with_cursor=False, **kwargs):
        '''Executes queries from sql files in the open connection'''
        cursor = self.con.cursor()
        try:
            query = self.get_string_query(query_fn, **kwargs)
            cursor.execute(query)
            res = cursor.fetchall()
            if with_cursor:
                return (res, cursor)
            else:
                return res
        finally:
            cursor.close()

    def get_df_from_query(self, query_fn, **kwargs):
        '''Transforms queries results to pandas dataframes'''
        result, cursor = self.execute_query(query_fn, with_cursor=True, **kwargs)
        print('STATUS: query executed')

        headers = list(map(lambda t: t[0], cursor.description))
        df = pd.DataFrame(result)
        df.columns = headers
        return df


if __name__ == "__main__":

    if (len(sys.argv) < 2):
        print('There has to be at least one sql file to query')
        exit()

    # create snowflake database object
    user = os.getenv('SNOWFLAKE_USER')
    password = os.getenv('SNOWFLAKE_PASSWORD')
    account = os.getenv('SNOWFLAKE_ACCOUNT')
    snow_db = SnowflakeDatabase(database, user, password, account, authenticator)

    # start connection
    snow_db.connect()

    # loop through queries
    for query_fn in sys.argv[1:]:
        print('STATUS: processing sql file:', query_fn)

        query_full = data_path + queries_path + query_fn + queries_extension
        df_query = snow_db.get_df_from_query(query_full)

        data_full = data_path + raw_path + query_fn + data_extension
        df_query.to_csv(data_full, sep=',')

        print('STATUS: data saved')

    # close connection
    snow_db.disconnect()
