# ltr-api
Implementation of learning to rank model using Extra Tree Regressor on python. Included Snowflake connector to data gathering. API with flask.

Please include a settings.py file with your paths to data, and set your env variables for Snowflake connection.
Calculations of Tf_IDF and BM25 included. 
model_server.py -> API with flask and pkl with your model.
Provide your queries for search logs and product data in data/queries
