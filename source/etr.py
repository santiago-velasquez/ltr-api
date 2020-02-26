import pandas
import settings
import numpy
import logging
from time import time
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
import traceback
import joblib
import fnmatch
import os
import dill as pkl
pkl.settings['recurse'] = True

class Ltr:  

    def __init__(self):

        self.model = model
        self.stopwords = self.__get_stop_words()
        self.columns = self.__load_model_columns()
 

    def __input_to_df(self, json_data):
        data = pandas.read_json(json_data, orient='records', encoding='utf-8')
        return data

    def __get_query(self, df_json):

        df = pandas.DataFrame(df_json['products'][0])
        query = df_json['query']
        for i in range(df.shape[0]-1):
            query = query.append(df_json['query'],ignore_index=True)
        return query
    
    def __get_names(self, df_json):
        df = pandas.DataFrame(df_json['products'][0])
        print(df['name'])
        return df['name']

    
    def __computeTFIDF(self, cv_doc, cv_query):
        tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(cv_doc)
        tfidf_search_terms = tfidf_transformer.transform(cv_query)
        tfidf_product_name = tfidf_transformer.transform(cv_doc)
        tfidf = cosine_similarity(tfidf_search_terms, tfidf_product_name).diagonal()
        return tfidf

    def __get_stop_words(self):
        """load stop words """

        burn = {'paìs', 'esa', 'apenas', 'largo', 'unas', 'cuántos', 'gran', 'través', 'a', 'aseguró', 'dijeron', 'horas', 'estuvieses', 'ésa', 'ahí', 'ahi', 'cuantos', 'estará', 'estuve', 'medio', 'entre', 'raras', 'tampoco', 'cosas', 'tendríais', 'ésta', 'les', 'vez', 'conseguimos', 'lado', 'sean', 'claro', 'estos', 'hago', 'nuevas', 'siete', 'tuvieses', 'ciertas', 'contra', 'ultimo', 'aquéllos', 'igual', 'pudo', 'teneis', 'tendrás', 'han', 'tarde', 'bajo', 'habían', 'breve', 'actualmente', 'puedo', 'momento', 'usan', 'habréis', 'nuestro', 'somos', 'haces', 'éstos', 'queremos', 'nadie', 'sabes', 'estuvisteis', 'quizás', 'consigue', 'estuviste', 'vuestras', 'intentamos', 'creo', 'sois', 'qeu', 'trabajas', 'estar', 'segundo', 'cuál', 'sigue', 'conocer', 'hacia', 'estuviera', 'conseguir', 'habíamos', 'considera', 'habida', 'yo', 'nuestras', 'i', 'ésos', 'estuviéramos', 'j', 'tengan', 'tuvisteis', 'tuyo', 'podriamos', 'quien', 'intenta', 'sobre', 'propias', 'esas', 'usais', 'otros', 'esta', 'habiendo', 'tendrían', 'habríais', 'tuvimos', 'c', 'estoy', 'cuales', 'lejos', 'primer', 'siempre', 'hayamos', 'quienes', 'habrían', 'seré', 'tres', 'tu', 'habia', 'delante', 'poco', 'un', 'dicho', 'diferente', 'ello', 'te', 'teníamos', 'estarían', 'atras', 'estaba', '_', 'fueseis', 'intentais', 'ningunos', 'decir', 'explicó', 'f', 'consigo', 'me', 'nosotros', 'otra', 'bueno', 'menos', 'ya', 'p', 'fuimos', 'pocas', 'habría', 'cada', '4', 'varios', 'hubiste', 'estuvieseis', 'fui', 'seríais', 'era', 'tuviera', 'hayan', 'ésas', 'cierta', 'tendremos', 'ninguna', 'proximo', 'anterior', 'dentro', 'incluso', 'sabe', 'vosotras', 'hacerlo', 'mientras', 'tendré', 'tenías', 'último', 'mayor', 'mi', 'mías', 'muy', 'este', 'hubieron', 'k', 'estéis', 'las', 'saben', 'antano', 'van', 'hubiesen', 'hace', 'tened', 'estad', 'tendría', 'fuera', 'seis', 'manifestó', 'suyas', 'tenéis', 'acuerdo', 'al', 'estuviese', 'vamos', 'estemos', 'todavia', 'ningún', 'nunca', 'otro', 'solamente', 'todos', 'soyos', 'estadas', 'hubieses', 'hubieran', 'dicen', 'demasiado', 'la', 'pero', 'algunas', 'mia', 'mas', 'tienen', 'varias', 'lleva', 'principalmente', 'suyos', 'más', 'nada', 'eres', 'd', 'quiza', 'todavía', 'tuyos', 'míos', 'uso', 'consideró', 'general', 'encima', 'hubiese', 'podrán', 'fuéramos', 'podeis', 'aquí', 'informo', 'cerca', 'hacen', 'ocho', 'hemos', 'tuviéramos', 'dia', 'buenas', 'mismas', 'vuestros', 'hubiera', 'quedó', 'cómo', 'mal', 'todas', 'mios', 'conmigo', 'estuvierais', 'él', 'estáis', 'tanto', 'hubieseis', 'realizar', 'llegó', 'y', 'estaréis', 'usted', 'estás', '1', 'habrán', 'podria', 'habrás', 'trabajo', 'cuanta', 'ambos', 'pasado', 'ellos', 'estamos', 'vaya', 'durante', 'antes', 'hubiéramos', 'estado', 'trabajar', 'pocos', 'cinco', 'hay', 'mio', 'fuiste', 'respecto', 'arriba', 'partir', 'habíais', 'seremos', 'aquéllas', 'estaríais', 'podrian', 'tienes', 'estan', 'no', 'éstas', 'mío', 'b', 'junto', 'estábamos', 'sólo', 'después', 'afirmó', 'primero', 'eran', 'saber', '9', 'hubimos', 'cuáles', 'como', 'serán', 'enseguida', 'empleas', 'estaríamos', 'tengamos', 'ser', 'ti', 'asi', 'será', 'desde', 'nosotras', 'n', 'hayas', 'estés', 'habremos', 'aun', 'tan', 'ademas', 'primeros', 'va', 'aún', 'vosotros', 'estuvieron', 'arribaabajo', 'buen', 'dejó', 'tendréis', 'dice', 'mediante', '7', 'he', 'estén', 'existe', 'luego', 'cuánta', 'en', 'serías', 'podrá', 'quizá', 'llevar', 'ustedes', 'valor', 'pues', 'intentan', 'aproximadamente', 'estarías', 'repente', 'z', 'ver', 'ha', 'tengas', 'tenían', '8', 'esté', 'deprisa', 'de', 'éramos', 'aquella', 'fin', 'alguno', 'estuviesen', 'o', 'misma', 'modo', 'hoy', 'cuatro', 'poner', 'verdadera', 'por qué', 'también', 'hablan', '0', 'menudo', 'mejor', 'estuvimos', 'consiguen', 'intento', 'lugar', 'trabajamos', 'u', 'despacio', 'adrede', 'usar', 'aquellas', 'quién', 'v', 'parece', 'fue', 'cuánto', 'tenidos', 'excepto', 'quiere', 'dado', 'para', 'vuestro', 'estando', 'por', 'hubo', 'detrás', 'comentó', 'serían', 'erais', 'mismos', 'tenía', 'segunda', 'seas', 'estaré', 'r', 'su', 'así', 'agregó', '5', 'bastante', 'fueses', 'si', 'temprano', 'realizó', 'tercera', 'tenida', 'con', 'algo', 'cual', 'grandes', 'sabeis', 'tambien', 'tuviese', 'próximos', 'habéis', 'allí', 'cuantas', 'salvo', 'se', 'estabas', 'sino', 'alguna', 'ante', 'haciendo', 'una', 'tener', 'tuviésemos', 'eramos', 'estuvieran', 'ningunas', 'cuanto', 'sera', 'última', 'nuestros', 'podría', 'hubierais', 'encuentra', 'mencionó', 'posible', 'usamos', 'm', 'mí', 'fuese', 'consigues', 'estaría', 'sé', 'dieron', 'están', 'que', 'estarás', 'donde', 'es', 'debe', 'trabaja', 'embargo', 'sal', '2', 'poca', 'tuyas', 'cualquier', 'ahora', 'pueda', 'haceis', 'demás', 'debido', 'peor', 'esos', 'detras', 'podriais', 'aquello', 'quiénes', 'trata', 'sido', 'tenga', 'e', 'uno', 'voy', 'muchas', 'ampleamos', 'habla', 'mía', 'tuvieran', 'fuerais', 'has', 'fuésemos', 'eso', 'ni', 'propios', 'toda', 'estarán', 'tendríamos', 'emplean', 'haya', 'aunque', 'seríamos', 'teníais', 'tuve', 'tuviesen', 'diferentes', 'empleo', 'estuvo', 'entonces', 'nuevos', 'algún', 'expresó', 'sabemos', 'hubiésemos', 'tuvieseis', 'lo', 'próximo', 'l', 'vais', '3', 'los', 'tiene', 'seréis', 'habidos', 'hubisteis', 'estada', 'pueden', 'días', 'suya', 'puede', 'quizas', 'parte', 'hayáis', 'despues', 'realizado', 'habido', 'casi', 'últimos', 'aquellos', 'tuvierais', 'aquél', 'veces', 'había', 'segun', 'cuántas', 'podemos', 'propio', 'adelante', 'según', 'buena', 'buenos', 'tuvo', 'algunos', 'habríamos', 'habrá', 'ejemplo', 'propia', 'dónde', '6', 'nuestra', 'mismo', 'contigo', 'ella', 'emplear', 'nuevo', 'pais', 'alli', 'añadió', 'alrededor', 'hecho', 'estados', 'cuándo', 'estabais', 'pronto', 'intentas', 'tenidas', 'tiempo', 'sea', 'tenemos', 'trabajan', 'gueno', 'aqui', 'final', 'tras', 'le', 'serás', 'tuvieras', 'ayer', 'tus', 'supuesto', 'día', 'hubieras', 'q', 'deben', 'tenido', 'el', 'pesar', 'sus', 'siguiente', 's', 'nos', 'éste', 'g', 'habrías', 'fueran', 'hacemos', 'verdad', 'fueron', 'hasta', 'dio', 'estuviésemos', 'hube', 'bien', 'nueva', 'qué', 'dias', 't', 'tengáis', 'porque', 'muchos', 'podrias', 'habidas', 'mias', 'últimas', 'eras', 'x', 'dos', 'os', 'solas', 'antaño', 'ninguno', 'sí', 'todo', 'sola', 'teniendo', 'estaban', 'siendo', 'suyo', 'fueras', 'hizo', 'usa', 'además', 'empleais', 'vuestra', 'primera', 'estas', 'existen', 'seamos', 'tendrá', 'usas', 'w', 'enfrente', 'mucho', 'seáis', 'trabajais', 'dijo', 'hacer', 'solos', 'ciertos', 'da', 'del', 'sin', 'indicó', 'intentar', 'mucha', 'estuvieras', 'ése', 'debajo', 'tendrías', 'haber', 'hicieron', 'aquélla', 'ex', 'tendrán', 'dar', 'informó', 'ellas', 'fuesen', 'tuya', 'cuando', 'habías', 'cuenta', 'ese', 'solo', 'mis', 'podrían', 'soy', 'poder', 'aquel', 'cierto', 'estaremos', 'manera', 'son', 'estais', 'otras', 'unos', 'tengo', 'está', 'ir', 'señaló', 'h', 'habré', 'pasada', 'tuvieron', 'tal', 'tuviste', 'tú', 'verdadero', 'esto', 'total', 'sería', 'fuisteis', 'dan'}

        return frozenset(burn)

    def __compute_count_vector(self, col):
        corpus = col.values
        cv = CountVectorizer(stop_words=self.stopwords)
        cv.fit(corpus)
        return cv


    def __load_model_columns(self):
        # model_columns = joblib.load(self.columns_pkl_path)
        model_columns = [#Provide the columns your model use for predict
			]
        return model_columns
        

    def predict(self, json_data):
                

        input_df = self.__input_to_df(json_data)
        query_col = self.__get_query(input_df)
        document_col = self.__get_names(input_df)
        count_vector = self.__compute_count_vector(document_col)
        count_vector_document_col = count_vector.transform(document_col)

        
        df = pandas.DataFrame(input_df['products'][0])

        try:
            count_vector_query_col = count_vector.transform(query_col)
        
            #Transformations
		#Perform any transformation required
	

            #Calculo TF_IDF
            df["TF_IDF"] = self.__computeTFIDF(count_vector_document_col,count_vector_query_col)

            prediction = list(self.model.predict(df[self.columns])) 
            df['prediction'] = prediction
            print(prediction)

            #Transformations
            df.columns = df.columns.str.lower()
            df.sort_values(by='prediction',ascending=False, inplace=True)
            df.drop('prediction', axis=1, inplace=True)
            
            return df
        
        except:
            df.columns = df.columns.str.lower()
            return df


def load_data(path):
    logging.info('Loading data')
    try:
        data = pandas.read_csv(path)
        return data

    except:
        logging.error('Failed to load data')
        traceback.print_last()


def pipeline_trainer(X, y, pipeline):

    logging.info('STATUS: Trainning the model')
    pipeline.fit(X, y)

    return pipeline

def write_to_pkl(pipeline, path, name):
    logging.info('Writting to pkl')
    try:
        model = {'pipeline':pipeline}
        with open(f'{path}/{name}.pkl', "wb") as f:
            pickle.dump(model, f)
        logging.info('Writed the pkl file')

    except:
        logging.error('Failed to write pkl file')

def pipeline_construct(numeric_features,categorical_features):
    try:
        logging.info('Constructing pipeline')
        # We create the preprocessing pipelines for both numeric and categorical data.
        numeric_features = numeric_features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())])

        categorical_features = categorical_features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

        #Append classifier to preprocessing pipeline.
        #Now we have a full prediction pipeline.
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('etr', ExtraTreesRegressor(n_estimators=200, min_samples_split=5, random_state=1, n_jobs=-1))])

        return pipeline
    
    except:
        logging.error('Failed to construct the pipeline')


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Logging level
    logging.getLogger().setLevel(logging.INFO)

    #Setting paths
    model_path = settings.ETR_MODEL_PATH
    proc_path = settings.PROCESSED_PATH
    country = settings.COUNTRY_CODE
    

    #Setting model name
    model_name = 'etr'

    #Getting data for train
    if f'{country.lower()}' in os.listdir('../models'):
        pass
    else: os.mkdir(f'../models/{country.lower()}')

    for file in os.listdir(f'{proc_path}'):
        if fnmatch.fnmatch(file, f'*{country}*.csv'):
            data = load_data(f'{proc_path}{file}')
            print('Loading file: ', file)  
    

    X = data[[ #Provide the columns of your X-Data
		]]
    y = data[#Provide The objective of your target-data]

    #Creating Pipeline
    numeric_features = [#Provide your numeric features
			]
    categorical_features = [#Provide your categorical features
				]
    pipeline = pipeline_construct(numeric_features,categorical_features)

    #Train Model
    model = pipeline_trainer(X,y,pipeline)
    
    #Write to pickle

    ltr = {'Ltr': Ltr()}
    with open(f"../", "wb") as f:
        pkl.dump(ltr, f, byref=False)
