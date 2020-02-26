# Dependencies
from flask import Flask, request, jsonify
import traceback
#import pandas
#import numpy
import json

#from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
#from sklearn.impute import SimpleImputer
#from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
import dill as pkl

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if ltr_object:
        try:
            json_ = request.data
            
            data = ltr_object['Ltr'].predict(json_)
            
            return jsonify(json.loads(data.to_json(orient='records', force_ascii=False)))

        except:
            
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    
    pkl._dill._reverse_typemap['ClassType'] = type
    ltr_file = open('models/uy/ltr.pkl', 'rb')
    ltr_object = pkl.load(ltr_file)

    print('Model loaded')
    print('Model columns loaded') 
    
    app.run(port=port, host='0.0.0.0', debug=True)