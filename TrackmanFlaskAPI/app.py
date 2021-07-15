from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import cpModel
from model import stModel

app = Flask(__name__)
api = Api(app)

model = cpModel()

clf_path = 'lib/clubClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'lib/models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictClub(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)

        # Assigning swing prediction strings
        if prediction == 23:
            pred_text = 'Degree wedge'
        elif prediction == 22:
            pred_text = 'Sand wedge'
        elif prediction == 21:
            pred_text = 'Gap wedge'
        elif prediction == 20:
            pred_text = 'Pitching wedge'
        elif prediction == 19:
            pred_text = '9 iron'
        elif prediction == 18:
            pred_text = '8 iron'
        elif prediction == 17:
            pred_text = '7 iron'
        elif prediction == 16:
            pred_text = '6 iron'
        elif prediction == 15:
            pred_text = '5 iron'
        elif prediction == 14:
            pred_text = '4 iron'
        elif prediction == 13:
            pred_text = '3 iron'
        elif prediction == 12:
            pred_text = '2 iron'
        elif prediction == 11:
            pred_text = '6 hybrid'
        elif prediction == 10:
            pred_text = '5 hybrid'
        elif prediction == 9:
            pred_text = '4 hybrid'
        elif prediction == 8:
            pred_text = '3 hybrid'
        elif prediction == 7:
            pred_text = '2 hybrid'
        elif prediction == 6:
            pred_text = '6 wood'
        elif prediction == 5:
            pred_text = '5 wood'
        elif prediction == 4:
            pred_text = '4 wood'
        elif prediction == 3:
            pred_text = '3 wood'
        elif prediction == 2:
            pred_text = '2 wood'
        elif prediction == 1:
            pred_text = 'Driver'
        elif prediction == 0:
            pred_text = 'Unknown'

        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 3)

        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}

        return output

# modelst = cstModel()
#
# clf_path = 'lib/swingTipClassifier.pkl'
# with open(clf_path, 'rb') as f:
#     modelst.clf = pickle.load(f)
#
# # argument parsing
# parser = reqparse.RequestParser()
# parser.add_argument('query')
#
# class SwingTip(Resource):
#     def get(self):
#         # use parser and find the user's query
#         args = parser.parse_args()
#         user_query = args['query']
#
#         # vectorize the user's query and make a prediction
#         uq_vectorized = modelst.vectorizer_transform(np.array([user_query]))
#         prediction = modelst.predict(uq_vectorized)
#         pred_proba = modelst.predict_proba(uq_vectorized)
#
#         # Output either 'Negative' or 'Positive' along with the score
#         if prediction == 0:
#             pred_text = 'Negative'
#         else:
#             pred_text = 'Positive'
#
#         # round the predict proba value and set to new variable
#         confidence = round(pred_proba[0], 3)
#
#         # create JSON object
#         output = {'prediction': pred_text, 'confidence': confidence}
#
#         return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictClub, '/clubPredict')
# api.add_resource(SwingTip, '/swingTip')

#Set debug to False if deploying to production
if __name__ == '__main__':
    app.run(debug=True)
