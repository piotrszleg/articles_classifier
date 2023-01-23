import keras
from process_text import process_text
from field_predictor import FieldPredictor, FIELDS
from constants import MODELS_PATH, FIELD_VALUES
from os import path

class FieldPredictorRNN(FieldPredictor):
    def __init__(self, field):
        super().__init__(field)
        self.field_values = FIELD_VALUES[field]
        self.model = keras.models.load_model(path.join(MODELS_PATH, f'RNN_{field}'))

    def predict(self, text):
        text = process_text(text)
        result = self.model.predict([text])
        result_parsed = list(zip(self.field_values, result[0]))
        return max(result_parsed, key=lambda p: p[1])[0]

    def predict_all(self, text):
        result = self.model.predict([text])
        result_parsed = list(zip(self.field_values, result[0]))
        result_parsed.sort(key=lambda p: p[1], reverse=True)
        result_parsed=dict(result_parsed)
        result_parsed

def all_predictors():
    predictors = {}    
    for field in FIELDS:
        predictors[field] = FieldPredictorRNN(field)
    return predictors

if __name__=="__main__":
    field = FIELDS[2]
    text = 'New York Yankees won the match!'
    print(FieldPredictorRNN(field).predict(text))