import pickle
from process_text import process_text
from field_predictor import FieldPredictor, FIELDS
from constants import MODELS_PATH, FIELD_VALUES
from os import path

class FieldPredictorLR(FieldPredictor):
    def __init__(self, field):
        super().__init__(field)
        self.field_values = FIELD_VALUES[field]
        self.vectorizer = pickle.load(open(path.join(MODELS_PATH, f"{field}.v"), 'rb'))
        self.model = pickle.load(open(path.join(MODELS_PATH, f"{field}.lrm"), 'rb'))

    def predict(self, text):
        text_vectorized = self.vectorizer.transform([process_text(text)])
        return self.model.predict(text_vectorized)[0]

    def predict_all(self, text):
        predict_result=self.predict(text)
        return {field : field==predict_result for field in self.field_values}

def all_predictors():
    predictors = {}    
    for field in FIELDS:
        predictors[field] = FieldPredictorLR(field)
    return predictors

if __name__=="__main__":
    print("Input text to get predictions")
    while True:
        text = input("> ")
        for field, predictor in all_predictors().items():
            print(field, predictor.predict(text))