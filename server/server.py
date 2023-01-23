from flask import Flask, request, jsonify
from field_predictor_rnn import FieldPredictorRNN
from field_predictor_lr import FieldPredictorLR

PREDICTOR_CHOICES = dict(
    # 0 for lr, 1 for rnn
    Gender=0,
    League=0,
    # Level predictor was left as RNN
    # for demonstration purposes
    Level=1,
    Sport=0,
    Team=0
)

predictors = {}
for field, choice in PREDICTOR_CHOICES.items():
    if choice==0:
        predictors[field] = FieldPredictorLR(field)
    else:
        predictors[field] = FieldPredictorRNN(field)  

app = Flask(__name__)
@app.route('/')
def index():
    text = request.args.get('text') or ""

    result = {}
    result['text'] = text
    for name, predictor in predictors.items():
        result[name] = predictor.predict_dash_to_none(text)

    return jsonify(result)

app.run()