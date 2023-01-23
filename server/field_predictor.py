from typing import Dict, Optional

FIELDS = "Gender,League,Level,Sport,Team".split(',')

class FieldPredictor(object):
    def __init__(self, field):
        if field not in FIELDS:
            raise ValueError("Invalid field")

    def predict(self, text:str)->str:
        raise NotImplementedError()

    def predict_dash_to_none(self, text:str)->Optional[str]:
        result = self.predict(text)
        if result == "-":
            return None
        else:
            return result

    def predict_all(self, text:str)->Dict[str, float]:
        raise NotImplementedError()