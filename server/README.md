# Articles classifier - API Server

API server for classifier for text data, mapping it to following attributes:

`Gender, League, Level, Sport, Team`

## Installation

```bash
pip install -r requirements.txt 
```

## Run

The folder already contains trained models, so you can run the server without training like this:

```bash
python server.py
```

And then test the server by opening this link in your browser:

```
127.0.0.1:5000/?text=New York Yankees won the match!
```

The server accepts a `text` query parameter and returns a json dict with following (nullable) fields: 

`Gender, League, Level, Sport, Team`

If you want to deploy the server folder alone you need to set correct models path in `constants.py` file.
