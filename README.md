# Articles classifier

Classifier for articles from a sample SQL database to following attributes:

`Gender, League, Level, Sport, Team`

The dataset is closed source so the repository is only left for viewing.

## Experiments summary

**Linear regression** with count vectorization turned out to be the best approach with over 90% accuracy on all fields.

Also **RNN network** was tested, containing **bidirectional LSTM layers** which are able to capture words sequences. But the accuracy was mostly comparable or slightly worse compared to linear regression model. 
Different dense, dropout and LSTM layers combinations were tested, but still the accuracy was only comparable to linear regression model. Which might come from the fact that in this dataset counting the words is the best approach and doing it directly instead of waiting for RNN to develop the same mechanism yields better results. 

Two other neural network approaches were tested: **1D convolution** and **count vectorizer connected with dense layers**. But they were comparable or worse than other approaches so they were only saved in the `archive` folder.

Classes and server integration for RNN model were created simply because, due to a human error, at the beginning only the article titles were used for training and RNN yielded slightly better results on those.

The `Teams` field couldn't be fully used for training. It contained a lot of classes with few examples, which must've been filtered. This could be fixed by either adding more examples of each team or grouping the teams into regions or types by a human specialist. Also the missing fields could be filled in by humans using their knowledge or the internet.

## Installation

```bash
pip install -r requirements.txt 
```

## Server

If you want to run the server on pre-trained models see [server README](./server/README.md).

## Training from scratch

### Preparing the data

You can download the data by running `download.py` script with database credentials filled in the code.

Run these scripts to process the data:

```bash
python process_titles.py
python filter_teams.py
```

Script `plot_teams.py` was used to decide on teams to use in training as a lot of them had too few representatives and there were simply too many classes for dataset this size. But you don't need to run it if you're not interested in that.

If you modified them though you'll need to run `generate_field_values.py` script and then set correct `server/constants.py` file.

### Training

First model consisiting of count vectorizer and logistic regression can be trained using `train_rl.py` script. 

RNN model can be trained using `train_rnn.py` script. 

Models can be compared using `compare.py` script.

To deploy the models you can pack and move all of the chosen `models/*` folders to your other project, you can also reuse the FieldPredictor classes from server.
