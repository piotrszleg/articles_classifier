import re

def process_text(text):
    # replace all unicode "'" like characters with "'"
    text = re.sub(r"[‘’‛]", "'", text)
    # remove characters that are not alphanumeric, "'" or whitespace
    text = re.sub(r"[^\w\s']", '', text)
    # change any whitespace to spaces
    text = re.sub(r'\s', ' ', text)

    return text