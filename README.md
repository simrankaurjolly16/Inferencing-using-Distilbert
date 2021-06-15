# Inferencing-using-Distilbert
Sentiment analysis: is a text positive or negative?
## Download and install using pip
$ pip install transformers
## usage in python
# Import generic wrappers
from transformers import AutoModel, AutoTokenizer 


# Define the model repo
model_name = "distilbert-base-uncased-finetuned-sst-2-english" 


# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Transform input tokens 
inputs = tokenizer("Hello world!", return_tensors="pt")

# Model apply
outputs = model(**inputs)
