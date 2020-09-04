import os


# Set the default data directory to be the one this file's directory + data/
data_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.realpath(os.path.join(data_dir, 'data'))

# Spacy model name
spacy_model_name = 'en_core_web_sm'
