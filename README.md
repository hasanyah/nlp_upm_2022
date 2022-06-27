# How to run

In order to run the application, simply execute the ner_video_games.rmd script.

# Aditional notes

The following steps are not necessary to execute the R script, but they can be used to re-create the custom NER model using spacy.

## Install spacy and language pack
pip3 install spacy==3.1.3
python -m spacy download en_core_web_lg

## Run the train_custom_ner.py file to re-generate custom_nlp_model.spacy

python3 train_custom_ner.py

## Generate Spacy config
python3 -m spacy init fill-config base_config.cfg config.cfg

## Train the newly generated model
python3 -m spacy train config.cfg --output ./ --paths.train ./custom_nlp_model.spacy --paths.dev ./custom_nlp_model.spacy

The newly generated model-best and  model-last can now be used by Python/R application.