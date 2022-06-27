import pandas
import random
import spacy
from spacy import util
from spacy.language import Language
from spacy.tokens import DocBin
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import filter_spans
from tqdm import tqdm

def print_doc_entities(_doc: Doc):
    if _doc.ents:
        for _ent in _doc.ents:
            print(f"     {_ent.text} {_ent.label_}")
    else:
        print("     NONE")

# Read video game database and retrieve the names
df = pandas.read_csv('steam.csv', 
            header=0,
            usecols=['name'])
df = df.reset_index()
train_data = []
# Format the train data in a way that is accepted by spacy. [VIDEO_GAME_NAME, {'entities': (START_POS, END_POS, LABEL)}]
for index, row in df.iterrows():
    game = row['name'].strip()
    train_data.append([game, {'entities': [(0, len(game), 'VGAME')]}])

# Inspect the final list
# df2 = pandas.DataFrame(train_data)
# df2.to_csv('myfile.csv')

nlp = spacy.load("en_core_web_lg") # load a new spacy model
print(f"Result before training:")
doc = nlp(u'I will play Spear of Destiny or Counter Strike and LEGO.')
print_doc_entities(doc)
disabled_pipes = []
for pipe_name in nlp.pipe_names:
    if pipe_name != 'ner':
        nlp.disable_pipes(pipe_name)
        disabled_pipes.append(pipe_name)


print("   Training ...")
doc_bin = DocBin() # create a DocBin object
optimizer = nlp.create_optimizer()
for _ in tqdm(range(1)):
    random.shuffle(train_data)
    for raw_text, entity_offsets in tqdm(train_data):
        doc = nlp.make_doc(raw_text)
        example = Example.from_dict(doc, entity_offsets)
        nlp.update([example], sgd=optimizer)
        doc_bin.add(doc)

# Enable all previously disabled pipe components
for pipe_name in disabled_pipes:
    nlp.enable_pipe(pipe_name)

# Result after training
print(f"Result after training:")
doc = nlp(u'I will play Spear of Destiny or Counter Strike and LEGO.')
print_doc_entities(doc)
doc_bin.to_disk("custom_nlp_model.spacy") # save the docbin object


# Alternative Method
# doc_bin = DocBin() # create a DocBin object

# if "ner" not in nlp.pipe_names:
#     nlp.add_pipe('ner')

# ner = nlp.get_pipe("ner")
# ner.add_label('VGAME')

# for training_example in tqdm(train_data): 
#     text = training_example[0]
#     labels = training_example[1]['entities']
#     doc = nlp.make_doc(text) 
#     ents = []
#     for start, end, label in labels:
#         span = doc.char_span(start, end, label=label, alignment_mode="contract")
#         if span is None:
#             print("Skipping entity")
#         else:
#             ents.append(span)
#     filtered_ents = filter_spans(ents)
#     doc.ents = filtered_ents 
#     doc_bin.add(doc)
# 
# doc_bin.to_disk("custom_nlp_model.spacy") # save the docbin object