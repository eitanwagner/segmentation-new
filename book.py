
from booknlp.booknlp import BookNLP
import json
import numpy as np
import spacy
from spacy.tokens import Token
from spacy.tokens import Doc
from spacy.tokens import Span
Token.set_extension("is_event", default=None, force=True)
Span.set_extension("ents", default=None, force=True)

model_params={
    "pipeline":"entity,quote,supersense,event,coref",
    "model":"big"
}



def get_sf_testimony_text(i, data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Obatin a list of sentences for testimony i (in the SF corpus)
    :param i:
    :param data_path:
    :return:
    """
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        text = json.load(infile)[str(i)]
    return text

def process(i, data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/"):
    booknlp = BookNLP("en", model_params)
    # Input file to process
    input_file = data_path + f"testimony_{i}.txt"
    # Output directory to store resulting files in
    output_directory = data_path + "booknlp/"
    # File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
    book_id=f"t_{i}"
    booknlp.process(input_file, output_directory, book_id)

def make_text_file(i, data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/"):
    text = get_sf_testimony_text(i)
    with open(data_path + f"testimony_{i}.txt", "w+") as outfile:
        outfile.write(text)

def extract_events(i, data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/"):
    with open(data_path + f"booknlp/t_{i}.tokens", "r") as infile:
        lines = infile.readlines()
    events = np.array([l.split()[-1] for l in lines[1:]])
    return events == "EVENT"

def extract_ents(i, data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/"):
    with open(data_path + f"booknlp/t_{i}.entities", "r") as infile:
        lines = infile.readlines()
    ents = [l.split()[:3] for l in lines[1:]]  # the end location *is included*
    return ents

def add_events_to_doc(doc, i, process=False):
    doc.spans["sents"] = [s for s in doc.sents]
    events = extract_events(i)
    for i, t in enumerate(doc):
        t._.is_event = events[i]

def add_ents_to_doc(doc, i, process=False):
    sents = [s for s in doc.sents]
    for s in sents:
        s._.ents = []
    ents = extract_ents(i)
    ent_locs = [int(e[1]) for e in ents]
    sents_starts = [s.start for s in sents]

    from utils import merge_locs
    locs_in_sents = merge_locs(sents_starts, ent_locs)
    for i, loc in enumerate(locs_in_sents):
        sents[loc-1]._.ents.append(int(ents[i][0]))


if __name__ == "__main__":
    ts = [20581]
    # make_text_file(ts[0])
    # process(ts[0])
    e = extract_events(ts[0])
    print(e)