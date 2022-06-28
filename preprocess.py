
import spacy
import json
import pandas as pd

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

def get_sf_unused_nums(data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Get list of testimony ids for the SF corpus
    :param data_path:
    :return:
    """
    with open(data_path + 'sf_unused3.json', 'r') as infile:
        nums = list(json.load(infile))
    return nums

def make_sents(doc):
    sents = []
    for s in doc.sents:
        if len(sents) > 0 and sents[-1][-1] == ":":
            sents[-1] = sents[-1].strip() + " " + s.text
        else:
            sents.append(s.text)
    return sents

def to_csv(sent_dicts, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/'):
    df = pd.DataFrame.from_dict(sent_dicts, orient='index').T
    df.to_csv(base_path + "sents.csv")

if __name__ == "__main__":
    nums = get_sf_unused_nums()[1:31]
    sent_dicts = {}
    nlp = spacy.load("en_core_web_sm")
    for n in nums:
        sent_dicts[str(n)] = make_sents(nlp(get_sf_testimony_text(n)))
    to_csv(sent_dicts)
