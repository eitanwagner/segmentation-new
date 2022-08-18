import spacy
import json
import numpy as np
import re
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from evaluation import edit_distance, gestalt_diff
from sklearn.metrics import precision_recall_fscore_support
import os
import sys
import wandb

import torch
from torch import nn
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

from sentence_transformers import SentenceTransformer, util
from loc_clusters import find_closest

import joblib
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dev2 = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cpu")

# import enum
# class Times(enum.Enum):
BEFORE = 0
AFTER_RISE = 1
BEFORE_INVASION = 2  # but after the war broke out
BEFORE_EXTERMINATION = 3  # but after the invasion to this country
DURING_WAR = 4
AFTER_WAR = 5


def make_time_category(terms):
    """
    Makes time category given a list of terms
    :param terms:
    :return:
    """
    def _make_time(str):
        """
        Converts sf label to time category (if relevant)
        :param label:
        :return:
        """
        # we need to add location data.
        # for now, we ignore classes 2 and 3
        years = re.findall(r'\d{4}', str)
        if len(years) == 0:
            return None
        elif len(years) >= 2:
            if int(years[1]) <= 1939:
                return BEFORE
            if int(years[1]) <= 1945:
                return DURING_WAR
            else:
                return AFTER_WAR
        else:
            if int(years[0]) < 1939:
                return BEFORE
            if 1939 < int(years[0]) < 1945:
                return DURING_WAR
            if int(years[0]) > 1945:
                return AFTER_WAR
        return None

    times = [_make_time(t) for t in terms]
    times = set([t for t in times if t is not None])
    # for t in terms:
    #     m_t = _make_time(t)
    #     if m_t is not None:
    #         return m_t
    if len(times) == 1:  # if no time or if two different times, return None
        return times.pop()
    return None

def make_time_data(data_path):
    with open(data_path + 'sf_all.json', 'r') as infile:
        data = json.load(infile)

    time_data = {}
    for t, d in data.items():
        time_data[t] = [[_d['text'], [int(make_time_category(_d['terms']))]] for _d in d if make_time_category(_d['terms']) is not None]

    # all_data = [t_d for t, d in time_data for t_d in d if t_d[1] is not None]
    with open(data_path + 'time_all.json', 'w') as outfile:
        json.dump(time_data, outfile)

def get_year(span):
    for t in span:
        _t = t.text
        while _t[-1] == "-":
            _t = _t[:-1]
            if len(_t) == 0:
                return None
        if t.is_digit:
            if len(t.text) == 4 and t.text[:2] == "19":
                return int(t.text)
        elif _t[-1] == "s" and len(_t) > 1:
            _t = _t[:-1]
        elif _t[-2:] in ["th", "rd", "nd"] and len(_t) > 2:
            _t = _t[:-2]

        if _t.isnumeric() and len(_t[1:]) == 2 and (_t[0] == "3" or _t[0] == "4"):
            return 1900 + int(_t)
    return None

def plot(x, y, t="", t_len=None, save=False, t_num=None):
    import matplotlib.pyplot as plt
    if t != "":
        plt.title(t)
    plt.plot(x, y, "ro")
    top = 800
    if t_len is not None:
        top = t_len + 10
    plt.axis([0, t_len, 1900, 1980])
    if save:
        plt.savefig(f"C:/Users/Eitan/nlplab/time_figs/{t_num}.png")
    plt.show()


def is_visit(terms, terms_df):
    for t in terms:
        row = terms_df[terms_df['Label'] == t].squeeze().to_dict()
        if len(row['Sub-Type']) > 0 and row['Sub-Type'].find("returns and visits") >= 0:
            return True
    return False


def extract_loc(terms, terms_df, return_cat=False, return_country=False):
    # TODO what if there are more than one?
    _terms = []
    loc_cats = ["cities in", "kibbutzim", "moshavim in", "German concentration camps in", "German death camps in", "displaced persons camps or",
                "refugee camps", "ghettos in", "administrative units in", "Croatian concentration camps in",
                "German prisoner of war camps in", "Slovakian concentration camps in", "Soviet concentration camps in",
                "Hungarian concentration camps in", "Romanian concentration camps in", "Polish concentration camps in", "Cambodian camps"]

    cat = ""

    country_cats = ["periodizations by country", "countries"]
    _country = ""
    c_cat = ""

    for t in terms:
        # for bad characters
        end = t.find("?")
        end2 = end + t[end+1:].find("?") + 1
        end3 = end2 + t[end2+1:].rfind("?") + 1

        # if end2 == 0:  # starts with ??
        #     print("bad term!!!!!")
        if end == -1:  # no '?'
            row = terms_df[terms_df['Label'] == t]
        elif end2 == end:  # only one '?'
            row = terms_df[pd.Series([_t[:end] + _t[end+1:] for _t in terms_df['Label']])
                           == t[:end] + t[end+1:]]
        elif end3 == end2:  # two '?'
            row = terms_df[pd.Series([_t[:end] + _t[end+1:] + _t[end2+1:] for _t in terms_df['Label']])
                           == t[:end] + t[end+1:] + t[end2+1:]]
        else:  # more than two '?'
            row = terms_df[pd.Series([_t[:end] + _t[end+1:end2] + _t[end3+1:] + str(_t.count("?")) for _t in terms_df['Label']])
                           == t[:end] + t[end+1:end2] + t[end3+1:] + str(t.count("?"))]
        if row.size == 0:
            row = row.squeeze().to_dict()
        else:
            row = row.iloc[0].squeeze().to_dict()

        if len(row['Label']) > 0:
            for _c in loc_cats:
                if row['Sub-Type'][:len(_c)] == _c:
                    _terms.append(row['Label'])
                    if cat == "":  # to take the first
                        # cat = _c
                        cat = row['Sub-Type']
            for _c in country_cats:
                if row['Sub-Type'][:len(_c)] == _c:
                    _country = row['Label']
                    if c_cat == "":  # to take the first
                        c_cat = _c
                    # print("!!!!!" + row['Label'])
        # print(row['Sub-Type'])
        # print(row['Label'])

    # returns only first
    if len(_terms) > 0 or len(c_cat) > 0:
        if len(_terms) == 0:
            _terms.append("")
        if return_cat:
            if not return_country:
                return _terms[0], cat
            return _terms[0], [cat, _country, c_cat]
        return _terms[0]
    if return_cat:
        if not return_country:
            return None, ""
        return None, ["","",""]
    return None

def make_loc_data(data_path, use_segments=True, with_cat=False, with_country=False):
    print("Starting")
    terms_df = pd.read_csv(data_path + "All indexing terms 4 April 2022.xlsx - vhitm_data (14).csv", encoding='utf-8')
    # terms_df = pd.read_csv(data_path + "All indexing terms 4 April 2022.xlsx - vhitm_data (14).csv", header=1)

    with open(data_path + 'sf_all.json', 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    with open(data_path + 'sf_unused5.json', 'r') as infile:
        unused = json.load(infile) + [45064]
        # unused = []

    loc_data = {}
    for t, d in data.items():
        visit = False
        print(t)
        last_loc = ""
        last_seg = ''
        cat = ""
        last_cat = ""
        if with_country:
            last_cat = [""]
        cs = []
        last_cs = []
        if int(t) not in unused:
            loc_data[t] = []
            if not use_segments:
                for s in d:
                    if with_cat:
                        current_loc, cat = extract_loc(s['terms'], terms_df=terms_df, return_cat=True, return_country=with_country)
                        if with_country:
                            c = cat[1]
                            # last_cs.append(c)
                    else:
                        current_loc = extract_loc(s['terms'], terms_df=terms_df, return_cat=False)
                    if current_loc is not None and current_loc != "" and current_loc != last_loc:
                        if with_cat:
                            if with_country:
                                loc_data[t].append([last_seg, last_loc, last_cat[0], list(set(last_cs) - {""})])
                                last_cs = []
                                # cs = []
                            else:
                                loc_data[t].append([last_seg, last_loc, last_cat])
                            last_cat = cat
                        else:
                            loc_data[t].append([last_seg, last_loc])
                        last_loc = current_loc
                        last_seg = ""
                    elif visit:
                        if with_cat:
                            if len(loc_data[t]) > 0:
                                if not with_country:
                                    current_loc = loc_data[t][-1][-2]  # last location added before the visit
                                else:
                                    current_loc = loc_data[t][-1][-3]
                            else:
                                if with_country:
                                    current_loc = ["","",""]
                                else:
                                    current_loc = ""
                            if with_country:
                                loc_data[t].append([last_seg, last_loc, last_cat[0], list(set(last_cs) - {""})])
                                last_cs = []
                                # cs = []
                            else:
                                loc_data[t].append([last_seg, last_loc, last_cat])
                            # last_cat = cat
                            last_cat = [loc_data[t][-1][-2]]
                        else:
                            if len(loc_data[t]) > 0:
                                current_loc = loc_data[t][-1][-1]  # last location added before the visit
                            else:
                                current_loc = ""
                            loc_data[t].append([last_seg, last_loc])

                        last_seg = ""
                        last_loc = current_loc
                    visit = is_visit(s['terms'], terms_df=terms_df)
                    last_seg = last_seg + s['text']
                    last_cs.append(c)
                if with_cat:
                    if with_country:
                        loc_data[t].append([last_seg, last_loc, last_cat[0], list(set(last_cs) - {""})])
                        # last_cs = []
                        # cs = []
                    else:
                        loc_data[t].append([last_seg, last_loc, last_cat])
                    # last_cat = cat
                else:
                    loc_data[t].append([last_seg, last_loc])
            else:
                for s in d:
                    if with_cat:
                        current_loc, cat = extract_loc(s['terms'], terms_df=terms_df, return_cat=True, return_country=with_country)
                    else:
                        current_loc = extract_loc(s['terms'], terms_df=terms_df, return_cat=False)
                    if current_loc is None:
                        current_loc = ""
                    if with_cat:
                        loc_data[t].append([s['text'], current_loc, cat, "visit" if is_visit(s['terms'], terms_df=terms_df) else ""])
                    else:
                        loc_data[t].append([s['text'], current_loc, "visit" if is_visit(s['terms'], terms_df=terms_df) else ""])

    if use_segments:
        with open(data_path + 'locs_segments_w_cat.json', 'w') as outfile:
            json.dump(loc_data, outfile)
    else:
        with open(data_path + 'locs_w_cat.json', 'w') as outfile:
            json.dump(loc_data, outfile)


def make_description_category_dict(data_path):
    """
    Makes dictionaries with the descriptions and categories for each location mentioned
    :param data_path:
    :return:
    """
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
    all_locs = set([_v[1] for v in data.values() for _v in v])
    terms_df = pd.read_csv(data_path + "All indexing terms 4 April 2022.xlsx - vhitm_data (14).csv", encoding='utf-8')
    desc_dict = {}
    cat_dict = {}
    for t in all_locs:
        if t!="":
            row = terms_df[terms_df['Label'] == t]
            desc_dict[t] = row['Definition'].to_list()[0]
            if row.isnull().values.any():
                desc_dict[t] = ""
            cat_dict[t] = row['Sub-Type'].to_list()[0]

    with open(data_path + 'loc_description_dict.json', 'w') as outfile:
        json.dump(desc_dict, outfile)
    with open(data_path + 'loc_category_dict.json', 'w') as outfile:
        json.dump(cat_dict, outfile)

# ************** train model
import torch

class Dataset(torch.utils.data.Dataset):
    """
    Dataset object
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class Dataset1(torch.utils.data.Dataset):
    """
    Dataset object
    """
    def __init__(self, data_path, labels):
        # make sure the order is correct!!
        self.data_files = os.listdir(data_path)
        self.data_files = sorted(self.data_files)
        self.labels = labels

    def __getitem__(self, idx):
        # encoded_data = []
        # for i in idx:
        with open(self.data_files[i], 'r') as infile:
            # encoded_data.append(json.load(infile))
            item = json.load(infile)  # assumes one per batch
        # return load_file(self.data_files[idx])
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def _add_loc_description(labels, desc_dict):
    return [l + ". " + desc_dict[l] for l in labels], labels

def _make_texts(data, unused, out_path, desc_dict=None, conversion_dict=None, vectors=None):
    """

    :param data:
    :param unused:
    :param out_path:
    :param desc_dict: whether uses location descriptions. In this case the labels are the original detailed locations
    :return:
    """
    c_dict = conversion_dict if conversion_dict is not None else {}

    if desc_dict is not None:
        desc_dict["START"] = "The beginning of the testimony."
        desc_dict["END"] = "The end of the testimony."
    texts = []
    labels = []
    for t, t_data in data.items():
        prev_text = ""
        prev_loc = ["START", "START"]
        # prev_loc = ""
        if t in unused:
            continue
        for i, d in enumerate(t_data + [["END", "END", ["END"], ""]]):
            if d[1] == ["", "", ""] or d[1] == "":
                if i > 0 and t_data[i-1][3] == "visit":  # if visit then the prevs stay
                    d[2][0] = labels[-2]
                d[1] = prev_loc[-1]
                if i <= 1:
                    # if vectors is not None:
                    #     d[2][0] = v_dict["START"]
                    # else:
                    d[2][0] = "START"
                else:
                    d[2][0] = labels[-1]
                # d[1] = ""
            if i == 0:
                prev_loc = ["START"] + [d[1]]
                # prev_text = " ".join(d[0].split()[-100:])
                prev_text = d[0]
                # texts.append(" [SEP] ".join([prev_loc[0], "", prev_loc[1], prev_text]))
            if i > 0:
                if desc_dict is None:
                    # if vectors is None:
                    labels.append(c_dict.get(d[2][0], d[2][0]))
                    # if vectors is not None:
                    #     if out_path[-1] == "6":
                    #         labels.append(torch.cat([labels[-1], vectors[c_dict.get(d[2][0], d[2][0])]]))
                    #     else:
                    #         labels.append(v_dict[c_dict.get(d[2][0], d[2][0])])
                else:
                    labels.append(c_dict.get(d[1], d[1]))
                # locs.append(prev_loc)
                # texts.append((prev_text, prev_loc))
                text = d[0]
                if out_path[-1] == "1":  # deberta1
                    texts.append(" [SEP] ".join([text]))
                elif out_path[-1] == "2":  # deberta2
                    texts.append(" [SEP] ".join([prev_loc[0], prev_text, prev_loc[1]]))
                elif out_path[-1] == "3":  # deberta3
                    texts.append(" [SEP] ".join([prev_loc[0], prev_loc[1]]))
                elif out_path[-1] == "4" or out_path.split("/")[-1][:6] == "distil":  # deberta4 or distilroberta
                    texts.append(" [SEP] ".join([prev_text, text]))
                elif out_path[-1] == "5" and desc_dict is not None:  # from loc to loc with description
                    texts.append(" [SEP] ".join([prev_loc[0] + ": " + desc_dict[prev_loc[0]],
                                                 prev_loc[1] + ": " + desc_dict[prev_loc[1]]]))
                elif out_path[-1] == "6" and desc_dict is not None:  # from prev to loc, with label
                    if len(labels) < 2:
                        texts.append(" [SEP] ".join([prev_text, "START"]))
                    else:
                        texts.append(" [SEP] ".join([prev_text, labels[-2]]))
                elif out_path[-1] == "6" and vectors is not None:
                    texts.append(" [SEP] ".join([prev_text]))
                else:
                    texts.append(" [SEP] ".join([prev_loc[0], prev_text, prev_loc[1], text]))

                # prev_text = " ".join(d[0].split()[-100:])
                prev_text = text
                prev_loc = [prev_loc[-1], d[1]]
    # if out_path.split("/")[-1][:6] ==  "distil":
    #     labels = _add_loc_description(labels, desc_dict)
    if vectors is not None:
        v_dict = {l:v for l,v in zip(*vectors)}
        label_vectors = [v_dict[l] for l in labels]
        if out_path[-1] == "6":
            label_vectors = [np.concatenate([l_v, label_vectors[i]]) for i, l_v in enumerate([v_dict["START"]] +
                                                                                        label_vectors[:-1])]
            # label_vectors = [torch.cat([l_v, label_vectors[i]]) for i, l_v in enumerate([v_dict["START"]] +
            #                                                                             label_vectors[:-1])]
        return texts, label_vectors
    return texts, labels


class MatrixTrainer(Trainer):
    def __init__(self, vectors=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectors = torch.from_numpy(vectors)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # print(labels.shape)
        # print(inputs)
        inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')  # of shape (batch_size, dim)?
        # print(logits.shape)
        # print(labels[:, :int(logits.shape[-1] ** .5)].unsqueeze(-1).shape)
        # print(logits.reshape(-1, int(logits.shape[-1] ** .5), int(logits.shape[-1] ** .5)).log_softmax(dim=1).shape)

        # out = torch.bmm(logits.reshape(-1, int(logits.shape[-1] ** .5), int(logits.shape[-1] ** .5)).log_softmax(dim=1),
        #                 labels[:, :int(logits.shape[-1] ** .5)].unsqueeze(-1)).squeeze(-1)

        # out is a tensor of shape (batch_size, vec_len)
        out = torch.bmm(logits.reshape(-1, int(logits.shape[-1] ** .5), int(logits.shape[-1] ** .5)),
                        labels[:, :int(logits.shape[-1] ** .5)].unsqueeze(-1)).squeeze(-1)
        # sims should be a tensor of shape (batch_size, num_classes)
        out_sims = util.cos_sim(out, self.vectors.to(logits.device))

        # label_sims = torch.zeros_like(out_sims)
        # label_sims[torch.arange(logits.shape[0]),
        #            [find_closest(vectors=torch.from_numpy(self.vectors).to(logits.device),
        #                          c_vector=l[int(logits.shape[-1] ** .5):])
        #             for l in labels]] = 1.

        labels = torch.stack([find_closest(vectors=self.vectors.to(logits.device),
                      c_vector=l[int(logits.shape[-1] ** .5):], tensor=True) for l in labels])

        # label_sims = util.cos_sim(labels[:, int(logits.shape[-1] ** .5):], self.vectors)

        loss_fct = nn.CrossEntropyLoss()
        # loss_fct = nn.MSELoss()
        # loss = loss_fct(out.squeeze(), labels[:, int(logits.shape[-1] ** .5):].squeeze())
        # loss = loss_fct(out_sims.softmax(dim=-1), label_sims)
        loss = loss_fct(out_sims.softmax(dim=-1), labels)
        return (loss, outputs) if return_outputs else loss

class VectorTrainer(Trainer):
    def __init__(self, vectors=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectors = torch.from_numpy(vectors)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')  # of shape (batch_size, vec_dim)?

        # should be of shape (batch_size, num_labels)
        out_sims = util.cos_sim(logits, self.vectors.to(logits.device))

        # label_sims = torch.zeros_like(out_sims)
        # label_sims[torch.arange(logits.shape[0]),
        #            [find_closest(vectors=self.vectors.to(logits.device), c_vector=l) for l in labels]] = 1.

        labels = torch.stack([find_closest(vectors=self.vectors.to(logits.device), c_vector=l, tensor=True) for l in labels])

        loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(out_sims.softmax(dim=-1), label_sims)
        loss = loss_fct(out_sims.softmax(dim=-1), labels)
        return (loss, outputs) if return_outputs else loss

def train_classifier(data_path, return_data=False, out_path=None, first_train_size=0.4, val_size=0.1, test_size=0.1,
                     conversion_dict=None, vectors=None, matrix=False):
    from sklearn.model_selection import train_test_split
    from transformers import Trainer, TrainingArguments
    from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from sklearn.metrics import accuracy_score
    from datasets import load_metric
    from sklearn.preprocessing import LabelEncoder
    import joblib

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("\n********* Training model on the GPU ***************")
    else:
        dev = torch.device("cpu")
        print("\nTraining model on the CPU")

    # if vectors is not None:
    #     v_dict = {l:v for l,v in zip(*vectors)}

    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def compute_metrics2(eval_pred):
        vs, labels = eval_pred
        predictions = [find_closest(vectors[0], vectors[1], v) for v in vs]
        eval_labels = [find_closest(vectors[0], vectors[1], v) for v in labels]
        return metric.compute(predictions=predictions, references=eval_labels)

    def compute_metrics3(eval_pred):
        with torch.no_grad():
            vs, labels = eval_pred
            _vs = [v.reshape(int(len(v)**.5), -1) @ l[:len(l)//2] for v, l in zip(vs, labels)]
            predictions = [find_closest(vectors[0], vectors[1], v) for v in _vs]
            eval_labels = [find_closest(vectors[0], vectors[1], l[len(l)//2:]) for l in labels]
            return metric.compute(predictions=predictions, references=eval_labels)

    with open(data_path + "sf_unused5.json", 'r') as infile:
        unused = json.load(infile) + ['45064']
    with open(data_path + "sf_five_locs.json", 'r') as infile:
        unused = unused + json.load(infile)
    # with open(data_path + "locs_w_cat.json", 'r') as infile:
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
        for u in unused:
            data.pop(u, None)

        if vectors is None:
            _, all_labels = _make_texts(data, [], out_path, conversion_dict=conversion_dict)
        if test_size > 0:
            train_data = {t: text for t, text in list(data.items())[:int(first_train_size * len(data))]}
            val_data = {t: text for t, text in list(data.items())[-int(test_size * len(data))-int(val_size * len(data)):
                                                                  -int(test_size * len(data))]}
            print(f"Training on {len(train_data)} documents")
        else:
            train_data = data


    train_texts, train_labels = _make_texts(train_data, unused, out_path, conversion_dict=conversion_dict, vectors=vectors)

    encoder = LabelEncoder()

    val_texts, val_labels = _make_texts(val_data, unused, out_path, conversion_dict=conversion_dict, vectors=vectors)
    if vectors is None:
        encoder.fit(all_labels)
        joblib.dump(encoder, out_path + '/label_encoder.pkl')
        val_labels = encoder.transform(val_labels)
        train_labels = encoder.transform(train_labels)

    print("made data")
    print(f"*************** {out_path.split('/')[-1]} **************")

    distilroberta = out_path.split('/')[-1].find("distil") >= 0
    if distilroberta:
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    print("made encodings")

    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)

    if return_data:
        return train_dataset, val_dataset

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=5,              # total number of training epochs
        learning_rate=5e-5,
        per_device_train_batch_size=8 if distilroberta else 4,  # batch size per device during training
        per_device_eval_batch_size=8 if distilroberta else 4,   # batch size for evaluation
        gradient_accumulation_steps=1 if distilroberta else 2,
        # learning_rate=5e-5,
        # per_device_train_batch_size=16,  # batch size per device during training
        # per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,

    )

    p_type = "multi_label_classification" if vectors is None else \
        "regression" if not matrix else None
    n_labels = len(encoder.classes_) if vectors is None else \
        len(vectors[1][0]) if not matrix else len(vectors[1][0]) ** 2
    if out_path.split('/')[-1].find("distil") >= 0:
        model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base",
                                                               cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                               num_labels=n_labels, problem_type=p_type)
    else:
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base",
                                                               cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                               num_labels=n_labels, problem_type=p_type)
    model.to(dev)
    print("Training")

    if vectors is None:
        _Trainer = Trainer
        trainer = _Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,
        )
    elif not matrix:
        _Trainer = VectorTrainer
        trainer = _Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics2,
            vectors=vectors[1]
        )
    else:
        _Trainer = MatrixTrainer
        trainer = _Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics3,
            vectors=vectors[1]
        )

    print(_Trainer.__name__)
    trainer.train()
    model.save_pretrained(out_path)

    # if return_data:
    return train_dataset, val_dataset


def preprocess_examples(texts, tokenizer, full_labels, data_path):
    # Check that it can be done in one shot!
    batch_size = 1
    _batch = 0
    encoded_data = {}
    for j in range(0, len(texts), batch_size):
        _batch += 1
        try:
            with open(data_path + f'/batch{_batch}.json', 'r') as infile:
                encoded_data = json.load(infile)
                continue
        except IOError as err:
            continue
            pass

        print("#batch: ", _batch)
        sys.stdout.flush()

        _texts = texts[j: j + batch_size]
        first_sentences = [[text] * len(full_labels) for text in _texts]
        header = "The current location is: "
        second_sentences = [[f"{header}{fl}" for fl in full_labels] for _ in _texts]
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        for k, v in tokenized_examples.items():
            # encoded_data[k] = encoded_data.get(k, []) + [v[i: i + len(full_labels)] for i in range(0, len(v), len(full_labels))]
            encoded_data[k] = [v[i: i + len(full_labels)] for i in range(0, len(v), len(full_labels))]
        with open(data_path + f'/batch{_batch}.json', 'w') as outfile:
            json.dump(encoded_data, outfile)


    # return {k: [v[i: i + len(full_labels)] for i in range(0, len(v), len(full_labels))] for k, v in tokenized_examples.items()}
    # return encoded_data


def train_multiple_choice(data_path, return_data=False, out_path=None, first_train_size=0.4, val_size=0.1, test_size=0.1,):
    from sklearn.model_selection import train_test_split
    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    from datasets import load_metric
    from sklearn.preprocessing import LabelEncoder
    import joblib

    with open(data_path + 'loc_description_dict.json', 'r') as infile:
        desc_dict = json.load(infile)

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("\n********* Training model on the GPU ***************")
    else:
        dev = torch.device("cpu")
        print("\nTraining model on the CPU")

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # with open(data_path + "locs_w_cat.json", 'r') as infile:
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
        id2label = _make_texts(data, [], out_path, desc_dict=desc_dict)[1][1]
        label2id = {l: i for i, l in enumerate(id2label)}
        with open(out_path + '/label2id.json', 'w') as outfile:
            json.dump(label2id, outfile)
        train_data = {t: text for t, text in list(data.items())[:int(first_train_size * len(data))]}
        val_data = {t: text for t, text in list(data.items())[-int(test_size * len(data))-int(val_size * len(data)):
                                                              -int(test_size * len(data))]}
        print(f"Training on {len(train_data)} documents")
        sys.stdout.flush()

    with open(data_path + "sf_unused5.json", 'r') as infile:
        unused = json.load(infile) + ['45064']


    train_texts, (train_full_labels, train_labels) = _make_texts(train_data, unused, out_path, desc_dict=desc_dict)
    train_labels = [label2id[l] for l in train_labels]
    val_texts, (val_full_labels, val_labels) = _make_texts(val_data, unused, out_path, desc_dict=desc_dict)
    val_labels = [label2id[l] for l in val_labels]
    print("made data")
    print(f"*************** {out_path.split('/')[-1]} **************")
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")

    preprocess_examples(texts=train_texts, tokenizer=tokenizer, full_labels=train_full_labels, data_path=data_path + "/full_loc_train")
    train_encodings = None
    preprocess_examples(texts=val_texts, tokenizer=tokenizer, full_labels=val_full_labels, data_path=data_path + "/full_loc_val")
    val_encodings = None
    # train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    # val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    print("made encodings")
    sys.stdout.flush()

    # label_batch_size = 16
    train_dataset = Dataset1(data_path=data_path + "/full_loc_train", labels=train_labels)
    val_dataset = Dataset1(data_path=data_path + "/full_loc_val", labels=val_labels)

    if return_data:
        return train_dataset, val_dataset

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=5,              # total number of training epochs
        learning_rate=5e-5,
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=1,   # batch size for evaluation
        gradient_accumulation_steps=8,
        # learning_rate=5e-5,
        # per_device_train_batch_size=16,  # batch size per device during training
        # per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    model = AutoModelForMultipleChoice.from_pretrained("distilroberta-base",
                                                               cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    model.to(dev)
    print("Training")

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(out_path)

    # if return_data:
    return train_dataset, val_dataset


def print_scores(preds, y):
    print("Accuracy: ", accuracy_score(y, preds))
    print("Balanced Accuracy: ", balanced_accuracy_score(y, preds))
    print("F1 macro: ", f1_score(y, preds, average='macro'))

def evaluate(model_path, val_dataset):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import joblib

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")

    encoder = joblib.load(model_path + "/label_encoder.pkl")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                               num_labels=len(encoder.classes_)).to(dev)

    model.eval()
    val_labels = val_dataset.labels[:]
    val_real = encoder.inverse_transform(val_labels)
    val_encodings = val_dataset.encodings
    val_texts = tokenizer.batch_decode(val_encodings['input_ids'][:], skip_special_tokens=True)
    tensors = torch.split(torch.tensor(val_encodings['input_ids'][:], dtype=torch.long), 1)
    preds = np.array([model(t.to(dev)).logits.detach().cpu().numpy().ravel() for t in tensors])  # should be a 2-d array
    pred_labels = np.argmax(preds, axis=1).astype(int)
    preds = encoder.inverse_transform(pred_labels)

    print_scores(preds=pred_labels, y=val_labels)

    t_p_r = [[text, pred, real] for text, pred, real in zip(val_texts, preds, val_real)]
    with open(model_path + 'preds.json', 'w') as outfile:
        json.dump(t_p_r, outfile)
    print(preds[:50])


# **************** generate and evaluate ******************

def greedy_decode(model, tokenizer, encoder, test_data, only_loc=False, only_text=False, conversion_dict=None):
    """

    :param model:
    :param texts: list of formatted texts for a testimony
    :param labels: real labels (for evaluation)
    :return:
    """
    texts, labels = _make_texts(test_data, unused=[], out_path="1", conversion_dict=conversion_dict)  # only current text
    model.eval()
    ll = 0.
    with torch.no_grad():
        output_sequence = []
        prev_text = ""
        _locs = ["START", "START"]
        for text in texts:
            if only_loc:
                t = " [SEP] ".join([_locs[-2], _locs[-1]])
            elif only_text:
                t = text
            else:
                t = " [SEP] ".join([_locs[-2], prev_text, _locs[-1]] + [text])
            encoding = tokenizer(t, truncation=True, padding=True)
            # encoding = torch.split(torch.tensor(encoding['input_ids'], dtype=torch.long), 1)
            encoding = torch.tensor(encoding['input_ids'], dtype=torch.long).to(dev)
            prediction = model(encoding.unsqueeze(0))
            _probs = torch.log_softmax(prediction.logits[0].detach().cpu(), dim=-1).numpy()
            # p = prediction.logits[0].detach().cpu().numpy()
            output_sequence.append(int(np.argmax(_probs)))
            ll += _probs[output_sequence[-1]]
            _locs = [_locs[-1]] + list(encoder.inverse_transform(output_sequence[-1:]))
            prev_text = text
    print(f"Greedy likelihood score: {ll}")
    return output_sequence, encoder.transform(labels)

def beam_decode(model, tokenizer, encoder, test_data, k=3, only_loc=False, only_text=False, conversion_dict=None):
    """

    :param model:
    :param texts: list of formatted texts for a testimony
    :param labels: real labels (for evaluation)
    :return:
    """
    #fix!!!
    texts, labels = _make_texts(test_data, unused=[], out_path="1", conversion_dict=conversion_dict)  # only current text
    model.eval()
    with torch.no_grad():
        #start with an empty sequence with zero score
        # output_sequences = [([], 0)]
        output_sequences = [(["START", "START"], 0)]

        prev_text = ""
        # _locs = ["START", "START"]
        for text in texts:
            new_sequences = []

            for old_seq, old_score in output_sequences:
                if only_loc:
                    t = " [SEP] ".join([old_seq[-2], old_seq[-1]])
                elif only_text:
                    t = text
                else:
                    t = " [SEP] ".join([old_seq[-2], prev_text, old_seq[-1]] + [text])
                encoding = tokenizer(t, truncation=True, padding=True)
                # encoding = torch.split(torch.tensor(encoding['input_ids'], dtype=torch.long), 1)
                encoding = torch.tensor(encoding['input_ids'], dtype=torch.long).to(dev)
                prediction = model(encoding.unsqueeze(0))
                _probs = torch.log_softmax(prediction.logits[0].detach().cpu(), dim=-1).numpy()

                for char_index in range(len(_probs)):
                    new_seq = old_seq + encoder.inverse_transform([char_index]).tolist()
                    #considering log-likelihood for scoring
                    new_score = old_score + _probs[char_index]
                    # new_score = old_score + np.log(_probs[char_index])
                    new_sequences.append((new_seq, new_score))

            #sort all new sequences in the decreasing order of their score
            output_sequences = sorted(new_sequences, key=lambda val: val[1], reverse=True)
            #select top-k based on score
            # *Note- best sequence is with the highest score
            output_sequences = output_sequences[:k]

            # _locs = [_locs[-1]] + list(encoder.inverse_transform(output_sequence[-1:]))
            prev_text = text
    output_sequences = [(os[0][2:], os[1]) for os in output_sequences]
    print(f"Top beam score: {output_sequences[0][1]}")
    return output_sequences, encoder.transform(labels)


class LocCRF:
    def __init__(self, model_path, model_path2=None, use_prior='', conversion_dict=None, vectors=None):
        from TorchCRF import CRF
        self.model_path = model_path
        self.model_path2 = model_path2
        self.conversion_dict = conversion_dict
        self.vectors = vectors
        if vectors is None:
            self.encoder = joblib.load(model_path + "/label_encoder.pkl")
            self.classes = self.encoder.classes_
            self.start_id = self.classes.index("START")
        else:
            self.classes = vectors[0]
            self.start_id = self.encoder.transform(["START"])[0]

        distilroberta = model_path.split('/')[-1].find("distil") >= 0
        if distilroberta:
            self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base",
                                                      cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base",
                                                      cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")

        # self.p_model = None
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                                        num_labels=len(self.encoder.classes_)).to(dev)
        if model_path2 is not None:
            self.model2 = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                             cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                                             num_labels=len(self.encoder.classes_)).to(dev2)

        pad_idx = None
        const = None
        pad_idx_val = None
        self.prior = use_prior
        if model_path2 is not None:
            self.prior = self.prior + "_trans"
        if use_prior == 'I':
            pad_idx = np.arange(len(self.classes))
        if use_prior == 'I1':
            pad_idx = np.arange(len(self.classes))
            pad_idx_val = -1.
        if use_prior == 'const':
            const = 0.1
        self.crf = CRF(len(self.classes), pad_idx=pad_idx, pad_idx_val=pad_idx_val, const=const).to(dev)
        with torch.no_grad():
            self.crf.trans_matrix[:, self.start_id] = -10000.
            self.crf.trans_matrix[self.start_id, self.start_id] = 0.


        # set self.crf.trans_matrix ?
        # self.crf.trans_matrix = nn.Parameter(torch.diag(torch.ones(len(self.encoder.classes_))))
        # set by pairwise distance??


    def _forward(self, texts):
        """

        :param texts: of shape (batch_size, num_segments)
        :return:
        """
        self.model.to(dev)
        # hidden = np.zeros((len(texts), max([len(t) for t in texts]), len(self.encoder.classes_)))
        hidden = torch.zeros(len(texts), max([len(t) for t in texts]), len(self.classes)).to(dev)

        if self.model_path2 is not None:
            self.model2.to(dev2)
            hidden2 = torch.zeros(len(texts), max([len(t) for t in texts]), len(self.classes),
                                  len(self.classes)).to(dev2)

        if len(texts) == 0:
            return torch.from_numpy(hidden).to(dev)
        for k, text in tqdm.tqdm(enumerate(texts), desc="Make encodings", leave=False):
            # print("text")
            # print(text)
            # encodings = [torch.tensor(self.tokenizer(t, truncation=True, padding=True)['input_ids'], dtype=torch.long).to(dev) for t in text]
            encodings = torch.tensor(self.tokenizer(text, truncation=True, padding=True)['input_ids'], dtype=torch.long).to(dev)

            if self.model_path2 is not None and self.vectors is None:
                cats = self.encoder.classes_
                encodings2 = [torch.tensor(self.tokenizer([_text + " [SEP] " + c for _text in text], truncation=True,
                                                          padding=True)['input_ids'], dtype=torch.long).to(dev2)
                              for c in cats]
                # print(encodings2[0])
            # print(encodings)
            # probs = [self.model(e.unsqueeze(0)).logits[0].detach().cpu().numpy() for e in encodings]
            # print([self.model(e.unsqueeze(0)).logits[0].detach() for e in encodings])
            # probs = np.array([self.model(e.unsqueeze(0).to(dev)).logits[0].detach().cpu().numpy() for e in encodings])
            for j, e in enumerate(encodings):
                # probs = self.model(e.unsqueeze(0).to(dev)).logits[0].detach()  #????
                if self.vectors is None:
                    probs = self.model(e.unsqueeze(0).to(dev)).logits[0]  #????
                else:
                    outputs = model(**encodings)
                    logits = outputs.get('logits')  # of shape (batch_size, vec_dim)?
                    # should be of shape (batch_size, num_labels)
                    probs = util.cos_sim(logits, self.vectors).softmax(dim=-1)

                    outputs2 = model2(**encodings)
                    logits2 = outputs2.get('logits')  # of shape (batch_size, vec_dim**2)?

                    # here we work with a single input
                    out2 = logits.reshape(int(logits2.shape[-1] ** .5), int(logits2.shape[-1] ** .5)) @ self.vectors.T
                    # sims should be a tensor of shape (num_classes)
                    probs2 = util.cos_sim(self.vectors, out2).softmax(dim=1)
                    hidden2[k, j, :, :] = probs2

                hidden[k, j, :probs.shape[0]] = probs

            if self.model_path2 is not None and self.vectors is None:
                for _c, _encodings2 in enumerate(encodings2):  # category
                    # probs2 = torch.zeros(len(self.encoder.classes_), len(self.encoder.classes_)).to(dev2)
                    for j in range(0, len(_encodings2), 8):  # location in testimony
                        probs2 = self.model2(_encodings2[j: j+8].to(dev2)).logits
                        # print(probs2.shape)
                        hidden2[k, j: j+probs2.shape[0], _c, :] = probs2

                    # for j, e in enumerate(_encodings2):  # location in testimony
                    #     probs2 = self.model2(e.unsqueeze(0).to(dev2)).logits[0]
                        # hidden2[k, j, _c, :] = probs2

            # probs = self.model(encodings).logits.detach().cpu().numpy()
            # hidden[k, :probs.shape[0], :probs.shape[1]] = probs
        # self.model.cpu()
        # hidden = torch.from_numpy(hidden).to(dev)
        # hidden.requires_grad_()
        # if self.model_path2 is not None:
        #     hidden2.requires_grad_()
        if self.model_path2 is not None:
            return hidden, hidden2
        return hidden

    def eval_loss(self, test_data=None):
        _eval = list(test_data.values())
        losses = []
        sys.stdout.flush()
        batch_size = 1
        self.model.eval()
        if self.model_path2 is not None:
            self.model2.eval()
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(_eval)), desc="Eval loss"):
                # for t, t_data in tqdm.tqdm(list(train_data.items())[:int(len(train_data) * 0.9)]):
                eval_batch = _eval[i: i+batch_size]
                _batch_size = len(eval_batch)

                texts = []
                label_mask = np.zeros((_batch_size, max([len(t) for t in eval_batch])), dtype=bool)
                labels = np.zeros((_batch_size, max([len(t) for t in eval_batch])), dtype=int)

                for j, _t in enumerate(eval_batch):
                    _texts, _labels = _make_texts({1: _t}, unused=[], out_path="1", conversion_dict=self.conversion_dict)  # only current text
                    label_mask[j, :len(_t)] = 1
                    labels[j, :len(_t)] = self.encoder.transform(_labels)
                    texts.append(_texts)
                    # labels.append(_labels)

                # hidden = self._forward(texts).detach()
                if self.model_path2 is not None:
                    hidden, hidden2 = self._forward(texts)
                    hidden2 = hidden2.to(dev)
                else:
                    hidden, hidden2 = self._forward(texts), None

                mask = torch.from_numpy(label_mask).to(dev)
                labels = torch.from_numpy(labels).to(dev)

                # self.crf.eval()
                loss = -self.crf.forward(hidden, labels, mask, h2=hidden2).mean()
                # self.crf.train()
                losses.append(loss.item())
                # print(loss.item())
        print("Eval loss:", np.mean(losses))
        wandb.log({'Eval loss': np.mean(losses)})

    def train(self, train_data, batch_size=4, epochs=10, test_dict=None, test_data=None, accu_grad=1):
        print(f"Training. Batch size: {batch_size}, epochs: {epochs}")
        print(f"Accu Grad: {accu_grad}")
        self.prior = self.prior + f"_b{batch_size}_e{epochs}"
        import random
        import torch.optim as optim
        # import torch.nn as nn
        # self.model.eval()  # don't train model
        # print("******************* leaving only layer 11.output unfrozen *******************")
        # print("******************* leaving only layer 11 unfrozen!!!!!!!!! *******************")
        print("******************* leaving all layer 5 unfrozen!!!!!!!!! *******************")
        # print("******************* all layers frozen (including classifier) ******************")
        # self.model.train()
        self.model.train()
        for name, param in self.model.named_parameters():
            # if name.find("11.output") == -1 and name.find("11.intermediate") == -1 and name.find("classifier") == -1: # choose whatever you like here
            # if name.find("11.output") == -1 and name.find("11.intermediate") == -1 and name.find("classifier") == -1:
            if name.find("5") == -1 and name.find("classifier") == -1 and name.find("pooler") == -1:
                # if name.find("classifier") == -1:  # choose whatever you like here
                param.requires_grad = False
            else:
                param.requires_grad = True
                # param.requires_grad = False
        if self.model_path2 is not None:
            self.model2.train()
            for name, param in self.model2.named_parameters():
                # if name.find("11.output") == -1 and name.find("classifier") == -1:
                if name.find("5") == -1 and name.find("classifier") == -1 and name.find("pooler") == -1:
                # if name.find("classifier") == -1:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # self.p_model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

        if self.model_path2 is None:
            optimizer = optim.Adam(list(self.crf.parameters()) + list(self.model.parameters()))
        else:
            optimizer = optim.Adam(list(self.crf.parameters()) + list(self.model.parameters()) +
                                   list(self.model2.parameters()))
        optimizer.zero_grad()

        # criterion = nn.CrossEntropyLoss()
        # self.crf = self.crf.to(dev)
        # criterion = criterion.to(dev)

        _train = list(train_data.values())  # list of lists
        # _train = list(train_data.items())[:int(len(train_data) * 0.9)]

        # first eval
        print("With random transitions")
        name = self.save_model(epoch=0)
        if self.model_path2 is not None:
            crf_eval(data_path=test_dict['data_path'], model_path=test_dict['model_path'], model_path2=test_dict['model_path2'],
                     test_size=test_dict['test_size'], val_size=test_dict['val_size'], name=name)
        else:
            crf_eval(data_path=test_dict['data_path'], model_path=test_dict['model_path'],
                     test_size=test_dict['test_size'], val_size=test_dict['val_size'], name=name)

        for e in range(epochs):
            self.model.train()
            if self.model_path2 is not None:
                self.model2.train()
            losses = []
            print("\n" + str(e))
            sys.stdout.flush()
            random.shuffle(_train)
            for i in tqdm.tqdm(range(0, len(_train), batch_size), desc="Train"):
                # for t, t_data in tqdm.tqdm(list(train_data.items())[:int(len(train_data) * 0.9)]):
                train_batch = _train[i: i+batch_size]
                _batch_size = len(train_batch)

                texts = []
                label_mask = np.zeros((_batch_size, max([len(t) for t in train_batch])), dtype=bool)
                labels = np.zeros((_batch_size, max([len(t) for t in train_batch])), dtype=int)

                for j, _t in enumerate(train_batch):
                    _texts, _labels = _make_texts({1: _t}, unused=[], out_path=self.model_path, conversion_dict=self.conversion_dict)  # only current text
                    label_mask[j, :len(_t)] = 1
                    labels[j, :len(_t)] = self.encoder.transform(_labels)
                    texts.append(_texts)
                    # labels.append(_labels)

                if self.model_path2 is not None:
                    hidden, hidden2 = self._forward(texts)
                    hidden2 = hidden2.to(dev)
                else:
                    hidden, hidden2 = self._forward(texts), None

                mask = torch.from_numpy(label_mask).to(dev)
                # mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size. sequence_size)
                # print(label)
                labels = torch.from_numpy(labels).to(dev)
                # labels = torch.LongTensor(self.encoder.transform(labels)).unsqueeze(0).to(dev)  # (batch_size, sequence_size)

                loss = -self.crf.forward(hidden, labels, mask, h2=hidden2).mean() / accu_grad
                # loss = criterion(y_pred, label)
                loss.backward()
                if accu_grad == 1 or (i+1) % accu_grad == 0 or i+accu_grad >= len(_train) or _batch_size < batch_size:
                    optimizer.step()
                    optimizer.zero_grad()

                losses.append(loss.item())
                wandb.log({'loss': loss})
                # print(loss.item())
            print("Epoch loss:", np.mean(losses))
            wandb.log({'Epoch loss': np.mean(losses)})
            if test_data is not None:
                self.eval_loss(test_data)
            if e % 3 == 0:
                name = self.save_model(epoch=e)
                crf_eval(data_path=test_dict['data_path'], model_path=test_dict['model_path'],
                         test_size=test_dict['test_size'], val_size=test_dict['val_size'], name=name)

        return self

    def decode(self, test_data):

        texts, labels = _make_texts(test_data, unused=[], out_path=self.model_path, conversion_dict=self.conversion_dict)  # only current text
        with torch.no_grad():
            self.model.eval()
            if self.model_path2 is None:
                hidden, hidden2 = self._forward([texts]), None
                # hidden = self._forward([texts])
                hidden.detach()
                mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size. sequence_size)

                return self.crf.viterbi_decode(hidden, mask, h2=hidden2)[0], labels  # predictions and labels
            else:
                self.model2.eval()
                hidden, hidden2 = self._forward([texts])
                hidden2.detach()
                mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size. sequence_size)
                return self.crf.viterbi_decode(hidden, mask, h2=hidden2.to(dev))[0], labels  # predictions and labels

    def save_model(self, epoch=None):
        if epoch is not None:
            print("Saving model " + f'/crf_{self.prior[:-3]}e{epoch}.pkl')
            joblib.dump(self, self.model_path + f'/crf_{self.prior[:-3]}e{epoch}.pkl')
            return f'/crf_{self.prior[:-3]}e{epoch}.pkl'
        else:
            print("Saving model " + f'/crf_{self.prior}.pkl')
            joblib.dump(self, self.model_path + f'/crf_{self.prior}.pkl')
            return f'/crf_{self.prior}.pkl'


def _eval(pred, labels):

    ed, sm = edit_distance(pred, labels), gestalt_diff(pred, labels)
    # print(len(pred), len(labels))
    # print(pred)
    # print(labels)
    acc = accuracy_score(y_pred=pred, y_true=labels)
    trans_pred = np.array(pred[:-1]) == np.array(pred[1:])
    trans_labels = np.array(labels[:-1]) == np.array(labels[1:])
    f1 = precision_recall_fscore_support(y_true=trans_labels, y_pred=trans_pred, average="binary")[2]

    # print("Edit distance: ", ed)
    # print("Sequence Matching: ", sm)
    return ed, sm, acc, f1


def decode(data_path, model_path, only_loc=False, only_text=False, val_size=0.1, test_size=0.1, c_dict=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import joblib

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")

    encoder = joblib.load(model_path + "/label_encoder.pkl")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                               num_labels=len(encoder.classes_)).to(dev)

    with open(data_path + "sf_unused5.json", 'r') as infile:
        unused = json.load(infile) + ['45064']
    with open(data_path + "sf_five_locs.json", 'r') as infile:
        unused = unused + json.load(infile)
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
        for u in unused:
            data.pop(u, None)

    # with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
    #     data = json.load(infile)

    if test_size > 0:
        test_data = {t: text for t, text in list(data.items())[-int(test_size*len(data)) - int(val_size*len(data)):
                                                               -int(test_size*len(data))]}  # !!!!
    else:
        test_data = {t: text for t, text in list(data.items())[-5:]}  #!!!!

    eds_g, sms_g, accs_g, f1s_g = [], [], [], []
    eds_b, sms_b, accs_b, f1s_b = [], [], [], []

    for t, t_data in test_data.items():
        pred, labels = greedy_decode(model, tokenizer, encoder, test_data={t: t_data}, only_loc=only_loc,
                                     only_text=only_text, conversion_dict=c_dict)
        # print(encoder.inverse_transform(pred[:10]))
        # print(encoder.inverse_transform(labels[:10]))
        ed_g, sm_g, acc_g, f1_g = _eval(pred, labels)
        preds, labels = beam_decode(model, tokenizer, encoder, test_data={t: t_data}, k=3, only_loc=only_loc,
                                    only_text=only_text, conversion_dict=c_dict)
        ed_b, sm_b, acc_b, f1_b = _eval(encoder.transform(preds[0][0]), labels)
        # print(preds[0][0][:10])
        # print(encoder.inverse_transform(labels[:10]))

        eds_g.append(ed_g / len(labels))
        eds_b.append(ed_b / len(labels))
        sms_g.append(sm_g)
        sms_b.append(sm_b)
        accs_g.append(acc_g)
        accs_b.append(acc_b)
        f1s_g.append(f1_g)
        f1s_b.append(f1_b)
    print("Greedy")
    print("Edit: " + str(np.mean(eds_g)))
    print("SM: " + str(np.mean(sms_g)))
    print("Accuracy: " + str(np.mean(accs_g)))
    print("F1: " + str(np.mean(f1s_g)))
    print("Beam")
    print("Edit: " + str(np.mean(eds_b)))
    print("SM: " + str(np.mean(sms_b)))
    print("Accuracy: " + str(np.mean(accs_b)))
    print("F1: " + str(np.mean(f1s_b)))
    print("Done")

def crf_decode(data_path, model_path, model_path2=None, first_train_size=0.4, val_size=0.1, test_size=0.1, use_prior='', batch_size=4,
               epochs=10, conversion_dict=None, accu_grad=1):
    """

    :param data_path:
    :param model_path:
    :param first_train_size: ratio of data used for the classifier training and not for CRF training
    :param val_size: ratio for validation
    :param test_size: ratio for test (not used now)
    :param use_prior:
    :param batch_size:
    :param epochs:
    :return:
    """

    with open(data_path + "sf_unused5.json", 'r') as infile:
        unused = json.load(infile) + ['45064']
    with open(data_path + "sf_five_locs.json", 'r') as infile:
        unused = unused + json.load(infile)
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
        for u in unused:
            data.pop(u, None)
    print(f"With {len(data)} documents")

    if test_size > 0:

        train_data = {t: text for t, text in list(data.items())[int(first_train_size*len(data)):
                                                                -int(test_size*len(data)) - int(val_size*len(data))]}
        test_data = {t: text for t, text in list(data.items())[-int(test_size*len(data)) - int(val_size*len(data)):
                                                               -int(test_size*len(data))]}  #!!!!
        test_dict = {"data_path": data_path, "model_path": model_path, "test_size": test_size, "val_size": val_size}
        if model_path2 is not None:
            test_dict["model_path2"] = model_path2
    else:
        train_data = data
    loc_crf = LocCRF(model_path, model_path2=model_path2, use_prior=use_prior,
                     conversion_dict=conversion_dict).train(train_data, batch_size=batch_size, epochs=epochs,
                                                            test_dict=test_dict, test_data=test_data, accu_grad=accu_grad)
    return loc_crf.save_model()

def crf_eval(data_path, model_path, model_path2=None, val_size=0., test_size=0., name=''):
    print(f"Eval ({name})")
    encoder = joblib.load(model_path + "/label_encoder.pkl")
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
    loc_crf = joblib.load(model_path + name)

    return_dict = {}
    if test_size > 0:
        test_data = {t: text for t, text in list(data.items())[-int(test_size*len(data)) - int(val_size*len(data)):
                                                               -int(test_size*len(data))]}  # !!!!

        eds, sms, accs, f1s = [], [], [], []
        print("CRF")
        for t, t_data in tqdm.tqdm(test_data.items(), desc="Eval"):
            pred, labels = loc_crf.decode(test_data={t: t_data})
            # print(encoder.inverse_transform(pred[:10]))
            # print(labels[:10])
            return_dict[t] = {"pred": encoder.inverse_transform(pred), "real": labels}
            ed, sm, acc, f1 = _eval(pred, encoder.transform(labels))
            eds.append(ed/len(labels))
            sms.append(sm)
            accs.append(acc)
            f1s.append(f1)
        print("Edit: " + str(np.mean(eds)))
        print("SM: " + str(np.mean(sms)))
        print("Accuracy: " + str(np.mean(accs)))
        print("F1: " + str(np.mean(f1s)))
        wandb.log({'Edit': np.mean(eds), 'SM': np.mean(sms), 'Accuracy': np.mean(accs), 'F1': np.mean(f1s)})
    print("Done")
    sys.stdout.flush()
    return return_dict


def main():

    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    data_path = base_path + 'data/'
    # make_description_category_dict(data_path)
    first_train_size = 0.8
    # prior = ''
    # prior = 'I1'
    # prior = 'const'
    # for prior in ["const", "I1", "I"]:
    model_name2 = None
    # model_name2 = "deberta6"
    # model_name2 = "distilroberta6"
    model_path2 = None
    c_dict = None
    use_vectors = 'v'
    for prior in ["I1"]:
        batch_size = 1
        accu_grad = 4
        epochs = 50
        name = f'/crf_{prior}_b{batch_size}_a{accu_grad}_e{epochs}_{use_vectors}.pkl'
        print("********************* Using MINUS log-likelihood as loss *****************")
        # print("********************* Using conversion dict *****************")
        from loc_clusters import make_loc_conversion
        c_dict = make_loc_conversion(data_path=data_path)
        print("Prior: " + prior)
        if use_vectors == "v":
            from loc_clusters import make_vectors
            vectors = make_vectors(data_path=data_path, cat=True, conversion_dict=c_dict)

        # make_loc_data(data_path, use_segments=True, with_cat=True, with_country=True)
        # make_loc_data(data_path, use_segments=False, with_cat=True, with_country=True)
        # make_loc_data(data_path, use_segments=False, with_cat=True)
        # make_time_data(data_path)
        for model_name in ["distilroberta6"]:
        # for model_name in ["distilroberta1"]:
        # for model_name in []:
        # for model_name in ["deberta"]:
            wandb.init(project="location tracking", config={"prior": prior, "batch_size": batch_size, "epochs": epochs,
                                                            "model_name": model_name})
            # for model_name in ["deberta4"]:
            # for model_name in ["deberta6"]:
            # for model_name in ["deberta11"]:
            # for model_name in ["deberta"]:
            # for model_name in []:
            if use_vectors != "":
                model_name = "v_" + model_name
            print(model_name)
            model_path = "/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/" + model_name
            if model_name2 is not None:
                print(model_name2)
                model_path2 = "/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/" + model_name2

            train_dataset, val_dataset = train_classifier(data_path, return_data=False, out_path=model_path,
                                                          first_train_size=first_train_size, val_size=0.1, test_size=0.1,
                                                          conversion_dict=c_dict, vectors=vectors,
                                                          matrix=model_name[-1] == "6")
            # train_dataset, val_dataset = train_multiple_choice(data_path, return_data=False, out_path=model_path,
            #                                                    first_train_size=first_train_size, val_size=0.1, test_size=0.1)
            # train_dataset, val_dataset = train_classifier(data_path, return_data=True, out_path=model_path)

            # first_train_size=0 mean we retrain on all documents
            # name = crf_decode(data_path, model_path, model_path2, first_train_size=0., val_size=0.1, test_size=0.1, use_prior=prior,
            #                   batch_size=batch_size, epochs=epochs, conversion_dict=c_dict, accu_grad=accu_grad)
            # name = "/crf_I1_b16_ee78.pkl"
            # d = crf_eval(data_path, model_path, val_size=0.1, test_size=0.1, name=name)
            #

            # evaluate(model_path, val_dataset=val_dataset)
    #     #
    # model_name = "deberta"
    # # # model_name = "deberta1"
    # print("Greedy and Beam ")
    # print(model_name)
    # model_path = "/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/" + model_name
    # # only_loc, only_text = False, False
    # only_loc = model_name == "deberta3"
    # only_text = model_name == "deberta1"
    # decode(data_path, model_path, val_size=0.1, test_size=0.1, only_loc=only_loc, only_text=only_text, c_dict=c_dict)
    #
    # for t, v in d.items():
    #     print("\n" + t)
    #     print("Preds:", np.array(v["pred"]))
    #     print("True:", np.array(v["real"]))


if __name__ == "__main__":
    main()
