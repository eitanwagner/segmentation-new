import spacy
import json
import numpy as np
import re
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

import torch
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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


def _make_texts(data, unused, out_path, desc_dict=None):
    if desc_dict is not None:
        desc_dict["START"] = "The beginning of the testimony."
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
                    labels.append(d[2][0])
                else:
                    labels.append(d[1])
                # locs.append(prev_loc)
                # texts.append((prev_text, prev_loc))
                text = d[0]
                if out_path[-1] == "1":  # deberta1
                    texts.append(" [SEP] ".join([text]))
                elif out_path[-1] == "2":  # deberta2
                    texts.append(" [SEP] ".join([prev_loc[0], prev_text, prev_loc[1]]))
                elif out_path[-1] == "3":  # deberta3
                    texts.append(" [SEP] ".join([prev_loc[0], prev_loc[1]]))
                elif out_path[-1] == "4":  # deberta4
                    texts.append(" [SEP] ".join([prev_text, text]))
                elif out_path[-1] == "5" and desc_dict is not None:
                    texts.append(" [SEP] ".join([prev_loc[0] + ": " + desc_dict[prev_loc[0]],
                                                 prev_loc[1] + ": " + desc_dict[prev_loc[1]]]))
                else:
                    texts.append(" [SEP] ".join([prev_loc[0], prev_text, prev_loc[1], text]))

                # prev_text = " ".join(d[0].split()[-100:])
                prev_text = text
                prev_loc = [prev_loc[-1], d[1]]
    return texts, labels


def train_classifier(data_path, return_data=False, out_path=None, first_train_size=0.4, val_size=0.1, test_size=0.1,):
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

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # with open(data_path + "locs_w_cat.json", 'r') as infile:
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
        if test_size > 0:
            train_data = {t: text for t, text in list(data.items())[:int(first_train_size * len(data))]}
            val_data = {t: text for t, text in list(data.items())[-int(test_size * len(data))-int(val_size * len(data)):
                                                                  -int(test_size * len(data))]}
            print(f"Training on {len(train_data)} documents")
        else:
            train_data = data
    with open(data_path + "sf_unused5.json", 'r') as infile:
        unused = json.load(infile) + ['45064']

    # locs = []
    texts, labels = _make_texts(train_data, unused, out_path)
        # labels.append("END")
        # locs.append(prev_loc)
        # texts.append(" [SEP] ".join([prev_loc[0], prev_text, prev_loc[1], text]))
        # texts.append((prev_text, prev_loc))

    encoder = LabelEncoder()

    if test_size == 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, encoder.fit_transform(labels), test_size=.2, random_state=42)
    else:
        val_texts, val_labels = _make_texts(val_data, unused, out_path)
        encoder.fit(labels + val_labels)
        val_labels = encoder.transform(val_labels)
        train_texts, train_labels = texts, encoder.transform(labels)
    print("made data")
    print(f"*************** {out_path.split('/')[-1]} **************")
    # out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/deberta1'
    joblib.dump(encoder, out_path + '/label_encoder.pkl')

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
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        gradient_accumulation_steps=2,
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

    model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base",
                                                               cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                               num_labels=len(encoder.classes_))
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

def greedy_decode(model, tokenizer, encoder, test_data, only_loc=False, only_text=False):
    """

    :param model:
    :param texts: list of formatted texts for a testimony
    :param labels: real labels (for evaluation)
    :return:
    """
    texts, labels = _make_texts(test_data, unused=[], out_path="1")  # only current text
    model.eval()
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
        output_sequence.append(int(np.argmax(prediction.logits[0].detach().cpu().numpy())))
        _locs = [_locs[-1]] + list(encoder.inverse_transform(output_sequence[-1:]))
        prev_text = text
    return output_sequence, encoder.transform(labels)

def beam_decode(model, tokenizer, encoder, test_data, k=3, only_loc=False, only_text=False):
    """

    :param model:
    :param texts: list of formatted texts for a testimony
    :param labels: real labels (for evaluation)
    :return:
    """
    #fix!!!
    texts, labels = _make_texts(test_data, unused=[], out_path="1")  # only current text
    model.eval()
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
    return output_sequences, encoder.transform(labels)


class LocCRF:
    def __init__(self, model_path, use_prior=''):
        from TorchCRF import CRF
        self.model_path = model_path
        self.encoder = joblib.load(model_path + "/label_encoder.pkl")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                                        num_labels=len(self.encoder.classes_)).to(dev)

        pad_idx = None
        const = None
        pad_idx_val = None
        if use_prior == 'I':
            pad_idx = np.arange(len(self.encoder.classes_))
        if use_prior == 'I1':
            pad_idx = np.arange(len(self.encoder.classes_))
            pad_idx_val = -1.
        if use_prior == 'const':
            const = 0.1
        self.crf = CRF(len(self.encoder.classes_), pad_idx=pad_idx, pad_idx_val=pad_idx_val, const=const).to(dev)
        self.prior = use_prior
        # set self.crf.trans_matrix ?
        # self.crf.trans_matrix = nn.Parameter(torch.diag(torch.ones(len(self.encoder.classes_))))
        # set by pairwise distance??


    def _forward(self, texts):
        self.model.to(dev)
        # hidden = np.zeros((len(texts), max([len(t) for t in texts]), len(self.encoder.classes_)))
        hidden = torch.zeros(len(texts), max([len(t) for t in texts]), len(self.encoder.classes_)).to(dev)
        if len(texts) == 0:
            return torch.from_numpy(hidden).to(dev)
        for k, text in enumerate(texts):
            # encodings = [torch.tensor(self.tokenizer(t, truncation=True, padding=True)['input_ids'], dtype=torch.long).to(dev) for t in text]
            encodings = torch.tensor(self.tokenizer(text, truncation=True, padding=True)['input_ids'], dtype=torch.long).to(dev)
            # print(encodings)
            # probs = [self.model(e.unsqueeze(0)).logits[0].detach().cpu().numpy() for e in encodings]
            # print([self.model(e.unsqueeze(0)).logits[0].detach() for e in encodings])
            # probs = np.array([self.model(e.unsqueeze(0).to(dev)).logits[0].detach().cpu().numpy() for e in encodings])
            for j, e in enumerate(encodings):
                probs = self.model(e.unsqueeze(0).to(dev)).logits[0].detach()
                hidden[k, j, :probs.shape[0]] = probs
            # probs = self.model(encodings).logits.detach().cpu().numpy()
            # hidden[k, :probs.shape[0], :probs.shape[1]] = probs
        # self.model.cpu()
        # hidden = torch.from_numpy(hidden).to(dev)
        hidden.requires_grad_()
        return hidden

    def eval_loss(self, test_data=None):
        _eval = list(test_data.values())
        losses = []
        sys.stdout.flush()
        batch_size = 1
        for i in tqdm.tqdm(range(len(_eval))):
            # for t, t_data in tqdm.tqdm(list(train_data.items())[:int(len(train_data) * 0.9)]):
            eval_batch = _eval[i: i+batch_size]
            _batch_size = len(eval_batch)

            texts = []
            label_mask = np.zeros((_batch_size, max([len(t) for t in eval_batch])), dtype=bool)
            labels = np.zeros((_batch_size, max([len(t) for t in eval_batch])), dtype=int)

            for j, _t in enumerate(eval_batch):
                _texts, _labels = _make_texts({1: _t}, unused=[], out_path="1")  # only current text
                label_mask[j, :len(_t)] = 1
                labels[j, :len(_t)] = self.encoder.transform(_labels)
                texts.append(_texts)
                # labels.append(_labels)

            hidden = self._forward(texts).detach()

            mask = torch.from_numpy(label_mask).to(dev)
            labels = torch.from_numpy(labels).to(dev)

            self.crf.eval()
            loss = -self.crf.forward(hidden, labels, mask).mean()
            self.crf.train()
            losses.append(loss.item())
            # print(loss.item())
        print("Eval loss:", np.mean(losses))
        wandb.log({'Eval loss': np.mean(losses)})

    def train(self, train_data, batch_size=4, epochs=10, test_dict=None, test_data=None):
        print(f"Training. Batch size: {batch_size}, epochs: {epochs}")
        self.prior = self.prior + f"_b{batch_size}_e{epochs}"
        import random
        import torch.optim as optim
        # import torch.nn as nn
        self.model.eval()  # don't train model
        optimizer = optim.Adam(self.crf.parameters())
        # criterion = nn.CrossEntropyLoss()
        # self.crf = self.crf.to(dev)
        # criterion = criterion.to(dev)

        _train = list(train_data.values())  # list of lists
        # _train = list(train_data.items())[:int(len(train_data) * 0.9)]

        # first eval
        print("With random transitions")
        name = self.save_model(epoch=0)
        crf_eval(data_path=test_dict['data_path'], model_path=test_dict['model_path'],
                 test_size=test_dict['test_size'], val_size=test_dict['val_size'], name=name)

        for e in range(epochs):
            losses = []
            print("\n" + str(e))
            sys.stdout.flush()
            random.shuffle(_train)
            for i in tqdm.tqdm(range(0, len(_train), batch_size)):
                # for t, t_data in tqdm.tqdm(list(train_data.items())[:int(len(train_data) * 0.9)]):
                train_batch = _train[i: i+batch_size]
                _batch_size = len(train_batch)

                texts = []
                label_mask = np.zeros((_batch_size, max([len(t) for t in train_batch])), dtype=bool)
                labels = np.zeros((_batch_size, max([len(t) for t in train_batch])), dtype=int)

                for j, _t in enumerate(train_batch):
                    _texts, _labels = _make_texts({1: _t}, unused=[], out_path=self.model_path)  # only current text
                    label_mask[j, :len(_t)] = 1
                    labels[j, :len(_t)] = self.encoder.transform(_labels)
                    texts.append(_texts)
                    # labels.append(_labels)

                optimizer.zero_grad()
                hidden = self._forward(texts)

                mask = torch.from_numpy(label_mask).to(dev)
                # mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size. sequence_size)
                # print(label)
                labels = torch.from_numpy(labels).to(dev)
                # labels = torch.LongTensor(self.encoder.transform(labels)).unsqueeze(0).to(dev)  # (batch_size, sequence_size)

                loss = -self.crf.forward(hidden, labels, mask).mean()
                # loss = criterion(y_pred, label)
                loss.backward()
                optimizer.step()
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

        texts, labels = _make_texts(test_data, unused=[], out_path=self.model_path)  # only current text
        self.model.eval()

        hidden = self._forward([texts])
        hidden.detach()
        mask = torch.ones((1, len(labels)), dtype=torch.bool).to(dev)  # (batch_size. sequence_size)

        return self.crf.viterbi_decode(hidden, mask)[0], labels  # predictions and labels

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
    from evaluation import edit_distance, gestalt_diff
    ed, sm = edit_distance(pred, labels), gestalt_diff(pred, labels)
    # print("Edit distance: ", ed)
    # print("Sequence Matching: ", sm)
    return ed, sm


def decode(data_path, model_path, only_loc=False, only_text=False, val_size=0.1, test_size=0.1):
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

    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)

    if test_size > 0:
        test_data = {t: text for t, text in list(data.items())[-int(test_size*len(data)) - int(val_size*len(data)):
                                                               -int(test_size*len(data))]}  # !!!!
    else:
        test_data = {t: text for t, text in list(data.items())[-5:]}  #!!!!

    eds_g, sms_g = [], []
    eds_b, sms_b = [], []

    for t, t_data in test_data.items():
        pred, labels = greedy_decode(model, tokenizer, encoder, test_data={t: t_data}, only_loc=only_loc, only_text=only_text)
        # print(encoder.inverse_transform(pred[:10]))
        # print(encoder.inverse_transform(labels[:10]))
        ed_g, sm_g = _eval(pred, labels)
        preds, labels = beam_decode(model, tokenizer, encoder, test_data={t: t_data}, k=3, only_loc=only_loc, only_text=only_text)
        ed_b, sm_b = _eval(encoder.transform(preds[0][0]), labels)
        # print(preds[0][0][:10])
        # print(encoder.inverse_transform(labels[:10]))

        eds_g.append(ed_g / len(labels))
        eds_b.append(ed_b / len(labels))
        sms_g.append(sm_g)
        sms_b.append(sm_b)
    print("Greedy")
    print("Edit: " + str(np.mean(eds_g)))
    print("SM: " + str(np.mean(sms_g)))
    print("Beam")
    print("Edit: " + str(np.mean(eds_b)))
    print("SM: " + str(np.mean(sms_b)))
    print("Done")

def crf_decode(data_path, model_path, first_train_size=0.4, val_size=0.1, test_size=0.1, use_prior='', batch_size=4, epochs=10):
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
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)

    if test_size > 0:

        train_data = {t: text for t, text in list(data.items())[int(first_train_size*len(data)):
                                                                -int(test_size*len(data)) - int(val_size*len(data))]}
        test_data = {t: text for t, text in list(data.items())[-int(test_size*len(data)) - int(val_size*len(data)):
                                                               -int(test_size*len(data))]}  #!!!!
        test_dict = {"data_path": data_path, "model_path": model_path, "test_size": test_size, "val_size": val_size}
    else:
        train_data = data
    loc_crf = LocCRF(model_path, use_prior=use_prior).train(train_data, batch_size=batch_size, epochs=epochs,
                                                            test_dict=test_dict, test_data=test_data)
    return loc_crf.save_model()

def crf_eval(data_path, model_path, val_size=0., test_size=0., name=''):
    print(f"Eval ({name})")
    encoder = joblib.load(model_path + "/label_encoder.pkl")
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
    loc_crf = joblib.load(model_path + name)

    if test_size > 0:
        test_data = {t: text for t, text in list(data.items())[-int(test_size*len(data)) - int(val_size*len(data)):
                                                               -int(test_size*len(data))]}  # !!!!

        eds, sms = [], []
        print("CRF")
        for t, t_data in test_data.items():
            pred, labels = loc_crf.decode(test_data={t: t_data})
            # print(encoder.inverse_transform(pred[:10]))
            # print(labels[:10])
            ed, sm = _eval(pred, encoder.transform(labels))
            eds.append(ed/len(labels))
            sms.append(sm)
        print("Edit: " + str(np.mean(eds)))
        print("SM: " + str(np.mean(sms)))
        wandb.log({'Edit': np.mean(eds), 'SM': np.mean(sms)})
    print("Done")
    sys.stdout.flush()


if __name__ == "__main__":
    import sys
    import wandb

    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    data_path = base_path + 'data/'
    # prior = ''
    # prior = 'I1'
    # prior = 'const'
    for prior in ["const", "I1", "I"]:
        batch_size = 16
        epochs = 1
        name = f'/crf_{prior}_b{batch_size}_e{epochs}.pkl'
        print("********************* Using MINUS log-likelihood as loss *****************")
        print("Prior: " + prior)
        wandb.init(project="location tracking", config={"prior": prior, "batch_size": batch_size, "epochs": epochs})

        # make_loc_data(data_path, use_segments=True, with_cat=True, with_country=True)
        # make_loc_data(data_path, use_segments=False, with_cat=True, with_country=True)
        # make_loc_data(data_path, use_segments=False, with_cat=True)
        # make_time_data(data_path)
        # for model_name in ["deberta3", "deberta", "deberta1", "deberta2"]:
        # for model_name in ["deberta4"]:
        for model_name in ["deberta1"]:
            # for model_name in []:
            print(model_name)
            model_path = "/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/" + model_name
            # train_dataset, val_dataset = train_classifier(data_path, return_data=False, out_path=model_path,
            #                                               first_train_size=0.4, val_size=0.1, test_size=0.1)
            # train_dataset, val_dataset = train_classifier(data_path, return_data=True, out_path=model_path)

            name = crf_decode(data_path, model_path, first_train_size=0.4, val_size=0.1, test_size=0.1, use_prior=prior, batch_size=batch_size, epochs=epochs)
            crf_eval(data_path, model_path, val_size=0.1, test_size=0.1, name=name)

            # evaluate(model_path, val_dataset=val_dataset)
        #
    model_name = "deberta"
    # model_name = "deberta1"
    print("Greedy and Beam ")
    print(model_name)
    model_path = "/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/" + model_name
    # only_loc, only_text = False, False
    only_loc = model_name == "deberta3"
    only_text = model_name == "deberta1"
    decode(data_path, model_path, val_size=0.1, test_size=0.1, only_loc=only_loc, only_text=only_text)
