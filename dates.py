import spacy
import json
import numpy as np
import re
import pandas as pd

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

def train_classifier(data_path, return_data=False):
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
        print("Running on the GPU")
    else:
        dev = torch.device("cpu")
        print("Running on the CPU")

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # with open(data_path + "locs_w_cat.json", 'r') as infile:
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
    with open(data_path + "sf_unused5.json", 'r') as infile:
        unused = json.load(infile) + ['45064']

    # locs = []
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
                labels.append(d[2][0])
                # locs.append(prev_loc)
                # texts.append((prev_text, prev_loc))
                text = d[0]
                texts.append(" [SEP] ".join([prev_loc[0], prev_text, prev_loc[1], text]))

                # prev_text = " ".join(d[0].split()[-100:])
                prev_text = text
                prev_loc = [prev_loc[-1], d[1]]

        # labels.append("END")
        # locs.append(prev_loc)
        # texts.append(" [SEP] ".join([prev_loc[0], prev_text, prev_loc[1], text]))
        # texts.append((prev_text, prev_loc))

    encoder = LabelEncoder()

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, encoder.fit_transform(labels), test_size=.2)
    print("made data")
    out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/deberta'
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
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,   # batch size for evaluation
        gradient_accumulation_steps=4,
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


def evaluate(model_path, dataset):
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
                                                               num_labels=len(encoder.classes_))

    model.eval()
    val_labels = dataset.labels[:30]
    val_real = encoder.inverse_transform(val_labels)
    val_encodings = dataset.encodings
    val_texts = tokenizer.batch_decode(val_encodings['input_ids'][:30], skip_special_tokens=True)
    tensors = torch.split(torch.tensor(val_encodings['input_ids'][:30], dtype=torch.long), 1)
    preds = np.array([model(t.to(dev)).logits.detach().cpu().numpy().ravel() for t in tensors])  # should be a 2-d array
    preds = encoder.inverse_transform(np.argmax(preds, axis=1).astype(int))
    t_p_r = [[text, pred, real] for text, pred, real in zip(val_texts, preds, val_real)]
    with open(model_path + 'preds.json', 'w') as outfile:
        json.dump(t_p_r, outfile)
    print(preds)




if __name__ == "__main__":
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    data_path = base_path + 'data/'
    model_path = "/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/deberta"
    # make_loc_data(data_path, use_segments=True, with_cat=True, with_country=True)
    # make_loc_data(data_path, use_segments=False, with_cat=True, with_country=True)
    # make_loc_data(data_path, use_segments=False, with_cat=True)
    # make_time_data(data_path)
    train_dataset, val_dataset = train_classifier(data_path, return_data=True)
    evaluate(model_path, dataset=val_dataset)
    pass
    # with open("C:/Users/Eitan/nlplab/raw_text.json", "r") as infile:
    #     texts = json.load(infile)
    # nlp2 = spacy.load("en_core_web_trf")
    # # t = "103"
    # ts = list(texts.keys())[:]
    # # ts = range(144, 155)
    # events = []
    # all_X, all_Y = [], []
    # all_X_n = np.array([])
    # for _t in ts:
    #     t = str(_t)
    #     doc2_2 = nlp2(texts[t])
    #
    #     sent2i = {s.text: i for i, s in enumerate(doc2_2.sents)}
    #
    #     X, Y = [], []
    #     for e in doc2_2.ents:
    #         if e.label_ == "EVENT":
    #             events.append((e.text, t, e.start))
    #         if e.label_ == "DATE":
    #             y = get_year(e)
    #             if y is not None:
    #                 # print(y, e.start)
    #                 Y.append(y)
    #                 # X.append(e.start)
    #                 X.append(sent2i[e.sent.text])
    #                 # print(e.text)
    #                 # print(e.start)
    #     print(f"\nT: {t}")
    #     # print(len(X))
    #     # print(X)
    #     # print(Y)
    #     t_len = len([s for s in doc2_2.sents])
    #     all_X = all_X + X
    #     all_Y = all_Y + Y
    #     all_X_n = np.concatenate([all_X_n, 100 * np.array(X) / t_len])
    #     # print("Sents: ", t_len)
    #     plot(X, Y, t=f"Testimony: {t}, ents: {len(X)}", t_len=t_len, save=True, t_num=_t)
    #
    # print(events)
    # plot(all_X, all_Y, t=f"All", t_len=3000, save=True, t_num="all")
    # plot(all_X_n, all_Y, t=f"All_n", t_len=100, save=True, t_num="all_n")