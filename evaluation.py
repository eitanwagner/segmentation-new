
import spacy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import numpy as np
from scipy.special import softmax
from gpt2 import GPT2Scorer
from nltk import metrics
import nltk
import difflib
import segeval
import json
from spacy.tokens import Doc
from spacy.vocab import Vocab
import pandas as pd
Doc.set_extension("topics", default=None, force=True)
from transitions import MC, MCClusters

import os
CACHE_DIR = "/cs/snapless/oabend/eitan.wagner/cache/"
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
os.environ['HF_METRICS_CACHE'] = CACHE_DIR
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

import joblib
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import load_metric
import logging

# if torch.cuda.is_available():
#     dev = torch.device("cuda:0")
#     logging.info("Running on the GPU")
# else:
#     dev = torch.device("cpu")
#     logging.info("Running on the CPU")



# ********************* levenshtein distance - from nltk with changes *******************

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i           # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j           # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2, transpositions=False, cor_matrix=None):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    if cor_matrix is None:

        cor = 1
    else:
        cor = (1 - cor_matrix[c1, c2]) / 2  # to be between 0 and 1
        # cor = (1 - cor_matrix[c1, c2])  # to be between 0 and 2. the average is around 1

    # substitution
    # c = lev[i - 1][j - 1] + (c1 != c2)
    c = lev[i - 1][j - 1] + cor * (c1 != c2)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    if cor_matrix is None:
        lev[i][j] = min(a, b, c, d)
    else:
        lev[i][j] = min(c, d)


def edit_distance(s1, s2, transpositions=False, cor_matrix=None):
    """
    This was modified to take correlations into account!

    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type transpositions: bool
    :rtype int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, i + 1, j + 1, s1, s2, transpositions=transpositions, cor_matrix=cor_matrix)
    return lev[len1][len2]


def gestalt_diff(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()

# ********************* calculate scores *******************

def get_per_sent(doc):
    """
    Get list of sentence segment-end labels.
    :param doc:  a spaCy doc
    :return: list with 1 for last in segment and 0 o.w., for each sentence
    """
    # converts a spacy doc with segments into a list of 0s (for no boundary after sent) and 1s (for last in segment).
    ends = [segment.end for segment in doc.spans["segments"]]
    logging.info(f"num_ends: {len(ends)}")
    # logging.info([i for i, s in enumerate(doc.spans["sents"]) if s.end in ends])
    if doc.spans.get("sents", None) is None:
        return [1 if s.end in ends else 0 for s in doc.sents]
    return [1 if s.end in ends else 0 for s in doc.spans["sents"]]

def accu_scores(pred_doc, gold_doc):
    """
    Calculate accuracy scores for the segmentation.
    :param pred_doc: spacy doc after the model's segmentation
    :param gold_doc: the reference doc
    :return: tuple of (precision, recall, f)
    """
    # assert that sentences are the same
    # logging.info(sum([(s1.start, s1.end) == (s2.start, s2.end) for s1, s2 in zip(pred_doc.spans['sents'], gold_doc.spans['sents'])]))

    y_true, y_pred = get_per_sent(gold_doc), get_per_sent(pred_doc)
    # logging.info(str(y_true))
    # logging.info(str(y_pred))
    logging.info("len true" + str(len(gold_doc.text)) + " " + str(len(gold_doc.spans['sents']))
                 + " " + str(len(gold_doc.spans["segments"])) + " " + str(sum(y_true)))
    logging.info("len pred" + str(len(pred_doc.text)) + " " + str(len(pred_doc.spans['sents']))
                 + " " + str(len(pred_doc.spans["segments"])) + " " + str(sum(y_pred)))
    # logging.info("len pred", len(pred_doc.text), len(list(pred_doc.sents)), len(pred_doc.spans["segments"]))
    # logging.info(y_true)
    # logging.info(y_pred)
    return precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='binary')[:3]

def windowdiff(pred_doc, gold_doc, k=0):
    """
    Calculate the windowdiff cost.
    :param pred_doc: spacy doc after the model's segmentation
    :param gold_doc: the reference doc
    :return: the windowdiff difference (lower is better).
    """
    # k should be half the average segment length
    if k == 0:  # not specified
        k = len(list(gold_doc.sents)) // (2 * len(gold_doc.spans["segments"]))  # use gold_doc average segment len
    y_true, y_pred = "".join([str(_y) for _y in get_per_sent(gold_doc)]), "".join([str(_y) for _y in get_per_sent(pred_doc)]),
    # logging.info(y_true)
    # logging.info(y_pred)
    return nltk.windowdiff(y_true, y_pred, k)

# put in class?
encoder_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'+'/models/xlnet-large-cased/'
encoder = joblib.load(encoder_path + "label_encoder.pkl")
with open('/cs/snapless/oabend/eitan.wagner/segmentation/' + 'data/topics.json', 'r') as infile:
    topics = json.load(infile)
encoder = LabelEncoder().fit(topics)
# encoder.classes_ = [c[:-1].strip() if c[-1] == '\xa0' else c.strip() for c in encoder.classes_]

def topics_score(pred_topics, gold_topics, method="gestalt", path='/cs/snapless/oabend/eitan.wagner/segmentation/models/xlnet-large-cased'):
    """
    Calculate cost for the topic list based on the edit distance.
    :param pred_doc: spacy doc after the model's segmentation
    :param gold_doc: the reference doc
    :return: edit distance. lower is better
    """
    # return nltk.edit_distance(encoder.transform(pred_topics), encoder.transform(gold_topics))
    # return nltk.edit_distance(pred_topics, gold_topics)
    if method == "edit":
        cor_matrix = np.load(path + "/correlation_matrix.npy")
        return edit_distance(pred_topics, gold_topics, transpositions=True, cor_matrix=None)
        # return edit_distance(pred_topics, gold_topics, transpositions=True, cor_matrix=cor_matrix)
    return gestalt_diff(pred_topics, gold_topics)

# ************************ baselines for segmentation **************************

class UniformSegmentor:
    """
    Segmentor for uniform-length segmentation
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')

    def segment(self, text, num_segments):
        """
        Segment a given text
        :param text:
        :param num_segments:
        :return: self
        """
        doc = self.nlp(text)
        doc.spans["sents"] = list(doc.sents)
        return self.segment_doc(doc, num_segments)

    def segment_doc(self, doc, num_segments):
        """
        Segment a given spaCy doc
        :param doc:
        :param num_segments:
        :return: self
        """
        sents = doc.spans["sents"]
        # logging.info("lens: ", len(list(doc.sents)), num_segments)
        sents_arr = np.arange(len(sents))
        # sents = [0 for sent in doc.sents]
        segments = np.array_split(sents_arr, num_segments)
        doc.spans["segments"] = [doc[sents[seg[0]].start:sents[seg[-1]].end] for seg in segments]
        return doc


class Gpt2Segmentor:
    """
    Segmentor based on gpt2 scores
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')
        self.scorer = GPT2Scorer()

    def segment(self, text, num_segments, window=1):
        """
        Segment a given text
        :param window: window for gpt2-scorer
        :param text:
        :param num_segments:
        :return: self
        """
        doc = self.nlp(text)
        doc.spans["sents"] = list(doc.sents)
        return self.segment_doc(doc, num_segments, window)

    def segment_doc(self, doc, num_segments, window, dynamic=False, alpha=0.8, t=None):
        """
        Segment a given spacy doc
        :param t: testimony number - for caching
        :param dynamic: wheter to use the dynamic spacing method
        :param alpha: weight for the dynamic method. If 0 then almost like uniform, and if 1 then like without the dynamic
        :param window: window for gpt2-scorer
        :param doc:
        :param num_segments:
        :return: self
        """
        if t is not None:
            self.scorer.load_cache(t)
        sents = list(doc.spans["sents"])

        diffs = []
        for j, s in enumerate(sents):
            if j < window or j + window >= len(sents):
                continue
            gpt2_p1 = self.scorer.sentence_score(doc[sents[j-window].start:sents[j+window].start].text)
            gpt2_p2 = self.scorer.sentence_score(doc[sents[j-window].start:sents[j].start].text) \
                      + self.scorer.sentence_score(doc[sents[j].start:sents[j+window].start].text)
            diffs.append((gpt2_p1 - gpt2_p2, j))  # j is the beginning?
        if t is not None:
            self.scorer.save_cache()

        # js = sorted([d[1] for d in diffs[:-num_segments+1]], reverse=True)  # this assumes num_segments>1

        if not dynamic:
            diffs.sort(reverse=False)  #????
            # diffs.sort(reverse=True)  #????
            last_js = sorted([d[1] for d in diffs[:num_segments-1]])

            doc.spans["segments"] = [doc[:sents[last_js[0]-1].end]]
            for i, j in enumerate(last_js[:-1]):
                doc.spans["segments"].append(doc[sents[j].start:sents[last_js[i+1]-1].end:])
            doc.spans["segments"].append(doc[sents[last_js[-1]].start:])

        else:
            k = num_segments
            # diffs = [-np.inf] * window + [d[0] for d in diffs] + [-np.inf] * window  # large diff is good so used -inf - No! large difference means not to divide!
            # diffs = [np.inf] * window + [d[0] for d in diffs] + [np.inf] * window  # large diff is good so used -inf - No! large difference means not to divide!
            diffs = [0.] * window + [d[0] for d in diffs] + [0.] * window  # large diff is good so used -inf - No! large difference means not to divide!
            # print(diffs)
            # diffs = softmax(diffs)
            # print(diffs)

            # print("LEN DIFFS", len(diffs))
            n = len(sents)
            # print("LEN DIFFS", n)
            L = int(n / k)
            # print(L)
            # prevs = [0]
            prevs = np.zeros((n, k-1), dtype=int)
            costs = np.zeros((n, k-1))
            costs[0, 1:] = np.inf
            for _n in range(1, n):
                for _k in range(1, k-1):
                    arr = costs[:_n, _k-1] + (1 - alpha) * abs((_n - np.arange(_n)) - L) / L  # not  like in the paper!!!
                    m = np.argmin(arr)
                    costs[_n, _k] = arr[m] - alpha * (diffs[_n]/50)
                    # costs[_n, _k] = arr[m] - alpha * (1-diffs[_n])
                    # costs[_n, _k] = arr[m] - alpha * np.exp(diffs[_n])
                    # costs[_n, _k] = arr[m] - alpha * (1 - np.exp(diffs[_n]))
                    # prevs.append(int(m))
                    prevs[_n, _k] = int(m)

            arr = costs[:n, k-2] + (1 - alpha) * abs(n - np.arange(n) - L) / L
            m = np.argmin(arr)

            # i = prevs[-1]
            i = int(m)  # best break for last. These are the beginnings
            doc.spans["segments"] = []
            # print("LEN ASSIGNMENT: ", len(doc.spans["segments"]))
            # print("LEN ASSIGNMENT: ", k)
            j = k - 2
            assignment = [i]
            # while i >= 0:
            while j > 0:
                # print("i:", i)
                # i = prevs[i]
                i = prevs[i, j]
                j -= 1
                # assignment.insert(i, 0)
                assignment.insert(0, i)
            assignment.insert(0, 0)
            assignment.append(n)  # we should get k+1 in assignment
            # print("LEN ASSIGNMENT: ", len(assignment))

            for i, j in enumerate(assignment[:-1]):
                doc.spans["segments"].append(doc[sents[j].start:sents[assignment[i+1]-1].end])
            # print("LEN ASSIGNMENT: ", len(doc.spans["segments"]))
        return doc


class NSPSegmentor:
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/'):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
        # self.model = None
        # self.tokenizer = None

        self.base_path = base_path
        self.cache = {}
        self.cache_id = None

    def save_cache(self):
        np.save(self.base_path + f"/nsp_cache{str(self.cache_id)}.npy", self.cache)

    def load_cache(self, i):
        self.cache_id = i
        try:
            self.cache = np.load(self.base_path + f"/nsp_cache{str(i)}.npy",
                                 allow_pickle='TRUE').item()
        except IOError as err:
            # except:
            pass

    def fit(self):
        return self

    def _NSP_score(self, sent1, sent2):
        """
        Calculates the probability that sent2 is the continuation of sent1
        :param sent1:
        :param sent2:
        :return:
        """
        encoding = self.tokenizer(sent1, sent2, return_tensors='pt')
        outputs = self.model(**encoding, labels=torch.LongTensor([1]))
        logits = outputs.logits
        return logits[0, 0]

    def segment_doc(self, doc, num_segments, window, t=None):
        """
        Segment a given spacy doc
        :param dynamic: wheter to use the dynamic spacing method
        :param window: window for NSP-scoring
        :param doc:
        :param num_segments:
        :param t:
        :return: self
        """
        if t is not None:
            self.load_cache(t)

        sents = doc.spans["sents"]

        diffs = []
        for j, s in enumerate(sents):
            if j < window or j + window >= len(sents):
                continue

            score = self._NSP_score(doc[sents[j-window].start:sents[j].start].text,
                                    doc[sents[j].start:sents[j+window].start].text)
            diffs.append((score, j))  # j is the beginning?
        if t is not None:
            self.save_cache()
        # print(diffs)

        # js = sorted([d[1] for d in diffs[:-num_segments+1]], reverse=True)  # this assumes num_segments>1

        diffs.sort(reverse=True)  #????
        last_js = sorted([d[1] for d in diffs[:num_segments-1]])

        doc.spans["segments"] = [doc[:sents[last_js[0]-1].end]]
        for i, j in enumerate(last_js[:-1]):
            doc.spans["segments"].append(doc[sents[j].start:sents[last_js[i+1]-1].end:])
        doc.spans["segments"].append(doc[sents[last_js[-1]].start:])

        return doc


# ************ bert IOB method  - not implemented yet ******************

def make_data(docs, context=1):
    for doc in docs:
        sents = list(doc.sents)
        iob = get_per_sent(doc)
        data = []
        for i, b in enumerate(iob):
            if i < context or i >= len(iob) - context:
                continue
            data.append((sents[i], " ".join(sents[i-context:i+context]), b))  # tuple of sent, context, label
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class BertIOBSegmentor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')
        self.model = None
        self.tokenizer = None

    def load(self):
        pass

    def fit(self, docs, context=1):
        """

        :param docs: list of reference pre-segmented spacy docs
        :return:
        """
        data = make_data(docs, context=context)

        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")

        encodings = np.array([self.tokenizer(text[0], text[1], truncation=True, padding=True) for text in data])
        labels = np.array(list(zip(*data))[2])
        train_encodings,val_encodings, train_labels, val_labels = train_test_split(encodings, labels, test_size=.2)
        logging.info("made encodings")

        train_dataset = Dataset(train_encodings, train_labels)
        val_dataset = Dataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            learning_rate=5e-5 * 1/16,
            per_device_train_batch_size=1,  # batch size per device during training
            per_device_eval_batch_size=1,   # batch size for evaluation
            # learning_rate=5e-5,
            # per_device_train_batch_size=16,  # batch size per device during training
            # per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            label_smoothing_factor=0.,
            # report_to=None,
        )

        model = RobertaForSequenceClassification.from_pretrained('roberta-large',
                                                                 cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                                 num_labels=2)
        model.to(dev)

        logging.info("Training")
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,             # evaluation dataset
            compute_metrics=compute_metrics,
        )
        trainer.train()

        out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/iob-roberta'
        model.save_pretrained(out_path)
        self.model = model
        return self

    def segment(self, text, num_segments, window=1):
        doc = self.nlp(text)
        sents = list(doc.sents)
        threshold = num_segments / len(sents)

        doc_data = make_data([self.nlp(doc)], window=window)
        # preds = np.zeros(len(sents))
        boundaries = []

        for i, d in enumerate(doc_data):
            inputs = self.tokenizer(d[0], d[1], return_tensors="pt")
            labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
            outputs = self.model(**inputs, labels=labels)
            logits = outputs.logits
            if logits[0][1] >= threshold:
                # preds[i + window] = 1
                boundaries.append(i+window)

        doc.spans["segments"] = [doc[:boundaries[0]]]
        for i, j in enumerate(boundaries[:-1]):
            doc.spans["segments"].append(doc[j:boundaries[i+1]:])
        doc.spans["segments"].append(doc[boundaries[-1]:])

        return doc


# ************************ baselines for topic assignment **************************

class TopicAssigner:
    """
    Assigns a random topic assignment (by markov chain or frequency)
    """
    def __init__(self, markov_chain=True, frequencies=None, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/',
                 name="models/transitions/mc.json"):
        """
        :param markov_chain: whether the assigner is based on MC probabilities. Otherwise by given probabilities
        :param frequencies: probilities for the topic. if None then uses uniform
        """
        if frequencies is None:
            frequencies = np.ones(len(encoder.classes_)) / len(encoder.classes_)
        self.frequencies = frequencies
        self.markov_chain = markov_chain
        if markov_chain:
            if name[-4:] == "json":
                self.mc = MC(base_path=base_path, name=name)
            elif name[-3:] == "pkl":
                with open(base_path + 'models/transitions/mcc5_iner5_iter15.pkl', 'rb') as infile:
                    self.mc = joblib.load(infile)


    def create(self, doc):
        """
        Creates an assignment for a segmented doc
        :param doc:
        :return: list of topics
        """
        k = len(doc.spans["segments"])
        if self.markov_chain:
            return list(self.mc.sample(k))
        else:
            return list(np.random.choice(len(encoder.classes_), size=k, p=self.frequencies))


# ******************** reference data ******************

def get_topic_list(data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/"):
    """
    Obtains list of topics (after conversion) for the reference data
    :param data_path:
    :return: list of all topics in the reference data
    """
    with open(data_path + 'title_w_segments.json', 'r') as infile:
        all_topics = [tw[0] for tw in json.load(infile)]
    # logging.info([e for e in enumerate(all_topics)])
    # logging.info(f"Count: {len(all_topics)}")

    topic2newtopic = pd.read_csv(data_path + 'noam_old2noam_newtopic.csv', header=None, index_col=0, squeeze=True).to_dict()
    newtopic2num = pd.read_csv(data_path + 'noam_newtopic2num.csv', header=None, index_col=0, squeeze=True).to_dict()
    num2newtopic = pd.read_csv(data_path + 'num2newtopic.csv', header=None, index_col=0, squeeze=True).to_dict()
    new_topics = [num2newtopic.get(newtopic2num.get(topic2newtopic.get(t, None), None), None) for t in all_topics]
    # logging.info(f"None count: {sum([1 for t in new_topics if t is None])}")

    return new_topics
    # new_words2topics = {w: num2newtopic.get(newtopic2num.get(t, None), None) for w, t in words2topics.items()}


def save_doc(doc, doc_num, path="/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/"):
    """
    Save the given doc
    :param doc:
    :param doc_num:
    :param path:
    :return:
    """
    doc.to_disk(path + "doc_" + str(doc_num))


def make_gold_csv(data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/", save=True):
    """
    Make spacy docs from annotated documents in csv format
    :param data_path:
    :return:
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.disable_pipes('ner')

    for i in range(111, 116):
        t = pd.read_csv(data_path + f'gold csv/testimony_{i}.csv', header=None)
        segments = list(t[:][0])
        topics = list(t[:][1])

        _docs = []
        sents = []
        num_tokens = 0
        for s in segments:
            _docs.append(nlp(s))
            sents = sents + [(s.start + num_tokens, s.end+num_tokens) for s in _docs[-1].sents]
            num_tokens += len(_docs[-1])

        doc = Doc.from_docs(list(_docs), ensure_whitespace=True)
        ends = np.cumsum([len(_d) for _d in _docs])  # does this count the whitespaces?? they shouldn't count
        starts = [0] + ends.tolist()[:-1]
        doc.spans["segments"] = [doc[s:e] for s, e in zip(starts, ends)]
        doc.spans["sents"] = [doc[s[0]:s[1]] for s in sents]
        doc._.topics = topics

        logging.info(f"#segments ({i}): " + str(len(doc.spans["segments"])))
        logging.info("#sents: " + str(len(doc.spans["sents"])))
        if save:
            save_doc(doc, doc_num=i)
    return


def make_gold_docs(data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/", save=True):
    """
    Makes gold doc as spacy docs with segments, and saves in new folder
    :param data_path:
    :return:
    """
    segment = ""
    _docs = []
    docs = {}
    t_num = -1
    sents = []
    num_tokens = 0

    # nlp = spacy.load("en_core_web_trf")
    nlp = spacy.load("en_core_web_sm")
    # nlp.disable_pipes('transformer', 'ner')
    nlp.disable_pipes('ner')

    with open(data_path + 'segmented.txt', encoding="utf-8") as infile:
        for l in infile:
            l = l.replace("\u2019", "'").replace("\u201c","\"").replace("\u201d","\"")
            if l[:3] == "***":  # segment boundary
                if segment != "":
                    _docs.append(nlp(segment.rstrip()))
                    sents = sents + [(s.start + num_tokens, s.end+num_tokens) for s in _docs[-1].sents]
                    num_tokens += len(_docs[-1])
                    # char_lens.append(len(segment.rstrip()))
                    segment = ""
                i = l.find("Testimony")
                if i != -1:
                    if t_num == -1:
                        t_num = int(l[i + len("Testimony "):i + len("Testimony ")+3])
                    if len(_docs) > 0:
                        docs[t_num] = Doc.from_docs(list(_docs), ensure_whitespace=True)
                        ends = np.cumsum([len(_d) for _d in _docs])  # does this count the whitespaces?? they shouldn't count
                        starts = [0] + ends.tolist()[:-1]
                        docs[t_num].spans["segments"] = [docs[t_num][s:e] for s, e in zip(starts, ends)]
                        docs[t_num].spans["sents"] = [docs[t_num][s[0]:s[1]] for s in sents]
                    _docs = []
                    sents = []
                    num_tokens = 0
                    t_num = int(l[i + len("Testimony "):i + len("Testimony ")+3])

            elif l.strip() != "":
                segment = segment + l.rstrip('\n') + " "

    # docs = []
    # for t in range(101, 110):
    #     doc = nlp(texts[str(t)])
    #     doc.spans["segments"] = []
    #     logging.info(len(doc.text))
    #     while start_char < len(doc.text):
    #         # for l in char_lens:
    #         l = char_lens.pop(0)
    #         logging.info(t, start_char, l)
    #         logging.info(doc.text[start_char:start_char+l])
    #         if start_char + l > len(doc.text):
    #             l = len(doc.text) - start_char
    #         doc.spans["segments"].append(doc.char_span(start_char, start_char+l, alignment_mode="contract"))
    #         # start_char = start_char + len(doc.spans["segments"][-1])
    #         start_char = start_char + l
    #     save_doc(doc, doc_num=t)
    #     start_char = 0

    # logging.info("Segment list:")
    # for segment in doc.spans["segments"]:
    #     logging.info(segment)
    # docs.append(doc)

    all_topics = get_topic_list(data_path)
    logging.info(["None!!!"+str(e[0]) for e in enumerate(all_topics) if e[1]==None])
    logging.info(len(all_topics))

    lens_notopic = []
    lens = []
    t_count = 0
    for i, doc in docs.items():
        # for s in doc.spans["segments"]:
        #     logging.info(s)
        logging.info(f"Old doc lengths (in tokens), {i}: ")
        logging.info(len(doc))
        logging.info(sum([len(s) for s in doc.spans["segments"]]))
        logging.info(sum([len(s) for s in doc.spans["sents"]]))
        doc._.topics = all_topics[t_count:t_count+len(doc.spans["segments"])]

        # doc2 = nlp(doc.text)
        # doc2.spans["segments"] = [doc2[s.start:s.end] for s in doc.spans["segments"]]
        # logging.info(f"segments:" + str(len(list(doc2.spans["segments"]))))
        # logging.info(f"sents:" + str(len(list(doc2.sents))))
        logging.info(f"sents_old generator:" + str(len(list(doc.sents))))
        logging.info(f"sents_old:" + str(len(doc.spans["sents"])))
        # logging.info(sum(get_per_sent(doc2)))
        logging.info(sum(get_per_sent(doc)))
        # save_doc(doc2, doc_num=i)

        # logging.info(all_topics[t_count:t_count+len(doc.spans["segments"])])
        logging.info(["Empty segment!!!!!" for s in doc.spans["segments"] if len(s) == 0])
        logging.info(len(all_topics[t_count:t_count+len(doc.spans["segments"])]))
        t_count += len(doc.spans["segments"])
        logging.info(f"Count: {t_count}")

        # merge same topics
        segments = [s for s in doc.spans["segments"]]
        topics = []
        for j, t in enumerate(doc._.topics):
            if j+1 < len(doc._.topics) and t == doc._.topics[j+1]:
                # segments[j:j+2] = [doc[doc.spans["segments"][j].start:doc.spans["segments"][j+1].end]]
                segments[j+1] = doc[segments[j].start:segments[j+1].end]
                segments[j] = None
            else:
                topics.append(t)
                # topics.pop(j)
        segments = [s for s in segments if s is not None]
        doc.spans["segments"] = segments
        doc._.topics = topics

        if save:
            save_doc(doc, doc_num=i)


        # lens_notopic.append([len(s) for j, s in enumerate(doc.spans["segments"]) if doc._.topics[j] == "NO_TOPIC"])
        lens_notopic = lens_notopic + [len(s) for j, s in enumerate(doc.spans["segments"]) if doc._.topics[j] == "NO_TOPIC"]
        # lens.append([len(s) for j, s in enumerate(doc.spans["segments"]) if doc._.topics[j] != "NO_TOPIC"])
        # lens = lens + [len(s) for j, s in enumerate(doc.spans["segments"]) if doc._.topics[j] != "NO_TOPIC"]
        lens = lens + [len(s) for j, s in enumerate(doc.spans["segments"])]
    logging.info(f"lens NO_TOPIC: {lens_notopic}")
    logging.info(f"lens other: {lens}")
    logging.info(f"avg len NO_TOPIC: {np.mean(lens_notopic)}")
    logging.info(f"std len NO_TOPIC: {np.std(lens_notopic)}")
    logging.info(f"avg len other: {np.mean(lens)}")
    logging.info(f"std len other: {np.std(lens)}")
    logging.info(f"Count: {t_count}")


def evaluate(doc, gold_doc, method="dynamic", t=None):
    """
    Evaluate doc segmentation against the gold_doc
    :param doc:
    :param gold_doc:
    :param return_all:
    :param only_dynamic: whether to evaluate the given (dynamic) segmentation
    :return: accuracy and windowdiff scores, if return_all then also for uniform and gpt2 baselines
    """
    estimated_segments = int(len(gold_doc) / 256.29)

    if method == "uniform":
        us = UniformSegmentor()
        doc2 = us.segment_doc(doc, estimated_segments)
        logging.info(f"Uniform segmentation scores:")

    if method == "dynamic":
        logging.info(f"Dynamic segmentation scores:")
        doc2 = doc

    if method == "gpt2_dynamic":
        alpha = 0.8
        gs = Gpt2Segmentor()
        # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
        # doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=True, alpha=alpha, t=t)
        doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=True, alpha=alpha)
        logging.info(f"GPT2 dynamic segmentation scores, alpha {alpha}: ")

    if method == "gpt2":
        gs = Gpt2Segmentor()
        # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
        # doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=False, t=t)
        doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=False)
        logging.info(f"GPT2 (only) segmentation scores: ")

    if method == "nsp":
        nsps = NSPSegmentor()
        doc2 = nsps.segment_doc(doc, estimated_segments, window=3)
        # doc2 = nsps.segment_doc(doc, estimated_segments, window=3, t=t)
        logging.info(f"NSP segmentation scores t_{t}: ")

    a_s = accu_scores(doc2, gold_doc)
    w_d = windowdiff(doc2, gold_doc, k=int(0.5 * len(list(gold_doc.sents)) / len(gold_doc.spans["segments"])))
    logging.info(a_s)
    logging.info(w_d)
    return a_s, w_d

    #
    # if not only_dynamic:
    #     us = UniformSegmentor()
    #     estimated_segments = int(len(gold_doc) / 256.29)
    #     doc2 = us.segment(gold_doc.text, estimated_segments)
    #     logging.info(f"Uniform segmentation scores:")
    #     uni_accu = accu_scores(doc2, gold_doc)
    #     uni_wd = windowdiff(doc2, gold_doc, k=int(0.5 * len(list(gold_doc.sents)) / len(gold_doc.spans["segments"])))
    #     logging.info(uni_accu)
    #     logging.info(uni_wd)
    #
    #     gs = Gpt2Segmentor()
    #     # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
    #     doc3 = gs.segment_doc(gold_doc, estimated_segments, window=3, dynamic=True)
    #     logging.info(f"GPT2 segmentation scores: ")
    #     gpt2_accu = accu_scores(doc3, gold_doc)
    #     gpt2_wd = windowdiff(doc3, gold_doc, k=int(0.5 * len(list(gold_doc.sents)) / len(gold_doc.spans["segments"])))
    #     logging.info(gpt2_accu)
    #     logging.info(gpt2_wd)
    #
    # logging.info(f"Dynamic segmentation scores:")
    # a_s, w_d = accu_scores(doc, gold_doc), windowdiff(doc, gold_doc, k=int(0.5 * len(list(gold_doc.sents)) / len(gold_doc.spans["segments"])))
    # logging.info(a_s)
    # logging.info(w_d)
    # if return_all:
    #     return a_s, w_d, uni_accu, uni_wd, gpt2_accu, gpt2_wd
    # return a_s, w_d


def _make_length_dict(doc, doc_dynamic, gold_doc, t=None, fixed_segments=False, segmentors=None,
                      out_path="/cs/snapless/oabend/eitan.wagner/segmentation/out_docs_max"):
    """
    Makes lengths for one testimony.
    :param doc:
    :param gold_doc:
    :param t:
    :param out_path:
    :return:
    """
    def add_to_dict(dict, doc, method):
        ends = np.array(get_per_sent(doc))
        # idxs = ends != 0
        idxs = [-1] + np.nonzero(ends)[0].tolist()  # the ends are included so the "previous end" is -1
        diffs = []
        for i in range(1, len(idxs)):
            diffs.append(idxs[i] - idxs[i-1])
        dict[method] = diffs

    dict = {}
    if not fixed_segments:
        estimated_segments = int(len(gold_doc) / 256.29)
    else:
        estimated_segments = len(doc_dynamic.spans["segments"])

    logging.info("Adding gold and dynamic")
    add_to_dict(dict, gold_doc, method="gold")
    add_to_dict(dict, doc_dynamic, method="dynamic")

    if segmentors is None:
        segmentor = UniformSegmentor()
    else:
        segmentor = segmentors["uniform"]
    us = segmentor
    doc2 = us.segment_doc(doc, estimated_segments)
    logging.info("Adding uniform")
    add_to_dict(dict, doc2, method="uniform")

    if segmentors is None:
        segmentor = Gpt2Segmentor()
    else:
        segmentor = segmentors["gpt2"]
    gs = segmentor
    # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
    doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=False)
    # doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=False, t=t)
    logging.info("Adding gpt2")
    add_to_dict(dict, doc2, method="gpt2")

    alpha = 0.8
    # gs = Gpt2Segmentor()
    # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
    # doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=True, alpha=alpha, t=t)
    doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=True, alpha=alpha)
    logging.info("Adding gpt2_dynamic")
    add_to_dict(dict, doc2, method="gpt2_dynamic")

    if segmentors is None:
        segmentor = NSPSegmentor()
    else:
        segmentor = segmentors["nsp"]
    nsps = segmentor
    # doc2 = nsps.segment_doc(doc, estimated_segments, window=3, t=t)
    doc2 = nsps.segment_doc(doc, estimated_segments, window=3)
    logging.info("Adding nsp")
    add_to_dict(dict, doc2, method="nsp")

    return dict

def make_len_dict(path="/cs/snapless/oabend/eitan.wagner/segmentation/", ratio=None, fixed_segments=False,
                  segmentors=None, r=None, method="max"):
    """
    Makes dictionary of segment lengths for each method, for using segeval.
    Saves the dict in path.
    :param path:
    :return:
    """
    if r is None:
        r = range(101, 111)
    dict = {}

    for i in r:
        logging.info(f"i: {i}")
        # if i == 108:
        #     continue
        gold_doc = Doc(Vocab()).from_disk(path + 'data/gold_docs/doc_' + str(i))
        doc = Doc(Vocab()).from_disk(path + 'data/gold_docs/doc_' + str(i))
        if ratio is None:
            doc_dynamic = Doc(Vocab()).from_disk(path + f'out_docs_{method}/doc_' + str(i))
        else:
            doc_dynamic = Doc(Vocab()).from_disk(path + f'out_docs_{method}/doc_' + str(i) + "_" + str(ratio))
        dict[str(i)] = _make_length_dict(doc, doc_dynamic, gold_doc, t=i, fixed_segments=fixed_segments, segmentors=segmentors)

    # reverse the order of testimonies and methods
    _dict = {k: {} for k in dict["101"].keys()}  # for each method
    for i, d in dict.items():
        for method, l in d.items():
            _dict[method][str(i)] = l

    full_dict = {"items": _dict, "segmentation_type": "linear"}

    with open(path + 'segmentation_lens.json', 'w+') as outfile:
        json.dump(full_dict, outfile)


def seg_eval(path="/cs/snapless/oabend/eitan.wagner/segmentation/", r=None):
    """
    Evaluate with segeval package.
    :param path:
    :return:
    """
    if r is None:
        r = range(101, 111)

    def f_measure(seg1, gold):
        conf_matrix = segeval.boundary_confusion_matrix(seg1, gold)
        return segeval.fmeasure(conf_matrix)
    def precision(seg1, gold):
        conf_matrix = segeval.boundary_confusion_matrix(seg1, gold)
        return segeval.precision(conf_matrix)

    dataset = segeval.input_linear_mass_json(path + 'segmentation_lens.json')

    sims = {}
    methods = ["gpt2", "gpt2_dynamic", "nsp", "uniform", "dynamic"]


    for func in [segeval.boundary_similarity, segeval.segmentation_similarity, segeval.window_diff,
                 segeval.pk, f_measure, precision]:
        avg = {}
        for m in methods:
            _sims = []
            # for i in range(101, 111):
            for i in r:
                # if i == 108:
                #     continue
                seg1 = dataset[m][str(i)]
                gold = dataset["gold"][str(i)]
                # sims.append(segeval.boundary_similarity(seg1, gold))
                _sims.append(func(seg1, gold))
            avg[m] = np.mean(_sims)
        sims[func.__name__] = avg
        logging.info(avg)

    logging.info(sims)
    return sims


def evaluate_topics(doc, gold_doc, evaluate_dynamic=True, method="gestalt"):
    """
    Evaluate using edit distance, for the given doc and for a random (markov) assignment, both compared to the gold_doc
    :param doc:
    :param gold_doc:
    :param evaluate_dynamic: whether to evaluate with the dynamic method also
    :return:
    """
    if evaluate_dynamic:
        logging.info(f"Dynamic topic scores:")
        logging.info(gold_doc._.topics)
        logging.info(doc._.topics)
        dynamic_score = topics_score(doc._.topics, encoder.transform(gold_doc._.topics), method=method)
        logging.info(dynamic_score)

    ta = TopicAssigner(name="mcc5_iner5_iter15.pkl")
    logging.info(f"Markov Chain (mixture) topic scores:")
    # logging.info(f"Markov Chain topic scores:")
    mc_score = topics_score(ta.create(doc), encoder.transform(gold_doc._.topics), method=method)
    logging.info(mc_score)

    ta2 = TopicAssigner(markov_chain=False)
    logging.info(f"Uniform topic scores:")
    uni_score = topics_score(ta2.create(doc), encoder.transform(gold_doc._.topics), method=method)
    logging.info(uni_score)
    if evaluate_dynamic:
        return dynamic_score, mc_score, uni_score
    else:
        return mc_score


def with_segeval(ratio=0.75, r=None, method="max"):
    segmentors = {"uniform": UniformSegmentor(),
                  "gpt2": Gpt2Segmentor(),
                  "nsp": NSPSegmentor()}
    logging.info("Made segmentors")

    logging.info(f"Ratio: {ratio}")
    logging.info("With fixed segment length: ")
    make_len_dict(ratio=ratio, fixed_segments=True, segmentors=segmentors, r=r, method=method)
    seg_eval(r=r)


if __name__ == '__main__':
    # import logging

    logging.basicConfig(level=logging.INFO)
    # with_segeval(ratio=0.75)


    # logging.info("With estimated segment length: ")
    # make_len_dict(ratio=0.75, fixed_segments=False, segmentors=segmentors)
    # seg_eval()
    # make_gold_docs(save=False)
    make_gold_csv(save=True)

    # for t in ['109', '110']:
    for t in []:
    # for t in range(101, 111):
        gold_doc = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}")
        doc2 = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}")
        doc3 = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}")
        doc4 = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}")

        score = evaluate_topics(doc2, gold_doc, evaluate_dynamic=False)

        us = UniformSegmentor()
        # doc2 = us.segment(doc.text, len(doc.spans["segments"]))
        # estimated_segments = int(len(gold_doc) / 256.29)
        estimated_segments = int(len(gold_doc) / 366)
        # doc2 = us.segment(gold_doc.text, len(gold_doc.spans["segments"]))
        doc2 = us.segment_doc(doc2, estimated_segments)
        # logging.info("Uniform segmentation scores:")
        # logging.info(accu_scores(doc2, gold_doc))
        # logging.info(windowdiff(doc2, gold_doc, k=int(0.5*len(list(gold_doc.sents))/len(gold_doc.spans["segments"]))))
        logging.info(f"Uniform segmentation scores t_{t}:")
        logging.info(accu_scores(doc2, gold_doc))
        logging.info(windowdiff(doc2, gold_doc, k=int(0.5*len(list(gold_doc.sents))/len(gold_doc.spans["segments"]))))
        # #
        gs = Gpt2Segmentor()
        # doc3 = gs.segment(gold_doc.text, len(gold_doc.spans["segments"]), window=3)
        doc3 = gs.segment_doc(doc3, estimated_segments, window=3, dynamic=True, t=t)
        # doc3 = gs.segment_doc(doc3, estimated_segments, window=3, dynamic=True, t=None)
        # doc3 = gs.segment_doc(doc3, estimated_segments, window=3, dynamic=False, t=t)
        logging.info(f"GPT2 segmentation scores t_{t}: ")
        logging.info(accu_scores(doc3, gold_doc))
        logging.info(windowdiff(doc3, gold_doc, k=int(0.5*len(list(gold_doc.sents))/len(gold_doc.spans["segments"]))))

        # nsps = NSPSegmentor()
        # # doc3 = gs.segment(gold_doc.text, len(gold_doc.spans["segments"]), window=3)
        # doc4 = nsps.segment_doc(doc4, estimated_segments, window=3, t=t)
        # logging.info(f"NSP segmentation scores t_{t}: ")
        # logging.info(accu_scores(doc4, gold_doc))
        # logging.info(windowdiff(doc4, gold_doc, k=int(0.5*len(list(gold_doc.sents))/len(gold_doc.spans["segments"]))))


        #
        # if t == '110':
        # # if True:
        #     doc = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/out_docs/doc_{t}")
        #     doc.spans["sents"] = list(gold_doc.sents)
        #     logging.info(f"Dynamic segmentation scores-old t_110:")
        #     logging.info(accu_scores(doc, gold_doc))
        #     logging.info(windowdiff(doc, gold_doc, k=int(0.5*len(list(gold_doc.sents))/len(gold_doc.spans["segments"]))))
        #
        # doc = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/out_docs_max/doc_{t}")
        # logging.info(f"Dynamic segmentation scores-max t_{t}:")
        # logging.info(accu_scores(doc, gold_doc))
        # logging.info(windowdiff(doc, gold_doc, k=int(0.5*len(list(gold_doc.sents))/len(gold_doc.spans["segments"]))))
        #
        # doc = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/out_docs_marginal/doc_{t}")
        # logging.info(f"Dynamic segmentation scores-marginal t_{t}:")
        # logging.info(accu_scores(doc, gold_doc))
        # logging.info(windowdiff(doc, gold_doc, k=int(0.5*len(list(gold_doc.sents))/len(gold_doc.spans["segments"]))))
        #
