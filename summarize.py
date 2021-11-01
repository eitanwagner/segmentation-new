

import os
CACHE_DIR = "/cs/snapless/oabend/eitan.wagner/cache/"
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
os.environ['HF_METRICS_CACHE'] = CACHE_DIR

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from scipy.special import log_softmax
import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson
# from torch import log_softmax
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging


if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

from spacy.tokens import Token
Token.set_extension("depth", default=None, force=True)

from summarizer import Summarizer as BertSummarizer

class Summarizer:
    def __init__(self, classifier, default_len=10, use_bert_summarizer=False):
        self.use_bert_summarizer = use_bert_summarizer
        if use_bert_summarizer:
            self.summarizer = BertSummarizer()
        self.default_len = default_len  # this is measured in tokens and not words
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.model_name = 'pegasus'
        self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum', cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.model.to(dev)
        # self.tokenizer = AutoTokenizer.from_pretrained("t5-base", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        # self.tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum', cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.classifier = classifier

    def sents_score(self, doc, texts, class_num):
        # returns scores for given texts this should be a 1-dim array
        # logging.info(f"len(texts): {len(texts)}")
        inputs1 = self.tokenizer(["summarize: " + doc.text for _ in texts], return_tensors='pt', padding=True)
        inputs1 = inputs1.to(dev)
        inputs2 = self.tokenizer([text for text in texts], return_tensors='pt', padding=True)
        inputs2 = inputs2.to(dev)
        # logging.info(f"inputs1: {inputs1['input_ids'][0].size()}")
        # logging.info(f"inputs2: {inputs2['input_ids'][0].size()}")

        out = self.model(input_ids=inputs1['input_ids'], labels=inputs2['input_ids'])['logits']
        ln = inputs2['input_ids'][0].detach().cpu().numpy()
        # logging.info(f"ln.shape: {ln.shape}")

        sm = log_softmax(out.detach().cpu().numpy(), axis=2)  # check dimensions!!
        # logging.info(f"sm.shape: {sm.shape}")
        # this is P(x|X)
        log_summary_probs = np.mean(sm[:, range(1, sm.shape[1]-1), ln[1:-1]], axis=1)  # using mean log-probability
        log_summary_probs += poisson.logpmf(len(inputs2), self.default_len * np.ones(len(texts)))  # !!!!
        # logging.info(f"log summary probs.shape: {log_summary_probs.shape}")

        # use classifier score. this is logP(t|x).
        log_class_probs = np.array([self.classifier.predict_raw(text)[class_num] for text in texts])
        # logging.info(f"log_class_probs.shape: {log_class_probs.shape}")
        # Assuming this does not depent on X, then the sum is logP(t,x|X) and assuming uniform P(t) the subtractions is logP(x|t,X)
        return log_summary_probs + log_class_probs - np.log(len(self.classifier.topics))

    def sents_classification_score(self, texts, class_num):
        # The input should be a list
        # returns a list even if only one text was given.
        log_class_probs = np.array([self.classifier.predict_raw(text)[class_num] for text in texts])
        return log_class_probs

    def add_depth(self, doc):
        # add ._.depth property to every token in the doc
        def _add_depth_recursive(node, depth):
            node._.depth = depth
            if node.n_lefts + node.n_rights > 0:
                return [_add_depth_recursive(child, depth + 1) for child in node.children]

        for sent in doc.sents:
            _add_depth_recursive(sent.root, 0)

    def make_random(self, sent, depth):
        words = []
        choices = np.random.uniform(size=len(sent)) > .2
        def _add_recursive(node):
            if node._.depth <= depth:
                words.append(node)
            if node.n_lefts + node.n_rights > 0:
                [_add_recursive(child) for i, child in enumerate(node.children) if choices[i]]
        for _ in range(100):
            _add_recursive(sent.root)
            if len(words) > 2 and len(sent) > 2:
                return " ".join([t.text for t in sent if t in words])
            words = []

    def make_sents(self, doc, max_depth, random=10):
        # makes list of new sentences for the whole doc with up to depth max_depth
        # random is the number of random samples for each sentence
        def _make_sent(sent, depth):
            return " ".join([t.text for t in sent if t._.depth <= depth])

        new_sents = []
        for sent in doc.sents:
            if random is None:
                for m_d in range(1, max_depth):  # not just the root
                    if len(sent) > 3:  # not too short
                        new_sents.append(_make_sent(sent, m_d))
            else:
                for _ in range(random):
                    new_sents.append(self.make_random(sent, depth=max_depth))
        if len(new_sents) > 0:
            s = set(new_sents)
            s.discard(None)
            return list(s)
        else:
            return []

    def make_sents_simple(self, doc):
        # returns list of sentences in the document
        sents = [s.text for s in doc.sents]
        return sents

    def generate(self, doc, num_summaries=3, max_length=20):
        # generates summaries (not necessarily extractive)
        if self.model_name == 'pegasus':
            inputs = self.tokenizer(doc.text, return_tensors="pt").input_ids  # do we need batches??
        else:
            inputs = self.tokenizer("summarize: " + doc.text, return_tensors="pt").input_ids
        inputs = inputs.to(dev)
        outputs = self.model.generate(input_ids=inputs, max_length=max_length, num_return_sequences=num_summaries, do_sample=True)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_ranked_sents(self, doc, max_depth, class_num, count, simple=True, ratio=0.2):
        """

        :param doc:
        :param max_depth:
        :param class_num:
        :return:
        """
        # returns a list of tuples tuples (log_prob,text)
        self.add_depth(doc)
        # new_sents = self.make_sents(doc, max_depth)
        if simple:
            new_sents = self.make_sents_simple(doc)
            idxs = range(len(new_sents))
        elif self.use_bert_summarizer:  # in this case the count is the number in the span and not the number of separate sentences
            # return [(1, self.summarizer(doc.text, num_sentences=count))]
            return [(1, self.summarizer(doc.text, ratio=ratio))]
        else:
            new_sents = self.generate(doc)
        if len(new_sents) == 0:
            return [(1, "")]
        # print(new_sents)
        # log_probs = self.sents_score(new_sents, class_num)
        # log_probs = [self.sents_score(doc, n_s, class_num)[0] for n_s in new_sents]
        log_probs = [self.sents_classification_score(n_s, class_num)[0] for n_s in new_sents]

        if simple:
            log_probs, idxs = list(zip(*sorted(list(zip(log_probs, idxs)), reverse=True)[:count]))
            return [(1, new_sents[i]) for i in sorted(idxs)]  # !!!!

        return sorted(list(zip(log_probs, new_sents)), reverse=True)[:count]


if __name__ == "__main__":
    from summarizer import Summarizer

    body = """But none of us-- I, uh, I shouldnt say none. Few of us want to go back into a ghetto kind of set up, with the exception of a few Hasidic groups that you have in-- in-- in New York, who have created their own ghettos and feel most comfortable in that way. 
    Because they feel completely threatened by the outside world. I think today, the majority of Jews, including orthodoxy, certainly want to live in the two worlds, the Jewish world and the general world, the culture. 
    Where the cultures do not conflict, we can be together. Where there"s a conflict, I think we have to be able to say, this I do not accept. 
    This is not for me. And that's the only way, I think, we can continue building upon the culture that was. 
    But I don't think is fit for the twentieth or twenty-first century anymore. 
    INTERVIEWER: Rabbi, thank you. Thank you very-- fantastic. And the reason I stopped it was, was the click of the camera, and we have to edit. Yeah, that was quiet. It was. The first time I heard it, it was-- """

    body2 = """Then the Warsaw-- overall, the Warsaw community, which I'm sure represented all kinds of Jews. Enlightened Jews, religious Jews, non-religious Jews, socialists. And they all had their own-- their own organizations and their-- their own get togethers. And yet, the overall spirit, however, I always think of Polish Jewry was a religious one.
    I think the religious community was probably the majority in Poland, not the other way around. The religious community was successful in electing representatives to the Polish parliament. So was the non-religious comm-- the Zionists-- now, for example, Zionism was frowned upon by this community completely. Whether religious Zionism or-- or Zionism, per se, that was frowned upon. But I remember, I had a few cousins who are Zionists. And their family was not very happy with that, because that already was a sign that they were not as religious. Which was true, but that's what they looked at.
    INTERVIEWER: In conclusion, about this culture that is no more.
    MEYER STRASSFELD: I don't think we-- we can recapture that culture. We can recapture some of the the themes, the goals of the culture, such as education, such as commitment to Judaism, such as a desire to ensure the continuity of the state of Israel.
    """

    model = Summarizer()
    print("Made summarizer")
    sum1 = model(body, num_sentences=2)
    print(len(body), sum1, len(sum1))
    sum2 = model(body2, num_sentences=2)
    print(len(body2), sum2, len(sum2))
    pass
