
import numpy as np
import json
import sys
import spacy
from spacy.tokens import Doc
Doc.set_extension("id", default=None, force=True)

import logging

from pmi import DocPMI
from pmi import T5_PMI
from segment_srl import SRLer
from book import add_events_to_doc, add_ents_to_doc
from clustering import cluster
# import segment_srl


def combine_sents(doc, js):
    """
    Make a span list based on a list of chosen sentence indices
    :param doc:
    :param js:
    :return:
    """
    # TODO
    if len(js) == 0:
        return []
    _sents = [s for i, s in enumerate(doc.sents) if i in js]

    n_sents = [doc[:_sents[0].start]]
    n_sents = n_sents + [doc[_s.start:_sents[i+1].start] for i, _s in enumerate(_sents[:-1])]
    n_sents.append(doc[_sents[-1].start:])
    return n_sents

def verb_threshold(spans, v_threshold, events=False):
    """
    Filers spans that are under the threshold regarding verb counts
    :param v_threshold:
    :param spans:

    :return:
    """
    def verb_count(span, events=False):
        if events:
            return len([t for t in span if t._.is_event])
        return len([t for t in span if t.pos_ == "VERB"])
    return [s for s in spans if verb_count(s, events=events) >= v_threshold]


class SubstoryExtractor:
    def __init__(self, nlp=None, window=5, s_window=3, token_window=0, large=True):
        self.large = large
        self.token_window = token_window
        self.window = window
        self.s_window = s_window
        if nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp

        self.doc_pmi = None
        self.srl_pmi = None
        self.doc = None

    def extract_by_cluster(self, doc, v_threshold=0, events=True, k=3, use_affinity=True):
        """

        :param doc:
        :param v_threshold:
        :param events:
        :return:
        """
        def make_X(doc, use_coref=True, use_pmi=False):
            if use_coref:
                num_ents = max([max(s._.ents) for s in doc.spans["sents"] if len(s._.ents) > 0])
                X = np.zeros((len(doc.spans["sents"]), 1 + num_ents))
                X[:, 0] = np.array([1 * (sum([tok._.is_event for tok in s]) > 0) for s in doc.sents])
                for i, s in enumerate(doc.spans["sents"]):
                    for e in s._.ents:
                        X[i, e] = 1 / 10
                        # X[i, e] = 1 / num_ents
                return X
            return np.array([1 * (sum([tok._.is_event for tok in s]) > 0) for s in doc.sents]).reshape(-1, 1)

        def make_affinity(doc, pmis, max_len=10):
            _pmis = np.array(pmis)
            _pmis = _pmis[_pmis > 0]
            _pmis = _pmis[_pmis < np.inf]
            pmi_mean = np.mean(_pmis)

            X = np.zeros((len(doc.spans["sents"]), len(doc.spans["sents"])))
            events = np.array([1 * (sum([tok._.is_event for tok in s]) > 0) for s in doc.sents])
            for i, s1 in enumerate(doc.spans["sents"]):
                for j, s2 in enumerate(list(doc.spans["sents"])[i+1: i+max_len]):
                    X[i, i+j+1] += 1 - (len(set(s1._.ents).intersection(s2._.ents))) / 5
                    X[i+j+1, i] = X[i, i+j+1]
                    if events[i] != events[j]:
                        X[i, i+j+1] += 1
                        X[i+j+1, i] = X[i, i+j+1]
                if i >= 1 and pmis[i - self.window] < np.inf:
                    X[i, i-1] -= pmis[i - self.window] / pmi_mean  # Check!!!
                    X[i-1, i] = X[i, i-1]
            X -= X.min()
            return X


        sents = list(doc.sents)
        if use_affinity:
            X = make_affinity(doc, pmis=self.get_PMIs(doc))
        else:
            X = make_X(doc)
        s_e = cluster(X, k=k, use_precomputed=use_affinity)
        all_subs = [doc[sents[s].start:sents[e-1].end] for s, e in s_e]
        doc.spans["substories"] = verb_threshold(all_subs, v_threshold, events=events)
        # TODO:
        # give score to each cluster
        # deal with (rare) overlaps

    def get_PMIs(self, doc):
        self.srl_pmi = T5_PMI(large=self.large)
        pmis = []
        for i, s in enumerate(doc.spans["sents"]):
            if i < self.window:
                continue
            pmis.append(self.srl_pmi.srl_pmi(doc, sent_id=i, window=self.window, s_window=self.s_window))
        return pmis

    def extract_by_srl(self, doc, v_threshold=0, factor=1., events=False):
        """

        :param doc:
        :param v_threshold:
        :param factor: large factor means smaller threshold which means less "cuts" and longer substories
        :return:
        """
        pmis = self.get_PMIs(doc)

        _pmis = np.array(pmis)
        _pmis = _pmis[_pmis > 0]
        _pmis = _pmis[_pmis < np.inf]

        # p_threshold = np.mean(np.array(pmis)[np.isfinite(pmis)])
        p_threshold = np.mean(np.array(_pmis))
        p_threshold = p_threshold - np.log(factor)

        # large PMI means we want to connect
        js = np.nonzero(pmis < p_threshold)[0] + self.window  # because the PMI starts after a window
        spans = combine_sents(doc, js=js)

        doc.spans["substories"] = verb_threshold(spans, v_threshold, events=events)

    def extract(self, doc, spacy_doc=True, p_method="avg", v_threshold=0, factor=1.):
        """

        :param factor: larger factor means more connections
        :param p_threshold:
        :param v_threshold:
        :param doc: a text file
        :param spacy_doc: whether the doc is a spacy object or plain text
        :return:
        """
        self.doc_pmi = DocPMI(doc=doc, spacy_doc=spacy_doc, nlp=self.nlp, window=self.window, token_window=self.token_window)
        pmis = self.doc_pmi.PMI_consecutive(return_mean=False)
        # mi = self.doc_pmi.PMI_consecutive(window=self.window, return_mean=True)

        p_threshold = 1.
        if p_method == "avg":
            p_threshold = self.doc_pmi.PMI_consecutive(return_mean=True)
            # p_threshold = p_threshold - np.log(0.1)
            p_threshold = p_threshold - np.log(factor)

        # large PMI means we want to connect
        js = np.nonzero(pmis < p_threshold)[0]
        spans = combine_sents(doc, js=js)
        doc.spans["substories"] = verb_threshold(spans, v_threshold)

    def print_subs(self, doc):
        logging.info(f'Sentences: {len([s for s in doc.sents])}')
        logging.info(f'Substories: {len(doc.spans["substories"])}')
        for s in doc.spans["substories"]:
            logging.info(s.text)


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

def get_sf_testimony_nums(data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Get list of testimony ids for the SF corpus
    :param data_path:
    :return:
    """
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        nums = list(json.load(infile).keys())
    return nums


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })

    logging.info("\n\nStarting")
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'

    # ts = get_sf_testimony_nums()
    ts = [20581]
    nlp = spacy.load("en_core_web_sm")
    logging.info(ts[0])
    text = get_sf_testimony_text(ts[0])
    doc = nlp(text[:])
    doc._.id = ts[0]
    add_events_to_doc(doc, doc._.id)
    add_ents_to_doc(doc, doc._.id)

    # TODO first add all srls!!
    srler = SRLer()
    srler.sent_parse(doc=doc, events=True)

    se = SubstoryExtractor(nlp=nlp, window=10, s_window=5, large=False)
    se.extract_by_cluster(doc, v_threshold=10, events=True)
    se.print_subs(doc)

    # se = SubstoryExtractor(nlp=nlp, window=6, token_window=50, large=False)
    se = SubstoryExtractor(nlp=nlp, window=10, s_window=5, token_window=50, large=True)
    # params = [(10, 2.), (10, 1.), (10, 0.01), (10, 0.25), (20, 2), (20, 1.)]
    # params = [(5, 1.), (5, 2.), (5, 4.), (5, .01), (3, 1.), (10, 1.)]
    params = [(15, 1.), (10, 20.), (10, 10.)]
    logging.info("Doc:\n" + doc.text)
    for v, f in params:
        logging.info((v, f))
        se.extract_by_srl(doc, v_threshold=v, factor=f, events=True)
        se.print_subs(doc)
