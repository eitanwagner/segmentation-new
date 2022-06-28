
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
from torch import nn
import torch
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

from gpt2 import GPT2Scorer
import numpy as np
import spacy
from segment_srl import SRLer
# import segment_srl


class T5_PMI:
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/', large=True):
        size = "large" if large else "small"
        self.tokenizer = T5Tokenizer.from_pretrained(f"t5-{size}", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.model = T5ForConditionalGeneration.from_pretrained(f"t5-{size}", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        self.model.to(dev)
        self.model.eval()

        self.base_path = base_path

    def tokenize_input(self, span, srls):
        """

        :param span:
        :param srls:
        :return:
        """
        # save srl tokens for predicted sent??
        num_srls = len(srls)
        # print(srls)
        sub_spans = [span.doc[span.start:srls[0].start].text]
        special_ids = self.tokenizer.additional_special_tokens_ids[:num_srls]

        for i, s in enumerate(srls):
            # sub_spans.append(s)
            # sub_spans.append(f"<extra_id_{i}>")
            sub_spans.append(self.tokenizer.convert_ids_to_tokens(special_ids[i]))
            if i == len(srls) - 1:
                sub_spans.append(span.doc[s.end: span.end].text)
            else:
                sub_spans.append(span.doc[s.end:srls[i + 1].start].text)

        # special_id_strings = self.tokenizer.additional_special_tokens[:num_srls]
        # print(span)
        # print(sub_spans)
        input_ids = self.tokenizer(" ".join(sub_spans), return_tensors="pt").input_ids
        # print(special_ids)
        # print(input_ids.squeeze())
        special_id_locs = [(input_ids.squeeze() == sid).nonzero() for sid in special_ids]
        return input_ids, special_ids, special_id_locs

    def tokenize_labels(self, special_ids, target_srls):
        """

        :param span:
        :param target_srls:
        :return:
        """
        sub_spans = []
        for sid, srl in zip(special_ids, target_srls):
            # sub_spans.append(s)
            sub_spans.append(self.tokenizer.convert_ids_to_tokens(sid))
            sub_spans.append(srl.text)
        sub_spans.append(self.tokenizer.eos_token)

        # print(" ".join(sub_spans))
        label_ids = self.tokenizer(" ".join(sub_spans), return_tensors="pt").input_ids
        special_id_locs = [(label_ids.squeeze() == sid).nonzero() for sid in special_ids]
        return label_ids, special_ids, special_id_locs


    def srl_pmi(self, doc, sent_id, window=4, s_window=3):
        """
        Works with spacy doc with extensions: doc.spans["srls"], doc.spans["sents"], doc._.sent2srls, doc._.srl2sent
        :param doc:
        :param sent_id:
        :param window:
        :return:
        """
        srls = doc._.sent2srls[sent_id]
        if len(srls) == 0:
            return np.inf
        span = doc[doc.spans["sents"][max(0, sent_id-window)].start: doc.spans["sents"][sent_id].end]
        # print(span)
        # print(sent_id)

        _first_srl = srls[0]
        for w in range(window, 0, -1):
            if len(doc._.sent2srls[max(0, sent_id-w)]) > 0:
                _first_srl = doc._.sent2srls[max(0, sent_id-w)][0]
                break

        # first in cutoff window
        _s_first_srl = srls[0]
        for w in range(s_window, 0, -1):
            if len(doc._.sent2srls[max(0, sent_id-w)]) > 0:
                _s_first_srl = doc._.sent2srls[max(0, sent_id-w)][0]
                break
        if _s_first_srl == srls[0]:
            return 0.
        # print(len(doc.spans["srls"]))
        # print(_first_srl)
        # print(srls[-1])
        all_srls = list(doc.spans["srls"])[_first_srl: srls[-1]+1]
        # print(all_srls)
        target_srls = list(doc.spans["srls"])[srls[0]: srls[-1]+1]
        # if len(all_srls) == len(srls):
        #     return np.inf

        input_ids, special_ids, _ = self.tokenize_input(span=span, srls=all_srls)
        special_ids2 = special_ids[-len(srls):]
        label_ids, _, special_id_locs = self.tokenize_labels(special_ids=special_ids2, target_srls=target_srls)  # P(Y)
        label_ids2, _, special_id_locs2 = self.tokenize_labels(special_ids=special_ids, target_srls=all_srls)  # P(Y|X)
        special_id_locs2 = special_id_locs2[-len(srls):]
        if special_id_locs[0].size()[0] == 0:
            return np.inf
        # print(input_ids.size())
        # print(label_ids2.size())
        # print(special_id_locs)
        # print(special_id_locs2)
        input_ids = input_ids.to(dev)
        label_ids = label_ids.to(dev)
        label_ids2 = label_ids2.to(dev)
        logits = self.model(input_ids=input_ids, labels=label_ids).logits.cpu()
        # out = out.cpu()
        # logits = out.logits
        logits2 = self.model(input_ids=input_ids, labels=label_ids2).logits.cpu()
        # out = out.cpu()
        # logits2 = out.logits
        # print(logits.size())
        log_probs = [(logits[0][sid_locs.squeeze(1)].gather(1, sid_locs).squeeze(1) - logits[0][sid_locs.squeeze(1)].logsumexp(1)).sum()
                     for sid_locs in special_id_locs]
        log_probs2 = [(logits2[0][sid_locs.squeeze(1)].gather(1, sid_locs).squeeze(1) - logits2[0][sid_locs.squeeze(1)].logsumexp(1)).sum()
                     for sid_locs in special_id_locs2]
        return (sum(log_probs2) - sum(log_probs)).detach().numpy()

# class SrlPMI:
#     pass

class DocPMI:
    def __init__(self, doc, spacy_doc=True, nlp=None, window=3, token_window=0):
        """

        :param token_window:
        :param nlp: a spacy pipeline
        :param doc:
        :param spacy_doc: whether the doc is a spacy object or plain text
        """
        self.lm = GPT2Scorer()
        self.pmis = None
        if nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp
        if not spacy_doc:
            self.doc = nlp(doc)
        else:
            self.doc = doc
        self.sents = list(doc.sents)
        self.cache = {}
        self.window = window
        self.token_window = token_window

    def pmi(self, s1, s2):
        """
        Calculate the PMI between s1 and s2.
        :param s1:
        :param s2:
        :return:
        """
        # switch to use the conditional variation (which should be better normalized)
        if (s1, s2) not in self.cache.keys():
            self.cache[(s1, s2)] = self.lm.sentence_score(s2, past=s1) - self.lm.sentence_score(s2)
            # self.cache[(s1, s2)] = self.lm.sentence_score(s1 + s2) - self.lm.sentence_score(s1) - self.lm.sentence_score(s2)
        return self.cache[(s1, s2)]

    def PMI_marginal(self, segment):
        """
        Calculate the average PMI of the given segment with all other segments of length window.
        :param doc: A spacy doc
        :param segment: A spacy span in the doc
        :param window:
        :return:
        """
        doc = self.doc
        sents = list(doc.sents)
        if self.token_window == 0:
            window = self.window
            pmis = [self.pmi(doc[s.start: sents[i + window - 1].end].text, segment.text) for i, s in enumerate(sents[:-window])]
        else:
            window = self.token_window
            pmis = [self.pmi(doc[s.start: s.start+window].text, segment.text) for s in sents[:-self.window]]

        return np.mean(pmis)

    def PMI_consecutive(self, return_mean=False):
        """
        Calculate the mutual information for consecutive segments of length window.
        :param return_mean: whether to return the mean (MI) or the pmi
        :param doc:
        :param window:
        :return:
        """
        doc = self.doc
        sents = list(doc.sents)
        if self.token_window == 0:
            window = self.window
            # i start from 0
            pmis = [self.pmi(doc[sents[i].start: s.start].text, doc[s.start: sents[i + 2*window - 1].end].text)
                    for i, s in enumerate(sents[window:-window-1])]
        else:
            window = self.token_window
            pmis = [self.pmi(doc[max(s.start-window, 0): s.start].text, doc[s.start: min(s.start+window, len(doc))].text)
                    for s in sents[self.window:-self.window-1]]

        if return_mean:
            return np.mean(pmis)
        else:
            return [np.inf] * window + pmis + [np.inf] * window

if __name__ == "__main__":
    pass