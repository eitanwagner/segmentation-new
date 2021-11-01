# import pandas as pd
import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
# from scipy.special import log_softmax
import numpy as np
# from scipy.special import logsumexp
# from scipy.stats import poisson
# from torch import log_softmax

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

# from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# MAX_LEN, MIN_LEN = 1000, 35
MAX_LEN, MIN_LEN = 800, 35


class GPT2Scorer:
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/'):
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
        self.model.to(dev)

        self.base_path = base_path
        self.cache = {}
        self.cache_id = None

    def save_cache(self):
        np.save(self.base_path + f"/gpt2_cache{str(self.cache_id)}.npy", self.cache)

    def load_cache(self, i):
        self.cache_id = i
        try:
            self.cache = np.load(self.base_path + f"/gpt2_cache{str(i)}.npy",
                                 allow_pickle='TRUE').item()
        except IOError as err:
            # except:
            pass


    def sliding(self, encodings):
        """
        Calculate the loss using a sliding window.
        Taken from HuggingFace.
        :param encodings:
        :return:
        """
        max_length = self.model.config.n_positions
        stride = 512

        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(dev)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        # ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return torch.stack(nlls).sum().cpu().numpy()

    def sentence_score(self, sent):
        # TODO: is this the probability or the loss??
        if sent in self.cache.keys():
            return self.cache[sent]

        # returns sentence probability (in log space)
        inputs = self.tokenizer(sent, return_tensors="pt")
        # print(inputs['input_ids'].shape)
        inputs = inputs.to(dev)

        if inputs.input_ids.size(1) > 1000:
            out = self.sliding(inputs)

        else:
            out = self.model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])['loss'].item() * inputs['input_ids'].shape[1]
        self.cache[sent] = out
        return out

        # check why this doesn't work!!
        # out = self.model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])['logits']
        # print(out.shape)
        # ln = inputs['input_ids'][0].detach().cpu().numpy()
        # sm = log_softmax(out[0].detach().cpu().numpy(), axis=0)
        # print(sm.shape)
        # return sum(sm[range(sm.shape[0]), ln])
        # return sum(sm[range(1, sm.shape[0] - 1), ln[1:-1]])

# class GPT2wDMM(GPT2Scorer):
#     def __init__(self, TM=None):
#         super().__init__()
#         self.TM = TM
#
#     def score_with_topic(self, topic_model, sent, log=True):
#         # returns sentence probability (in log space)
#         inputs = self.tokenizer(sent, return_tensors="pt")
#         # print(inputs['input_ids'].shape)
#         inputs = inputs.to(dev)
#         logits = self.model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])['logits'][0].detach().cpu().numpy()
#
#         w2lemmatized = self.TM.lemmatize(sent) # should return a dict from word to the lemmatized word, with None for out of vocab words
#         tm_logps = self.TM.probabilities(list(w2lemmatized.values()))  # should return prior log probabilites for words with None
#         for i, w in enumerate(list(w2lemmatized.keys())):
#             w2t = inputs.word_to_tokens(list(w2lemmatized.values()))
#             w_logits = np.sum(logits[:, w2t], axis=1)



if __name__ == "__main__":
    pass