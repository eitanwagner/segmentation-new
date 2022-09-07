

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import json
import tqdm
import sys

def T5_joint():
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    import numpy as np
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    pair = ("dog", "black")
    orders = [(0, 1), (1, 0)]
    for order in orders:
        input_ids = tokenizer("The <extra_id_0> is <extra_id_1>.", return_tensors="pt").input_ids
        labels = tokenizer(
            f"<extra_id_{order[0]}> {pair[order[0]]} <extra_id_{order[1]}> {pair[order[1]]} <extra_id_2>",
            return_tensors="pt").input_ids
        # the forward function automatically creates the correct decoder_input_ids
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)  # is this enough or should we look at the logits??
        loss = out.loss
        print(labels[0].numpy())
        logits = out.logits[0].log_softmax(dim=1).numpy()
        print(logits.shape)
        print(logits[np.arange(len(labels)), labels])
        print(np.mean(logits[np.arange(len(labels)), labels]))
        print(order)
        print(loss.item())

def _T5_mask_filling(model=None, tokenizer=None, w1=None, w2=None, batch_size=1):
    """

    :param model:
    :param tokenizer:
    :param w1: a list of length batch_size
    :param w2:
    :param batch_size:
    :return:
    """


    input_ids = tokenizer(["The <extra_id_0> is <extra_id_1>."] * batch_size, return_tensors="pt", padding=True).input_ids
    input_ids2 = tokenizer([f"The {_w1} is <extra_id_1>." for _w1 in w1], return_tensors="pt", padding=True).input_ids
    input_ids3 = tokenizer([f"The <extra_id_0> is {_w2}." for _w2 in w2], return_tensors="pt", padding=True).input_ids
    labels = tokenizer([f"<extra_id_0> {_w1} <extra_id_1> {_w2} <extra_id_2>" for _w1, _w2 in zip(w1, w2)], return_tensors="pt", padding=True).input_ids
    labels2 = tokenizer([f"<extra_id_1> {_w2} <extra_id_2>" for _w2 in w2], return_tensors="pt", padding=True).input_ids
    labels3 = tokenizer([f"<extra_id_0> {_w1} <extra_id_1>" for _w1 in w1], return_tensors="pt", padding=True).input_ids
    # the forward function automatically creates the correct decoder_input_ids
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)  # is this enough or should we look at the logits??
        out2 = model(input_ids=input_ids2, labels=labels2)  # is this enough or should we look at the logits??
        out3 = model(input_ids=input_ids3, labels=labels3)  # is this enough or should we look at the logits??
    # loss = out.loss
    # print(labels[0].numpy())
    probs = out.logits.log_softmax(dim=-1).numpy()
    probs2 = out2.logits.log_softmax(dim=-1).numpy()
    probs3 = out3.logits.log_softmax(dim=-1).numpy()

    # dog_id = tokenizer(w1).input_ids[0]
    w1_ids = [tokenizer(_w1).input_ids[:-1] for _w1 in w1]
    w2_ids = [tokenizer(_w2).input_ids[:-1] for _w2 in w2]
    # black_id = tokenizer(w2).input_ids[0]
    # print(probs[1, dog_id] + probs[3, black_id])
    scores = {"independent":
                  np.array([float(probs[i, np.arange(1, 1+len(w1_ids[i])), w1_ids[i]].sum(dtype=float)
                         + probs[i, np.arange(2+len(w1_ids[i]), 2+len(w1_ids[i])+len(w2_ids[i])), w2_ids[i]].sum(dtype=float))
                   for i in range(batch_size)]),
              "w1 first":
                  np.array([float(probs[i, np.arange(1, 1+len(w1_ids[i])), w1_ids[i]].sum(dtype=float)
                         + probs2[i, np.arange(1, 1+len(w2_ids[i])), w2_ids[i]].sum(dtype=float))
                   for i in range(batch_size)]),
                  # float(probs[np.arange(1, 1+len(w1_ids)), w1_ids].sum(dtype=float) + probs2[np.arange(1, 1+len(w2_ids)), w2_ids].sum(dtype=float)),
              "w2 first":
                  np.array([float(probs[i, np.arange(2+len(w1_ids[i]), 2+len(w1_ids[i])+len(w2_ids[i])), w2_ids[i]].sum(dtype=float)
                         + probs3[i, np.arange(1, 1+len(w1_ids[i])), w1_ids[i]].sum(dtype=float))
                   for i in range(batch_size)]),
                  # float(probs[np.arange(3, 3+len(w2_ids)), w2_ids].sum(dtype=float) + probs3[np.arange(1, 1+len(w1_ids)), w1_ids].sum(dtype=float))}
              }
    # print(probs[np.arange(1, 1+len(w1_ids)), w1_ids].sum() + probs[np.arange(3, 3+len(w2_ids)), w2_ids].sum())
    # print(probs[1, dog_id] + probs2[1, black_id])
    # print(probs[np.arange(1, 1+len(w1_ids)), w1_ids].sum() + probs2[np.arange(1, 1+len(w2_ids)), w2_ids].sum())
    # print(probs[3, black_id] + probs3[1, dog_id])
    # print(probs[np.arange(3, 3+len(w2_ids)), w2_ids].sum() + probs3[np.arange(1, 1+len(w1_ids)), w1_ids].sum())
    # probs[3, black_id]
    # probs2[1, black_id]
    # probs3[1, dog_id]
    # print(logits.shape)
    # print(logits[np.arange(len(labels)), labels])
    # print(np.mean(probs[np.arange(len(labels)), labels]))
    # print(loss.item())
    return scores

def T5_mask_filling(noun_count=100, adj_count=50, batch_size=1):
    from nltk.corpus import wordnet as wn
    all_nouns = list(set([word for synset in wn.all_synsets('n') for word in synset.lemma_names()
                          if (word.find("_") == -1 and len(word) >= 3)]))
    all_adjs = list(set([word for synset in wn.all_synsets('a') for word in synset.lemma_names()
                         if (word.find("_") == -1 and len(word) >= 3)]))

    print("all nouns: ", len(all_nouns))
    print("all adjs: ", len(all_adjs))
    all_nouns = all_nouns[:len(all_nouns)//4]
    all_adjs = all_adjs[:len(all_adjs)//4]
    print("all nouns: ", len(all_nouns))
    print("all adjs: ", len(all_adjs))

    freq_dict = make_freq()
    all_nouns.sort(key=lambda w: freq_dict.get(w, 0), reverse=True)
    all_adjs.sort(key=lambda w: freq_dict.get(w, 0), reverse=True)
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/sorted_noun_list.json', 'w') as outfile:
        json.dump(all_nouns, outfile)
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/sorted_adj_list.json', 'w') as outfile:
        json.dump(all_adjs, outfile)

    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/sorted_noun_list.json', 'r') as f:
        all_nouns = json.load(f)
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/sorted_adj_list.json', 'r') as f:
        all_adjs = json.load(f)


    if noun_count is None:
        noun_count = len(all_nouns)
    if adj_count is None:
        adj_count = len(all_nouns)
    # all_scores_w1 = np.zeros((len(all_nouns), len(all_adjs)))
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_w1.npy', 'rb') as f:
        all_scores_w1 = np.load(f)
    # all_scores_w2 = np.zeros((len(all_nouns), len(all_adjs)))
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_w2.npy', 'rb') as f:
        all_scores_w2 = np.load(f)
    # all_scores_i = np.zeros((len(all_nouns), len(all_adjs)))
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_i.npy', 'rb') as f:
        all_scores_i = np.load(f)

    # calculated_nouns = []
    calculated_nouns = list(set(all_nouns[:2380]))


    with torch.no_grad():
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        # w1, w2 = "dog", "black"
        # noun_count = 100
        # adj_count = 50
        # all_scores = {}
        print("batch_size", batch_size)
        print("noun_count", noun_count)
        print("adj_count", adj_count)
        for i, w1 in tqdm.tqdm(enumerate(all_nouns[:noun_count])):
            if w1 in calculated_nouns:
                continue
            # for j, w2 in enumerate(all_adjs[:adj_count]):
            for j in range(0, len(all_adjs[:adj_count]), batch_size):
                w2 = all_adjs[j: j+batch_size]
                scores = _T5_mask_filling(model=model, tokenizer=tokenizer, w1=[w1]*len(w2), w2=w2, batch_size=len(w2))
                all_scores_w1[i,j:j+batch_size] = scores["w1 first"]
                all_scores_w2[i,j:j+batch_size] = scores["w2 first"]
                all_scores_i[i,j:j+batch_size] = scores["independent"]
            calculated_nouns = calculated_nouns + [w1]
                # print(w1, w2)
                # print(scores)
                # all_scores[f"{w1},{w2}"] = scores

            if i % 50 == 0:
                with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_w1.npy', 'wb') as f:
                    np.save(f, all_scores_w1)
                with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_w2.npy', 'wb') as f:
                    np.save(f, all_scores_w2)
                with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_i.npy', 'wb') as f:
                    np.save(f, all_scores_i)
                with open(f'/cs/snapless/oabend/eitan.wagner/calibration/calculated_noun_list.json', 'w') as outfile:
                    json.dump(calculated_nouns, outfile)

        # json.dump(all_scores, outfile)
    return noun_count, adj_count

def make_freq():
    # from: https://github.com/hermitdave/FrequencyWords/blob/master/content/2018/en/en_full.txt
    with open('/cs/snapless/oabend/eitan.wagner/segmentation/en_full.txt', 'r') as infile:
        lines = infile.readlines()
    freq_dict = {l.split()[0]: int(l.split()[1]) for l in lines}
    return freq_dict

def divergence2(noun_count, adj_count):
    with open(f'/cs/snapless/oabend/eitan.wagner/segmentation/all_scores_{noun_count}_{adj_count}.json', 'r') as infile:
        all_scores = json.load(infile)

    p1, p2 = [], []
    for t, s in all_scores.items():
        p1.append(s["w1 first"])
        p2.append(s["w2 first"])

    from scipy.spatial import distance
    p1, p2 = np.exp(p1), np.exp(p2)
    d = distance.jensenshannon(p1 / p1.sum(), p2 / p2.sum(), 2.)
    print("Jensen-Shannon: ")
    print(d)
    return d

def divergence(noun_count, adj_count):
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_w1.npy', 'rb') as f:
        p1 = np.load(f)[:noun_count, :adj_count]
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_w2.npy', 'rb') as f:
        p2 = np.load(f)[:noun_count, :adj_count]
    with open(f'/cs/snapless/oabend/eitan.wagner/calibration/all_scores_i.npy', 'rb') as f:
        all_scores_i = np.load(f)[:noun_count, :adj_count]


    from scipy.spatial import distance
    p1, p2 = np.exp(p1.ravel()), np.exp(p2.ravel())
    d = distance.jensenshannon(p1 / p1.sum(), p2 / p2.sum(), 2.)
    print("Jensen-Shannon: ")
    print(d)
    return d

def mask_filling():
    import torch
    # from transformers import BartTokenizerFast, BartForConditionalGeneration
    # tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    from transformers import RobertaTokenizerFast, RobertaForMaskedLM

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    # TXT = "My friends are <mask> but they eat too many carbs."

    # for t, i in tokenizer.vocab.items():
    pair = ("dog", "black")
    TXT = "The <mask> is <mask>."
    TXT1 = f"The {pair[0]} is <mask>."
    TXT2 = f"The <mask> is {pair[1]}."
    TXT3 = f"The {pair[0]} is {pair[1]}."
    # TXT = f"My friends are <mask> t"
    inputs = tokenizer([TXT], return_tensors="pt")
    inputs1 = tokenizer([TXT1], return_tensors="pt")
    inputs2 = tokenizer([TXT2], return_tensors="pt")
    inputs3 = tokenizer([TXT3], return_tensors="pt")
    print(inputs["input_ids"][0])
    print(inputs1["input_ids"][0])
    print(inputs2["input_ids"][0])
    print(inputs3["input_ids"][0])

    with torch.no_grad():
        logits = model(**inputs).logits
        logits1 = model(**inputs1).logits
        logits2 = model(**inputs2).logits
        logits3 = model(**inputs3).logits

    masked_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    masked_index1 = (inputs1.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    masked_index2 = (inputs2.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    masked_index3 = (inputs3.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    print(masked_index)
    print(masked_index1)
    print(masked_index2)
    print(masked_index3)

    probs = logits[0, masked_index].log_softmax(dim=1)
    probs1 = logits1[0, masked_index].log_softmax(dim=1)
    probs2 = logits2[0, masked_index].log_softmax(dim=1)
    probs3 = logits3[0, masked_index].log_softmax(dim=1)
    print(probs[0, [2335, 909]].sum())
    print(probs1[0, [2335, 909]].sum())
    print(probs2[0, [2335, 909]].sum())
    print(probs3[0, [2335, 909]].sum())
    print(probs[0, [2335, 909]][0] + probs1[0, [2335, 909]][1])
    print(probs[0, [2335, 909]][1] + probs2[0, [2335, 909]][0])

    values, predictions = probs.topk(5)


def causal_lm():
    import torch
    import numpy as np
    # from transformers import BartTokenizerFast, BartForCausalLM
    # tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    # model = BartForCausalLM.from_pretrained("facebook/bart-base", add_cross_attention=False)
    # assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."

    from transformers import RobertaTokenizerFast, RobertaForCausalLM, RobertaConfig
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    config = RobertaConfig.from_pretrained("roberta-base")
    config.is_decoder = True
    model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)
    # model = RobertaForCausalLM.from_pretrained("roberta-base")

    # probs = np.zeros((len(tokenizer.decoder)))
    # probs = torch.zeros(len(tokenizer.vocab))
    # for i, t in tokenizer.decoder.items():
    # for t, i in tokenizer.vocab.items():
    TXT = "My friends are <mask> fat"
    # fat_id = tokenizer.encoder["fat"]
    # fat_id = tokenizer.vocab["fat"]
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = model(input_ids).logits
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)

        # prob = logits[0, [5, 6]].softmax(dim=0)
        # probs[i] = prob[[0,1], [i, fat_id]].sum().item()

    values, predictions = probs.topk(5)
    print(tokenizer.decode(predictions).split())
    print(values)

    return probs

    # expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
    # list(logits.shape) == expected_shape


def main():
    # mask_filling()
    # causal_lm()
    # T5_joint()
    # noun_count, adj_count = T5_mask_filling(noun_count=100, adj_count=50)
    # noun_count, adj_count = 100, 50
    # print(noun_count, adj_count)
    # divergence(noun_count, adj_count)
    #
    # noun_count, adj_count = T5_mask_filling(noun_count=1000, adj_count=500)
    # noun_count, adj_count = 1000, 500
    # print(noun_count, adj_count)
    # divergence(noun_count, adj_count)
    # sys.stdout.flush()

    # noun_count, adj_count = T5_mask_filling(noun_count=5000, adj_count=500)
    noun_count, adj_count = T5_mask_filling(noun_count=None, adj_count=None, batch_size=16)
    noun_count, adj_count = 1000, 100
    # print(noun_count, adj_count)
    divergence(noun_count, adj_count)
    # T5_mask_filling(w1="encyclopedia", w2="exclusive")
    print("Done")


if __name__ == "__main__":
    main()
