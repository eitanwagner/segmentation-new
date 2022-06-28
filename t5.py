from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM
import numpy as np
from torch import nn
import torch

sm = nn.LogSoftmax(dim=1)
for w in ["mean", "nice", "insulting", "funny", "rude"]:
    print(w)
    shift = 1 if w == "insulting" else 0
    tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    model = T5ForConditionalGeneration.from_pretrained("t5-small", cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")

    input_ids = tokenizer("The man said <extra_id_0> and the boy <extra_id_1>.", return_tensors="pt").input_ids
    labels = tokenizer(f"<extra_id_0> something {w} <extra_id_1> was insulted <extra_id_2>", return_tensors="pt").input_ids

    out = model(input_ids=input_ids, labels=labels)
    loss2 = out.loss
    logits = out.logits

    insulted_id = tokenizer("was insulted", return_tensors="pt").input_ids[0]
    mean_id = tokenizer("something mean", return_tensors="pt").input_ids[0]
    insulting_id = tokenizer("something insulting", return_tensors="pt").input_ids[0]
    print(labels)
    print(insulted_id)
    # nice_id = tokenizer("nice", return_tensors="pt").input_ids[0]
    # happy_id = tokenizer("happy", return_tensors="pt").input_ids[0]

    # logits[0][4:8].gather(1, insulted_id[:-1].unsqueeze(1)).squeeze(1)
    log_probs = logits[0][4:8].gather(1, insulted_id[:-1].unsqueeze(1)).squeeze(1) - logits[0][4:8].logsumexp(1)
    log_probs1 = sm(logits[0])[np.arange(4, 8)+shift, insulted_id[:-1]]
    # logits[0][np.arange(2, 4), mean_id[:-1]].sum()
    print((sm(logits[0])[np.arange(4, 8)+shift, insulted_id[:-1]]).sum())


    input_ids = tokenizer("The man said <extra_id_0> and the boy <extra_id_1>.", return_tensors="pt").input_ids
    # labels = tokenizer("<extra_id_0> <extra_id_1> <extra_id_2> was insulted <extra_id_3>", return_tensors="pt").input_ids
    labels = tokenizer("<extra_id_1> was insulted <extra_id_2>", return_tensors="pt").input_ids
    # labels = tokenizer(f"<extra_id_1> was insulted <extra_id_0> something {w} <extra_id_2>", return_tensors="pt").input_ids
    print(labels)

    out = model(input_ids=input_ids, labels=labels)
    logits = out.logits

    insulted_id = tokenizer("was insulted", return_tensors="pt").input_ids[0]

    # torch.softmax(logits.gather(1, insulted_id))
    print(sm(logits[0])[np.arange(1, 5), insulted_id[:-1]].sum())


    # labels = tokenizer("<extra_id_1> was insulted <extra_id_2>", return_tensors="pt").input_ids
    labels = tokenizer(f"<extra_id_1> was insulted <extra_id_0> something {w} <extra_id_2>", return_tensors="pt").input_ids
    print(labels)

    out = model(input_ids=input_ids, labels=labels)
    logits = out.logits

    insulted_id = tokenizer("was insulted", return_tensors="pt").input_ids[0]

    # torch.softmax(logits.gather(1, insulted_id))
    print(sm(logits[0])[np.arange(1, 5), insulted_id[:-1]].sum())


for i in []:
    print(i)
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xlarge',
                                  cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    model = DebertaV2ForMaskedLM.from_pretrained("microsoft/deberta-v2-xlarge",
                                                       cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    # tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-small',
    #                               cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    # model = DebertaV2ForMaskedLM.from_pretrained("microsoft/deberta-v3-small",
    #                                                    cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")

    insulted_id = tokenizer("insulted", return_tensors="pt").input_ids[0]
    mean_id = tokenizer("mean", return_tensors="pt").input_ids[0]
    nice_id = tokenizer("nice", return_tensors="pt").input_ids[0]
    happy_id = tokenizer("happy", return_tensors="pt").input_ids[0]

    print("Mean:")
    inputs = tokenizer("The man said something mean and the boy was insulted.", return_tensors="pt")
    # inputs = tokenizer("The man said something nice and the boy was happy.", return_tensors="pt")
    # inputs = tokenizer("The dog ate in the park and scared the boy.", return_tensors="pt")
    input_ids = inputs.input_ids
    # input_ids = tokenizer("The [MASK] walks in [MASK] park", return_tensors="pt").input_ids
    # labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    labels = input_ids.clone()
    # input_ids[0,2] = input_ids[0,5] = tokenizer.mask_token_id
    # input_ids[0,2] = tokenizer.mask_token_id
    input_ids[0, 10] = tokenizer.mask_token_id
    labels[input_ids != tokenizer.mask_token_id] = -100


    # the forward function automatically creates the correct decoder_input_ids
    out = model(input_ids=input_ids, labels=labels,
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"])
    loss2 = out.loss
    logits = out.logits
    # dog_id = tokenizer("dog", return_tensors="pt").input_ids[0]
    # the_id = tokenizer("the", return_tensors="pt").input_ids[0]
    # run_id = tokenizer("run", return_tensors="pt").input_ids[0]
    print(loss2.item())

    # inputs = tokenizer("The dog barked in the park and scared the boy.", return_tensors="pt")
    inputs = tokenizer("The man said something mean and the boy was insulted.", return_tensors="pt")
    # inputs = tokenizer("The dog ate in the park and scared the boy.", return_tensors="pt")
    input_ids = inputs.input_ids
    # input_ids = tokenizer("The [MASK] walks in [MASK] park", return_tensors="pt").input_ids
    # labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    labels = input_ids.clone()
    input_ids[0,5] = input_ids[0,10] = tokenizer.mask_token_id
    # input_ids[0,2] = tokenizer.mask_token_id
    # input_ids[0,5] = tokenizer.mask_token_id
    labels[input_ids != tokenizer.mask_token_id] = -100
    labels[0,5] = -100

    # the forward function automatically creates the correct decoder_input_ids
    out = model(input_ids=input_ids, labels=labels,
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"])
    loss3 = out.loss
    logits = out.logits
    # dog_id = tokenizer("dog", return_tensors="pt").input_ids[0]
    # the_id = tokenizer("the", return_tensors="pt").input_ids[0]
    # run_id = tokenizer("run", return_tensors="pt").input_ids[0]
    print(loss3.item())


    print("Ate:")
    inputs = tokenizer("The man said something nice and the boy was insulted.", return_tensors="pt")
    # inputs = tokenizer("The dog ate in the park and scared the boy.", return_tensors="pt")
    input_ids = inputs.input_ids
    labels = input_ids.clone()
    input_ids[0, 10] = tokenizer.mask_token_id
    labels[input_ids != tokenizer.mask_token_id] = -100
    out = model(input_ids=input_ids, labels=labels,
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"])
    loss2 = out.loss
    logits = out.logits
    print(loss2.item())

    # inputs = tokenizer("The dog ate in the park and scared the boy.", return_tensors="pt")
    inputs = tokenizer("The man said something nice and the boy was insulted.", return_tensors="pt")
    input_ids = inputs.input_ids
    labels = input_ids.clone()
    input_ids[0,5] = input_ids[0,10] = tokenizer.mask_token_id
    labels[input_ids != tokenizer.mask_token_id] = -100
    labels[0,5] = -100

    out = model(input_ids=input_ids, labels=labels,
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"])
    loss3 = out.loss
    logits = out.logits
    print(loss3.item())


    # print("Growled:")
    # inputs = tokenizer("The dog growled in the park and scared the boy.", return_tensors="pt")
    # input_ids = inputs.input_ids
    # labels = input_ids.clone()
    # input_ids[0, 8] = tokenizer.mask_token_id
    # labels[input_ids != tokenizer.mask_token_id] = -100
    # out = model(input_ids=input_ids, labels=labels,
    #             attention_mask=inputs["attention_mask"],
    #             token_type_ids=inputs["token_type_ids"])
    # loss2 = out.loss
    # logits = out.logits
    # print(loss2.item())
    #
    # inputs = tokenizer("The dog growled in the park and scared the boy.", return_tensors="pt")
    # input_ids = inputs.input_ids
    # labels = input_ids.clone()
    # input_ids[0,3] = input_ids[0,8] = tokenizer.mask_token_id
    # labels[input_ids != tokenizer.mask_token_id] = -100
    # labels[0,3] = -100
    #
    # out = model(input_ids=input_ids, labels=labels,
    #             attention_mask=inputs["attention_mask"],
    #             token_type_ids=inputs["token_type_ids"])
    # loss3 = out.loss
    # logits = out.logits
    # print(loss3.item())