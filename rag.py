from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from transformers import RagTokenForGeneration
import torch

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq", cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True, cache_dir='/cs/snapless/oabend/eitan.wagner/cache/'
)
#
# # initialize with RagRetriever to do everything in one forward call
# model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever, cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
# # inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
# inputs = tokenizer("How many kilograms does a lion weigh?", return_tensors="pt")

# # for n in ['2', '5', '10', '100']:
# for n in ['10', '50', '100', '150', '200', '250', '300']:
#     with tokenizer.as_target_tokenizer():
#         # targets = tokenizer(f"In Paris, there are {n} million people.", return_tensors="pt")
#         targets = tokenizer(f"A lion weighs {n} kilograms.", return_tensors="pt")
#
#     input_ids = inputs["input_ids"]
#     labels = targets["input_ids"]
#     outputs = model(input_ids=input_ids, labels=labels)
#     print(f"{n}: ", outputs["loss"])



# or use retriever separately
questions = ["How much does a lion weigh?",
             "How many people live in Paris?",
             "How long does a train ride take?",
             "How long does a domestic flight take?",
             "How long does an international flight take?",
             "How long does a flight from New York to London take?",
             "How long does it take to make hard boiled eggs?",
             "How long does it take to grow a palm tree?"]
for q in questions:
    inputs = tokenizer(q, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True, cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True, cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')

    # 1. Encode
    question_hidden_states = model.question_encoder(input_ids)[0]

    # 2. Retrieve
    docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
    doc_scores = torch.bmm(
        question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)

    # # 3. Forward to generator
    # outputs = model(
    #     context_input_ids=docs_dict["context_input_ids"],
    #     context_attention_mask=docs_dict["context_attention_mask"],
    #     doc_scores=doc_scores,
    #     decoder_input_ids=labels,
    # )
    # print("2: ", outputs["logits"].shape)

    # or directly generate

    generated = model.generate(
        context_input_ids=docs_dict["context_input_ids"],
        context_attention_mask=docs_dict["context_attention_mask"],
        doc_scores=doc_scores,
    )

    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
    print(f"Q: {q} \nA: {generated_string[0]}\n")
