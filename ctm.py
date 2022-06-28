
# pip install contextualized-topic-models==2.2.0
# pip install nltk
# pip install pyldavis
# wget https://raw.githubusercontent.com/vinid/data/master/italian_documents.txt
# wget https://raw.githubusercontent.com/vinid/data/master/dbpedia_sample_abstract_20k_unprep.txt

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import nltk
import json
import spacy
import re
from tqdm import tqdm
import numpy as np


NERs = ["PERSON", "NORP", "FACILITY", "ORGANIZATION", "GPE", "LOCATION", "PRODUCT", "EVENT", "WORK", "LAW", "LANGUAGE",
        "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "ORG", "FAC"]

extra_stops = []


def preprocess(s):
    """
    Extra preprocessing
    :param s:
    :return:
    """
    replace_tuples = [("ISUBJECT", "INT"), ("PERSON", ""), ("CARDINAL", ""), ("ORDINAL", "")]
    for r_t in replace_tuples:
        s = s.replace(*r_t)
    return s


def remove_extra_stopwords(docs):
    """

    :param docs: a list of documents
    :return:
    """

    return [" ".join([w for w in doc.split() if w not in extra_stops]) for doc in docs]


def predict(ctm, tp, texts):
    # ****************** predict for given segment
    sp = WhiteSpacePreprocessing(texts, stopwords_language='english')  # TODO: take out of the function and add all relevant documents
    preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess(keep_brackets_underscore=True)
    test_dataset = tp.transform(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
    print(preprocessed_documents)
    thetas = ctm.get_thetas(test_dataset)
    # alpha = ctm.get_topic_word_distribution()
    # print(thetas[0])
    alpha = ctm.get_topic_word_distribution()
    print(alpha.shape)
    for i, p_d in enumerate(preprocessed_documents):
        # ids = [word2id[w] for w in p_d.split]
        test_dataset = tp.transform(text_for_contextual=[unpreprocessed_corpus[i]], text_for_bow=[preprocessed_documents[i]])
        # word2id = {w: i for i, w in test_dataset.idx2token.items()}
        word2id = {w: i for i, w in ctm.train_data.idx2token.items()}
        print("***********")
        print([word2id.get(w, None) for w in p_d.split() if word2id.get(w, None) is not None])
        thetas = ctm.get_thetas(test_dataset)
        likelihoods = alpha[:, [word2id.get(w, None) for w in p_d.split() if word2id.get(w, None) is not None]].sum(axis=1)
        weighted = thetas @ likelihoods
        print(f"i: {i}", weighted)


if __name__ == "__main__":

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    path = "/cs/snapless/oabend/eitan.wagner/segmentation/CTM_w_time_40/"
    ctm = CombinedTM(bow_size=1998, contextual_size=768, n_components=40, num_epochs=20)
    ctm.load(model_dir=path + "contextualized_topic_model_nc_40_tpm_0.0_tpv_0.975_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99",
             epoch=19)

    texts = ["Hello world", "Holocaust survivor", "The survivor", "George Eisenberg", "The interviewer","United States of America"]
    data_path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/'

    with open(data_path + 'sf_all.json', 'r') as infile:
        dd = json.load(infile)
    with open(data_path + 'sf_unused.json', 'r') as infile:
        unused = json.load(infile)
    # documents = [t['text'] for l in dd.values() for t in l if len(t['text'])<500]
    documents = [t['text'] for te, l in dd.items() if te not in unused for t in l if te]

    # with open(data_path + "tm_docs.json", "r") as infile:
    #     _documents = json.load(infile)
    # documents = [s for v in _documents.values() for s in v if len(s)<500]
    # print(documents[0])
    sp = WhiteSpacePreprocessing(documents, stopwords_language='english')
    preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess(keep_brackets_underscore=True)

    import pickle
    with open("/cs/snapless/oabend/eitan.wagner/segmentation/tp.pkl", 'rb') as infile:
        tp = pickle.load(infile)

    # tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v1")

    # tp = TopicModelDataPreparation("/cs/snapless/oabend/eitan.wagner/segmentation/models/distilroberta-time-base_model")
    # training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

    # with open("/cs/snapless/oabend/eitan.wagner/segmentation/tp.pkl", 'wb') as outfile:
    #     pickle.dump(tp, outfile)

    import warnings
    warnings.filterwarnings("ignore")
    predict(ctm, tp=tp, texts=texts)


    if False:
        make_docs = False
        use_docs = False
        sf_docs = False
        yf_docs = True
        load = True
        # load = False

        if not sf_docs or not yf_docs:
            nlp = spacy.load("en_core_web_trf")
            nlp_sm = spacy.load("en_core_web_sm", disable=['ner'])
            nlp2 = spacy.load("/cs/snapless/oabend/eitan.wagner/segmentation/ner/model-best")
        nltk.download('stopwords')
        CHUNK_SIZE = 50
        BINS = 10

        # use_docs = True

        original_docs =[]
        data_path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/'
        if yf_docs:
            with open(data_path + "yf_segments.json", "r") as infile:
                documents = json.load(infile)

        if make_docs or use_docs or sf_docs:
            with open(data_path + 'sf_all.json', 'r') as infile:
                dd = json.load(infile)
            topics = [t['terms'] for l in dd.values() for t in l]
            if sf_docs:
                documents = [t['text'] for l in dd.values() for t in l]
        elif not yf_docs:
            # text_file = "dbpedia_sample_abstract_20k_unprep.txt" # EDIT THIS WITH THE FILE YOU UPLOAD
            # text_file = "YF_testimonies.txt" # EDIT THIS WITH THE FILE YOU UPLOAD
            text_file = "/cs/snapless/oabend/eitan.wagner/segmentation/SF_testimony_segments.txt" # EDIT THIS WITH THE FILE YOU UPLOAD
            #
            documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()]
            original_docs = list(documents)
        if make_docs:
            _documents = {}
            for _t, d in tqdm(dd.items()):
                # print(_t)
                text = " ".join([t['text'] for t in d])
                doc = nlp_sm(text)
                # doc2 = nlp2(text)
                sents = [s for s in doc.sents]
                r = list(range(0, len(sents), CHUNK_SIZE)) + [len(sents)]
                bins = [(_r * BINS)//len(sents) for _r in r[:-1]]
                segments_sm = [doc[sents[_r].start:sents[r[i+1]-1].end].text for i, _r in enumerate(r[:-1])]
                segments = [nlp(s) for s in segments_sm]
                # segments2 = [doc2[sents[_r].start:sents[r[i+1]-1].end] for i, _r in enumerate(r[:-1])]  #should be the same but with different ents
                segments2 = [nlp2(s) for s in segments_sm]  #should be the same but with different ents

                # fix this (or the preprocessing)! since the preprocessing will remove _
                segments = [re.sub('[A-Z][A-Z]:', 'SUBJECT:', s.text) +
                            " ".join([e.label_ for e in s.ents if e.label_ != 'GPE'] + [e.label_ for e in s2.ents] + [f"BIN_{b}"])
                            for s, s2, b in zip(segments, segments2, bins)]
                _documents[_t] = segments

                with open(data_path + "tm_docs.json", "w") as outfile:
                    json.dump(_documents, outfile)
                # documents = [[s for s in v] for v in _documents.values()]
                documents = [s for v in _documents.values() for s in v]
        elif use_docs:
            with open(data_path + "tm_docs.json", "r") as infile:
                _documents = json.load(infile)

            # documents = [[s for s in v] for v in _documents.values()]

            documents = [s for v in _documents.values() for s in v]

        if not load:
            sp = WhiteSpacePreprocessing(documents, stopwords_language='english')
            preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess(keep_brackets_underscore=True)

            # tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v1")
            tp = TopicModelDataPreparation("/cs/snapless/oabend/eitan.wagner/segmentation/models/distilroberta-time-base_model")

            training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

        else:
            # SF Vocab size:  1999
            # YF Vocab size:  1998
            path = "/cs/snapless/oabend/eitan.wagner/segmentation/CTM_w_time_yf_40/"
            ctm = CombinedTM(bow_size=1998, contextual_size=768, n_components=40, num_epochs=20)
            ctm.load(model_dir=path+"contextualized_topic_model_nc_40_tpm_0.0_tpv_0.975_hs_prodLDA_ac_(100, 100)_do_softplus_lr_0.2_mo_0.002_rp_0.99", epoch=19)
            for i in range(40):
                ctm.get_wordcloud(topic_id=i, n_words=15, background_color="white", save=path+f"wordclouds/topic{i}")

        # print(tp.vocab[:10])
        from contextualized_topic_models.evaluation.measures import CoherenceNPMI, CoherenceUMASS, CoherenceWordEmbeddings, TopicDiversity
        texts = [d.split() for d in preprocessed_documents]
        # print(preprocessed_documents[:10])
        # ncs = [5, 10, 15, 20, 25, 30, 35]
        # ncs = range(5, 70, 3)
        # ncs = [40]
        ncs = []
        print("Vocab size: ", len(tp.vocab))
        diversities = []
        coherences = []
        for i, nc in enumerate(ncs):
            print("*****************************")
            print(f"Training TM. Num topics: {nc}")
            # ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=nc, num_epochs=20)
            ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=nc, num_epochs=20)
            ctm.fit(training_dataset) # run the model

            # ctm.save(models_dir="/cs/snapless/oabend/eitan.wagner/segmentation/CTM_w_time_yf_40")
            ctm_topics = ctm.get_topic_lists(15)
            print(ctm_topics)
            # cwe = CoherenceWordEmbeddings(topics=topics)
            # print(f"Coherence (word embedding) score: {cwe.score()}")
            td = TopicDiversity(topics=ctm.get_topic_lists(25))
            diversities.append(td.score())
            print(f"Topic diversity score: {td.score()}")
            npmi = CoherenceNPMI(texts=texts, topics=ctm_topics)
            coherences.append(npmi.score())
            print(f"NPMI score: {npmi.score()}")
            # umass = CoherenceNPMI(texts=texts, topics=topics)
            # print(f"UMASS score: {umass.score()}")




        # print("Num Clusters: ", ncs)
        # print("Diversities: ", diversities)
        # print("Coherences: ", coherences)
        # scores = [[n, d, c] for n, d, c in zip(ncs, diversities, coherences)]
        # with open(data_path + "tm_scores_beta3.json", "w") as outfile:
        #     json.dump(scores, outfile)
        # topic_lists = ctm.get_topic_lists(20)
        # topic_lists = [[t for t in t_l if t.upper() not in NERs][:10] for t_l in topic_lists]
        # print(topic_lists)

        topics_predictions = ctm.get_thetas(training_dataset, n_samples=5)  # get all the topic predictions
        # print(len(topics_predictions))
        # print(len(topics_predictions[0]))
        #
        # import numpy as np
        #
        l = []
        #
        # # for i in range(100):
        # for p_d, t_p, t in zip(preprocessed_documents, topics_predictions, topics):
        if yf_docs:
            topics = [""] * len(topics_predictions)
        for d, t_p, t in zip(unpreprocessed_corpus, topics_predictions, topics):
            # print(p_d)
            topic_number = int(np.argmax(t_p)) # get the topic id of the
            # print(topics_number)
            # print(preprocessed_documents[i])
            # print(topics_predictions[i])

            l.append([d, topic_number, ctm.get_topic_lists(10)[topic_number], t])

        s = ""
        if yf_docs:
            s = "_yf"
        elif sf_docs:
            s = "_sf"
        with open(data_path + f"tm_out{s}_40.json", "w") as outfile:
            json.dump(l, outfile)

        # import pyLDAvis as vis
        #
        # lda_vis_data = ctm.get_ldavis_data_format(tp.vocab, training_dataset, n_samples=10)
        #
        # ctm_pd = vis.prepare(**lda_vis_data)
        # vis.display(ctm_pd)

