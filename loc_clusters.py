

import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
import json
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

from evaluation import edit_distance
from sentence_transformers import SentenceTransformer, util

# **************

def make_vectors(data_path, cat=False, conversion_dict=None):
    with open(data_path + 'loc_description_dict.json', 'r') as infile:
        desc_dict = json.load(infile)
        desc_dict["START"] = "The beginning of the testimony."
        desc_dict["END"] = "The end of the testimony."
    if cat:
        with open(data_path + 'loc_category_dict.json', 'r') as infile:
            cat_dict = json.load(infile)
            cat_dict["START"] = "START"
            cat_dict["END"] = "END"
        cat_list = {conversion_dict.get(c): [] for c in set(cat_dict.values())}
        for l, c in cat_dict.items():
            cat_list[conversion_dict.get(c)].append(l)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    if cat:
        dict = {}
        for c, l in cat_list.items():
            # embeddings = model.encode([_l + desc_dict.get(_l, "") for _l in l], convert_to_tensor=True)
            embeddings = model.encode([_l + desc_dict.get(_l, "") for _l in l], convert_to_tensor=False)
            # dict[c] = embeddings.mean(dim=0)
            dict[c] = embeddings.mean(axis=0)
        # return list(dict.keys()), torch.stack(list(dict.values()), dim=0)
        return list(dict.keys()), np.stack(list(dict.values()), axis=0)
        # return list(dict.keys()), list(dict.values())

    # embeddings = model.encode([l+d for l, d in desc_dict.items()], convert_to_tensor=True)
    embeddings = model.encode([l+d for l, d in desc_dict.items()], convert_to_tensor=False)
    return list(desc_dict.keys()), embeddings
    # vector_dict = {loc: model.encode(v) for loc, v in desc_dict.items()}
    # embeddings = model.encode(sentences)


def find_closest(locs=None, vectors=None, c_vector=None, tensor=False):
    sims = util.cos_sim(vectors, c_vector)
    # return locs[np.argmax(sims)]
    if tensor:
        return torch.argmax(sims)
    return int(np.argmax(sims))


class SBertEncoder(nn.Module):
    def __init__(self, vectors=None, data_path=None, cat=False, conversion_dict=None):
        super().__init__()
        if vectors is not None:
            self.classes_ = vectors[0]
            self.vectors = vectors[1]
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            if data_path is None:
                base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
                data_path = base_path + 'data/'
            with open(data_path + 'loc_description_dict.json', 'r') as infile:
                desc_dict = json.load(infile)
                desc_dict["START"] = "The beginning of the testimony."
                desc_dict["END"] = "The end of the testimony."
                self.desc_dict = desc_dict
            if cat:
                with open(data_path + 'loc_category_dict.json', 'r') as infile:
                    cat_dict = json.load(infile)
                    cat_dict["START"] = "START"
                    cat_dict["END"] = "END"
                cat_list = {conversion_dict.get(c): [] for c in set(cat_dict.values())}
                for l, c in cat_dict.items():
                    cat_list[conversion_dict.get(c)].append(l)
                self.cat_list = cat_list
                dict = {}
                for c, l in cat_list.items():
                    # embeddings = model.encode([_l + desc_dict.get(_l, "") for _l in l], convert_to_tensor=True)
                    embeddings = self.model.encode([_l + desc_dict.get(_l, "") for _l in l], convert_to_tensor=True)
                    dict[c] = embeddings.mean(dim=0)
                self.classes_, self.vectors = list(dict.keys()), torch.stack(list(dict.values()), dim=0)
                self.vectors.requires_grad = True

        # self.vector_dict = {c: v for c, v in zip(vectors)}

    def inverse_transform(self, v_labels):
        """
        from label numbers to label strings
        :param labels:
        :param tensors:
        :return:
        """
        return [self.classes_[id] for id in v_labels]

    def inverse_transform2(self, v_labels, tensors=True):
        """
        from label vectors to label strings
        :param labels:
        :param tensors:
        :return:
        """
        label_ids = torch.stack([find_closest(vectors=self.vectors, c_vector=v, tensor=True) for v in v_labels])
        return [self.classes_[id] for id in label_ids]

    def transform(self, labels):
        """
        from label strings to label number
        :return:
        """
        return [self.classes_.index(l) for l in labels]

    def transform2(self, labels):
        """
        from label strings to label vectors
        :return:
        """
        return [self.vectors[self.classes_.index(l)] for l in labels]



# *************

def make_loc_conversion(data_path):
    df = pd.read_csv(data_path + "loc_conversion.csv", names=["from", "to"])
    return dict(zip(df["from"], df["to"]))

def remove_duplicates(sequences):
    from itertools import groupby
    return [[_s[0] for _s in groupby(s)] for s in sequences]

def clustering(sequences, dist_func):
    similarity = -1 * np.array([[dist_func(s1, s2) for s1 in sequences] for s2 in tqdm(sequences)])

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, preference=-80)
    affprop.fit(similarity)

    centroids = {}
    for cluster_id in np.unique(affprop.labels_):
        exemplar = sequences[affprop.cluster_centers_indices_[cluster_id]]
        # cluster = np.unique(sequences[np.nonzero(affprop.labels_ == cluster_id)])
        in_cluster_ids = np.nonzero(affprop.labels_ == cluster_id)[0]

        centroids[cluster_id] = in_cluster_ids[np.argmax([np.mean(similarity[id, in_cluster_ids]) for id in in_cluster_ids])]

    print("Centriods:")
    for id, c in centroids.items():
        print(id)
        print(sequences[c])
    print("Done")

def load_data(data_path, train_size=0.8, conversion_dict=None):
    from dates import _make_texts
    with open(data_path + "sf_unused5.json", 'r') as infile:
        unused = json.load(infile) + ['45064']
    with open(data_path + "sf_five_locs.json", 'r') as infile:
        unused = unused + json.load(infile)
    # with open(data_path + "locs_w_cat.json", 'r') as infile:
    with open(data_path + "locs_segments_w_cat.json", 'r') as infile:
        data = json.load(infile)
        for u in unused:
            data.pop(u, None)

        train_data = {t: text for t, text in list(data.items())[:int(train_size * len(data))][:]}  # !!!!

    labels = [_make_texts({t: t_data}, unused, "1", conversion_dict=conversion_dict)[1] for t, t_data in train_data.items()]
    return labels

def main():
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    data_path = base_path + 'data/'

    model_name = "deberta11"
    model_path = "/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/" + model_name

    encoder = joblib.load(model_path + "/label_encoder.pkl")
    c_dict = None
    c_dict = make_loc_conversion(data_path=data_path)
    data = remove_duplicates(load_data(data_path, conversion_dict=c_dict))
    clustering(sequences=data, dist_func=edit_distance)

def main2():
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    data_path = base_path + 'data/'
    c_dict = make_loc_conversion(data_path=data_path)
    locs, vectors = make_vectors(data_path, cat=True, conversion_dict=c_dict)
    # locs, vectors = make_vectors(data_path)
    cat2id = {c: i for i, c in enumerate(locs)}
    cor_matrix = util.cos_sim(vectors, vectors).detach().numpy()
    # cor_matrix.detach().numpy()
    _c_m = {(s1, s2): cor_matrix[cat2id[s1], cat2id[s2]] for s1 in locs for s2 in locs}
    data = remove_duplicates(load_data(data_path, conversion_dict=c_dict))
    clustering(sequences=data, dist_func=lambda s1, s2: edit_distance(s1, s2, cor_matrix=_c_m))
    print("Done")

def main3():
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    data_path = base_path + 'data/'
    c_dict = make_loc_conversion(data_path=data_path)
    locs, vectors = make_vectors(data_path, cat=True, conversion_dict=c_dict)

    model_name = "vector_distilroberta6"
    model_path = "/cs/snapless/oabend/eitan.wagner/segmentation/models/locations/" + model_name
    from dates import train_classifier
    train_classifier(data_path, out_path=model_path, first_train_size=0., val_size=0.1, test_size=0.1,
                     conversion_dict=c_dict, vectors=vectors, matrix=True)

if __name__ == "__main__":
    main2()