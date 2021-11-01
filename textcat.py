
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from scipy.stats import poisson
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

import logging
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler

import joblib
from spacy.tokens import DocBin
from spacy.vocab import Vocab

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import parse_sf  # this should set all the extensions too

VECTOR_LEN = 768
NUM_VECTORS = 2
BINS = 10
ENTS = 18


class PartialSVM(SVC):
    # for stacking
    def __init__(self, feature_types="tfidf", probability=True, class_weight='balanced', C=1):
        logging.info(f"SVM C: {C}")
        super().__init__(probability=probability, class_weight=class_weight, C=C)
        # self.model = svm.SVC(kernel='rbf', probability=True, class_weight='balanced', C=4)
        self.feature_types = feature_types

    def _transform(self, X):
        if self.feature_types == 'bins':
            X = X[:, -BINS:]
        elif self.feature_types == 'vectors':
            # X = X[:, -778:-10]
            X = X[:, -BINS - VECTOR_LEN * NUM_VECTORS:-BINS]
        elif self.feature_types == 'ents':  # vectors and bin
            # X = X[:, -796:-778]
            # X = X[:, -4171:-4153]  # !!!!
            X = X[:, -ENTS - VECTOR_LEN * NUM_VECTORS - BINS:- VECTOR_LEN * NUM_VECTORS - BINS]  # !!!!
        elif self.feature_types == 'tfidf':
            # X = X[:, :-4171] # !!!!!
            X = X[:, :-ENTS - VECTOR_LEN * NUM_VECTORS - BINS] # !!!!!
        return X

    def fit(self, X, y):
        super().fit(self._transform(X), y)
        # self.model.fit(X,y)

    def predict_proba(self, X):
        # X is two dimensional!!
        return super().predict_proba(self._transform(X))
        # return self.model.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X))



class SVMTextcat:
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/', extra_features='ents_bin'):
        self.extra_features = extra_features
        self.model = None
        self.vectorizer = None
        self.base_path = base_path
        self.topics = None
        self.prior_length = None

    def from_path(self, model_path='models/svm.pkl', vectorizer_path='models/vectorizer.pkl'):
        self.model = joblib.load(self.base_path + model_path)
        self.vectorizer = joblib.load(self.base_path + vectorizer_path)

        # self.topics = self.model.classes_  # do we need to transform this??
        self.topics = self.vectorizer.encoder.classes_  # these are the strings. Is topics[0] the same is label_inverse_transform(0)??
        return self

    def fit(self, data_path=None, test=False):
        dataset = TextcatDataset(self.base_path + data_path, self.extra_features)
        self.vectorizer = dataset.vectorizer
        # self.model = train_svm(dataset, random_state=42, out_path=self.base_path + "models/")  # !!!
        if self.extra_features != 'ents_srls_bin2':
        # if True:
            self.model = train_svm(dataset, random_state=42, out_path=None)  # !!!
        else:
            self.model = train_svm(dataset, random_state=42, out_path=self.base_path + 'models/')
        if test:
            test_model(self.model, dataset, out_path=self.base_path + 'models/results/')

        self.topics = self.vectorizer.encoder.classes_  # these are the strings
        return self

    def predict(self, span):
        # returns probabilities for a Span, as a tuple of (marginal_likelihood, classification_probability)
        x = self.vectorizer.transform([span.text], [span._.feature_vector])  # this is 2 dimensional!

        # do we need P(t)?
        pred_logp = np.ravel(self.model.predict_log_proba(x))  # logP(t|x,s)
        vocab_size = len(self.vectorizer.tfidf.vocabulary_)
        lm_logp = -len(span) * np.log(vocab_size)  # logP(x|s)
        len_logp = poisson.logpmf(len(span)//10, self.prior_length//10)  # logP(s). Check for a more accurate prior!!
        return logsumexp(pred_logp + lm_logp + len_logp), pred_logp

    def predict_max(self, span, prev_topic=None):
        # returns a tuple for the probability for best topic and index of the best topic, and also the probs for classification
        x = self.vectorizer.transform([span.text], [span._.feature_vector])  # this is 2 dimensional!
        pred_logp = np.ravel(self.model.predict_log_proba(x))  # logP(t|x,s)
        if prev_topic is not None:
            pred_logp[prev_topic] = -np.inf  # no need to normalize since we choose the max. yes we do!!
            pred_logp = pred_logp - logsumexp(pred_logp)
        vocab_size = len(self.vectorizer.tfidf.vocabulary_)
        lm_logp = -len(span) * np.log(vocab_size)  # logP(x|s)
        len_logp = poisson.logpmf(len(span)//10, self.prior_length//10)  # logP(s). Check for a more accurate prior!!
        logp = pred_logp + lm_logp + len_logp
        max_p = int(np.argmax(logp))
        return logp[max_p], pred_logp, max_p

    def predict_raw(self, text):
        # returns classifier probabilities for a plain text. without using extra features
        x = self.vectorizer.transform([text], None)
        return np.ravel(self.model.predict_log_proba(x))

    def find_priors(self):
        # TODO: use new data. Or at least calculate real prior!!
        # this is the average number of *words*
        with open(self.base_path + 'data/title_w_segments.json', 'r') as infile:
            title_w_segments = json.load(infile)

        lengths = [len(text.split(" ")) for _, text in title_w_segments]
        self.prior_length = sum(lengths) / len(title_w_segments)  # the average length. not exact!!!
        logging.info("calculated priors")


class Vectorizer:
    # creates vectors for texts
    # the text should be spaCy Spans
    # this also deals with the labels
    def __init__(self, extra_features='ents_bin'):
        self.use_extra_features = extra_features
        self.tfidf = TfidfVectorizer()
        self.encoder = LabelEncoder()

        self.vectorizer = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()
        self.std_scaler = StandardScaler()
        self.max_scaler = MaxAbsScaler()

    def fit_transform(self, texts, extra_features):
        # texts = [s.text for s in spans]
        # extra_features = [s._.feature_vectors for s in spans]  # any span should have this
        # this does not use scaling!!!

        X = self.tfidf.fit_transform(texts).astype(dtype=float)
        if self.use_extra_features == 'all':
            # only here we will scale all data
            # vecs = self.std_scaler.fit_transform(np.array(extra_features, dtype=float)[:, -778:-10])
            # ents_srls = self.max_scaler.fit_transform(np.array(extra_features, dtype=float)[:, :-778])
            vecs = self.std_scaler.fit_transform(np.array(extra_features, dtype=float)[:, -BINS - VECTOR_LEN*NUM_VECTORS:-BINS])
            ents_srls = self.max_scaler.fit_transform(np.array(extra_features, dtype=float)[:, :-BINS - VECTOR_LEN*NUM_VECTORS])
            X = hstack((X, ents_srls, vecs, np.array(extra_features, dtype=float)[:, -BINS:]))
        elif self.use_extra_features == 'bins':
            X = hstack((X, np.array(extra_features, dtype=float)[:, -10:]))
        elif self.use_extra_features == 'vectors':  # vectors and bin
            X = np.array(extra_features, dtype=float)[:, -778:]
        if self.use_extra_features == 'bin_only':
            X = np.array(extra_features, dtype=float)[:, -10:]
        elif self.use_extra_features == 'vectors_only':
            X = np.array(extra_features, dtype=float)[:, -778:-10]
        elif self.use_extra_features == 'ents_only':  # vectors and bin
            X = np.array(extra_features, dtype=float)[:, :18]
        elif self.use_extra_features == 'ents_srls_bin':  # but no vectors
            X = hstack((X, np.array(extra_features, dtype=float)[:, :-778], np.array(extra_features, dtype=float)[:, -10:]))
        elif self.use_extra_features == 'ents_bin':
            X = self.tfidf.fit_transform(texts).astype(dtype=float)
            X = hstack((X, np.array(extra_features, dtype=float)[:, :18], np.array(extra_features, dtype=float)[:, -10:]))
        elif self.use_extra_features == 'ents_bin2':
            X = hstack((self.vectorizer.fit_transform(texts), np.array(extra_features, dtype=float)[:, :18], np.array(extra_features, dtype=float)[:, -10:]))
            X = csr_matrix(np.nan_to_num(X.todense()), dtype=float)
            X = self.tfidf_transformer.fit_transform(X)
        elif self.use_extra_features == 'ents_srls_bin2':
            X = hstack((self.vectorizer.fit_transform(texts), np.array(extra_features, dtype=float)[:, :-778], np.array(extra_features, dtype=float)[:, -10:]))
            X = csr_matrix(np.nan_to_num(X.todense()), dtype=float)
            X = self.tfidf_transformer.fit_transform(X)
        elif self.use_extra_features == 'bin2':
            X = hstack((self.vectorizer.fit_transform(texts), np.array(extra_features, dtype=float)[:, -10:]))
            X = csr_matrix(np.nan_to_num(X.todense()), dtype=float)
            X = self.tfidf_transformer.fit_transform(X)
        # here we add features
        return X

    def transform(self, texts, extra_features):  # TODO: fix!!!!
        # texts = [s.text for s in spans]
        # extra_features = [s._.feature_vectors for s in spans]  # any span should have this
        X = self.tfidf.transform(texts)
        if extra_features is None:
            return hstack((X, np.zeros((len(texts), 28))))
        # here we add features
        if self.use_extra_features == 'all':
            # only here we will scale all data
            # vecs = self.std_scaler.transform(np.array(extra_features, dtype=float)[:, -778:-10])
            # ents_srls = self.max_scaler.transform(np.array(extra_features, dtype=float)[:, :-778])
            # X = hstack((X, ents_srls, vecs, np.array(extra_features, dtype=float)[:, -10:]))
            vecs = self.std_scaler.transform(
                np.array(extra_features, dtype=float)[:, -BINS - VECTOR_LEN * NUM_VECTORS:-BINS])
            ents_srls = self.max_scaler.transform(
                np.array(extra_features, dtype=float)[:, :-BINS - VECTOR_LEN * NUM_VECTORS])
            X = hstack((X, ents_srls, vecs, np.array(extra_features, dtype=float)[:, -BINS:]))
        elif self.use_extra_features == 'bins':
            X = hstack((X, np.array(extra_features, dtype=float)[:, -10:]))
        elif self.use_extra_features == 'vectors':  # vectors and bin
            X = np.array(extra_features, dtype=float)[:, -778:]
        elif self.use_extra_features == 'ents_srls_bin':  # but no vectors
            X = hstack((X, np.array(extra_features, dtype=float)[:, :-778], np.array(extra_features, dtype=float)[:, -10:]))
        elif self.use_extra_features == 'ents_bin':
            X = self.tfidf.transform(texts).astype(dtype=float)
            X = hstack((X, np.array(extra_features, dtype=float)[:, :18], np.array(extra_features, dtype=float)[:, -10:]))
        elif self.use_extra_features == 'ents_bin2':
            X = hstack((self.vectorizer.transform(texts), np.array(extra_features, dtype=float)[:, :18], np.array(extra_features, dtype=float)[:, -10:]))
            X = csr_matrix(np.nan_to_num(X.todense()), dtype=float)
            X = self.tfidf_transformer.transform(X)
        elif self.use_extra_features == 'ents_srls_bin2':
            X = hstack((self.vectorizer.transform(texts), np.array(extra_features, dtype=float)[:, :-778], np.array(extra_features, dtype=float)[:, -10:]))
            X = csr_matrix(np.nan_to_num(X.todense()), dtype=float)
            X = self.tfidf_transformer.transform(X)
        elif self.use_extra_features == 'bin2':
            X = hstack((self.vectorizer.transform(texts), np.array(extra_features, dtype=float)[:, -10:]))
            X = csr_matrix(np.nan_to_num(X.todense()), dtype=float)
            X = self.tfidf_transformer.transform(X)
        return X

    def label_fit(self, y):
        # learns label encoding (from string to id)
        self.encoder.fit(y)
        return self

    def label_transform(self, label):
        # transform string to id
        return self.encoder.transform(label)

    def label_inverse_transform(self, y):
        # converts a response vector back to the original label strings
        return self.encoder.inverse_transform(y)


# dataset definition
class TextcatDataset:
    # hold the dataset and performs transforming and splitting
    def __init__(self, data_path, extra_features=None):
        self.path = data_path  # this is  in data/docs/
        # load the data
        # with open(data_path + "data1.json", 'r') as infile:
        #     data = json.load(infile)
        with open(data_path + "data2.json", 'r') as infile:
            data = json.load(infile)
        # with open(data_path + "data2_2.json", 'r') as infile:
        #     data1 = json.load(infile)
        # data.update(data1)

        # doc_bin = DocBin().from_disk(path + "docs/data.spacy")
        # vocab = Vocab().from_disk(self.path + "docs/vocab")
        # data = [(segment, segment._.real_topic) for doc in doc_bin.get_docs(vocab) for segment in doc.spans["segments"]]
        data = [data_tuple for t_data in data.values() for data_tuple in t_data]
        # np.random.shuffle(data)
        # store the inputs and outputs
        texts, vectors, self.y = list(zip(*data))
        # self.vectorizer = TfidfVectorizer()
        self.vectorizer = Vectorizer(extra_features)
        self.X = self.vectorizer.fit_transform(texts, vectors)
        self.X = csr_matrix(self.X, dtype=float)

        # label encode target and ensure the values are floats
        self.y = [t[0] for t in self.y]
        self.vectorizer.label_fit(self.y)
        self.y = self.vectorizer.label_transform(self.y).astype(dtype=float)
        # logging.info(f"y_shape {self.y}")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # self.X_train = self.X[:int(0.8*self.X.shape[0]), :]
        # self.X_test = self.X[int(0.8*self.X.shape[0]):, :]
        # self.y_train = self.y[:int(0.8*len(self.y))]
        # self.y_test = self.y[int(0.8*len(self.y)):]

        self.dim = self.X.shape[1]

    def get_docs(self):
        # vocab = Vocab().from_disk(self.path + "vocab")
        with open('doc_names.json', "r") as infile:
            names = json.load(infile)
        # docs = [Doc(Vocab()).from_disk(name) for name in names]
        docs = []
        for name in names:
            with open(self.path + name) as infile:
                docs.append(pickle.load(infile))
        return docs

    # number of rows in the dataset
    def __len__(self):
        return self.X.shape[0]

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2, random_state=42):
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=random_state)
        # return X_train, X_test, y_train, y_test
        return self.X_train, self.X_test, self.y_train, self.y_test

    def id2lables(self, y):
        # converts a response vector back to the original label strings
        return self.vectorizer.encoder.inverse_transform(self.y)


def train_svm(dataset, random_state=42, out_path=None):
    # trains an SVM model on the given dataset
    X_train, X_test, y_train, y_test = dataset.get_splits(0.2, random_state=random_state)
    X_train = csr_matrix(np.nan_to_num(X_train.todense()), dtype=float)
    X_test = csr_matrix(np.nan_to_num(X_test.todense()), dtype=float)
    v = np.nan_to_num(X_train.todense()).var()

    # clf = svm.SVC(kernel='poly', probability=True, gamma='scale', class_weight='balanced')
    # clf = svm.SVC(kernel='rbf', probability=True, gamma='scale', class_weight='balanced', C=2)
    clf = svm.SVC(kernel='rbf', probability=True, gamma=1/(2*X_train.shape[1] * v), class_weight='balanced', C=4)
    clf = svm.SVC(kernel='linear', probability=True, gamma='scale', class_weight='balanced')
    # Train the model using the training sets
    logging.info("Training SVM")
    # logging.info(f"X_shape {X_train.shape}")
    # logging.info(f"y_shape {y_train.shape}")

    # logging.info(f"nan: {len(np.argwhere(np.isnan(X_train.todense())))}")
    # logging.info(f"finite: {np.all(np.isfinite(X_train.todense()))}")
    # X_train = X_train.todense()
    # X_train[np.argwhere(np.isnan(X_train))] = 0
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_train)
    logging.info(f"SVM Accuracy (train): {accuracy_score(y_train, y_pred)}")
    y_pred = clf.predict(X_test)
    logging.info(f"SVM Accuracy (test): {accuracy_score(y_test, y_pred)}")
    # save
    if out_path is not None:
        joblib.dump(clf, out_path + 'svm.pkl')
        joblib.dump(dataset.vectorizer, out_path + 'vectorizer.pkl')
    return clf

def ensemble(docs_path, Cs = (4,4,4,4,0.5)):
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    dataset = TextcatDataset(docs_path, extra_features="all")
    X_train, X_test, y_train, y_test = dataset.get_splits(0.2, random_state=42)
    X_train = csr_matrix(np.nan_to_num(X_train.todense()), dtype=float)
    X_test = csr_matrix(np.nan_to_num(X_test.todense()), dtype=float)

    estimators = [('svm1', PartialSVM(feature_types="tfidf", C=float(Cs[0]))),
                  ('svm2', PartialSVM(feature_types="bins", C=float(Cs[1]))),
                  ('svm3', PartialSVM(feature_types="ents", C=float(Cs[2]))),
                  ('svm4', PartialSVM(feature_types="vectors", C=float(Cs[3])))]
    clf = StackingClassifier(estimators=estimators,
                             final_estimator=LogisticRegression(max_iter=300, solver='saga', C=float(Cs[4])))

    logging.info("Training ensemble - saga")
    clf.fit(X_train, y_train)

    at_k = eval_at_k(model=clf, X=X_train, y=y_train, k=len(clf.classes_))
    logging.info(f"ensemble Accuracy (train): {at_k}")
    at_k = eval_at_k(model=clf, X=X_test, y=y_test, k=len(clf.classes_))
    logging.info(f"ensemble Accuracy (test): {at_k}")
    #Predict the response for test dataset
    # y_pred = clf.predict(X_train)
    # y_pred = clf.predict(X_test)
    return

def eval_at_k(model, X, y, k):
    def find_k(arr):
        # the label is the last entry in the array
        return np.where(arr[:-1] == arr[-1])[0][0]
    preds = model.predict_proba(X)
    _sorted = np.argsort(preds, axis=1)
    ks = np.apply_along_axis(func1d=find_k, axis=1, arr=np.vstack((_sorted.T, y)).T)
    accs = [sum(ks <= _k) / len(ks) for _k in range(k)]
    return accs

def test_model(model, dataset, random_state=42, out_path=None):
    X_train, X_test, y_train, y_test = dataset.get_splits(0.2, random_state=random_state)
    X_train = csr_matrix(np.nan_to_num(X_train.todense()), dtype=float)
    X_test = csr_matrix(np.nan_to_num(X_test.todense()), dtype=float)
    labels = dataset.vectorizer.encoder.classes_
    import pandas as pd

    train_conf = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)
    eval_conf = pd.DataFrame(np.zeros((len(labels), len(labels))), index=labels, columns=labels)

    # train_texts = list(zip(X_train, dataset.encoder.inverse_transform(y_train)))
    # eval_texts = list(zip(X_test, dataset.encoder.inverse_transform(y_test)))
    y_labels_train = dataset.vectorizer.encoder.inverse_transform(y_train)
    y_labels_eval = dataset.vectorizer.encoder.inverse_transform(y_test)
    train_correct = 0
    eval_correct = 0
    # column is the predicted and row is the real. NO!!! the opposite!
    preds = dataset.vectorizer.encoder.inverse_transform(model.predict(X_train))
    for i, p in enumerate(preds):
        train_conf[y_labels_train[i]][p] += 1

    preds = dataset.vectorizer.encoder.inverse_transform(model.predict(X_test))
    for i, p in enumerate(preds):
        eval_conf[y_labels_eval[i]][p] += 1

    if out_path:
        train_conf.to_csv(out_path + f"train_conf_svm.csv")
        eval_conf.to_csv(out_path + f"eval_conf_svm.csv")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        Cs = sys.argv[1:]
    else:
        # tfidf, bins, ents, vecs, lr
        Cs = [4, 4, 4, 0.5, 0.5]

    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })

    logging.info("linear kernel:")
    # logging.info("RBF with C=4 and gamma/2")
    # logging.info(f"VECTORS: {NUM_VECTORS}")
    # logging.info(f"Cs: {Cs}")
    path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    docs_path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/docs/'
    # ensemble(docs_path=docs_path, Cs=Cs)
    logging.info("\nTraining - no extra features")
    model = SVMTextcat(base_path=path, extra_features="")
    model.fit(data_path='data/docs/')
    # #
    # logging.info("\nTraining - only bins in tfidf")
    # model = SVMTextcat(base_path=path, extra_features='bin2')
    # model.fit(data_path='data/docs/')
    #
    # logging.info("\nTraining - ents and bins in tfidf")
    # model = SVMTextcat(base_path=path, extra_features='ents_bin2')
    # model.fit(data_path='data/docs/')

    # logging.info("\nTraining - ents, srls and bins in tfidf")
    # model = SVMTextcat(base_path=path, extra_features='ents_srls_bin2')
    # model.fit(data_path='data/docs/')

    # logging.info("Training - all extra features")
    # model = SVMTextcat(base_path=path, extra_features='all')
    # model.fit(data_path='data/docs/')
    #
    # logging.info("Training - only bin")
    # model = SVMTextcat(base_path=path, extra_features='bin_only')
    # model.fit(data_path='data/docs/')
    #
    # logging.info("Training - only ents")
    # model = SVMTextcat(base_path=path, extra_features='ents_only')
    # model.fit(data_path='data/docs/')
    #
    # logging.info("Training - only vectors and no bin")
    # model = SVMTextcat(base_path=path, extra_features='vectors_only')
    # model.fit(data_path='data/docs/')


    #
    # dataset = TextcatDataset(path)
    # model = train_svm(dataset, random_state=42, out_path='/cs/snapless/oabend/eitan.wagner/segmentation/models/')
    # # now you can save it to a file
    # # model = joblib.load('/cs/snapless/oabend/eitan.wagner/segmentation/models/svm.pkl')
    # test_model(model, dataset, random_state=42, out_path='/cs/snapless/oabend/eitan.wagner/segmentation/models/')
