
import numpy as np
from pomegranate import MarkovChain
from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable
import json
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import logging
from scipy.special import logsumexp


class MC:
    """
    Class for a Markov Chain over the topics
    """
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/', name="models/transitions/mc.json"):
        self.base_path = base_path
        if name != '':
            with open(base_path + name, 'r') as f:
                self.mc = MarkovChain.from_json(f.read())
        else:
            self.mc = None

        encoder_path = base_path + '/models/xlnet-large-cased/'
        self.encoder = joblib.load(encoder_path + "label_encoder.pkl")  # this does not have a "begin" and "end" label

    def fit(self, chains, out_name=None):
        """
        Fit from a list of chains
        :param chains: list of topic-lists
        :param out_name:
        :return: self
        """
        # create uniform and then update
        dict = {i: 1 / len(self.encoder.classes_) for i in range(len(self.encoder.classes_))}
        d1 = DiscreteDistribution(dict)
        t_list = [[i, j, 1 / (len(self.encoder.classes_) - 1)] if i != j else [i, j, 0]
                  for i in range(len(self.encoder.classes_)) for j in range(len(self.encoder.classes_))]
        d2 = ConditionalProbabilityTable(t_list, [d1])
        self.mc = MarkovChain([d1, d2])

        self.mc.fit(chains, inertia=0.5)
        # self.mc = MarkovChain.from_samples(chains)
        if out_name:
            with open(base_path + out_name, 'w+') as f:
                f.write(self.mc.to_json())
        return self

    def predict(self, topic, prev_topic, encoded=True):
        """
        Predict the probability for a topic given a previous topic
        :param topic:
        :param prev_topic:
        :param encoded:
        :return: the probability
        """
        if not encoded:
            topic, prev_topic = self.encoder.transform([topic, prev_topic])

        return self.mc.distributions[1].parameters[0][self.mc.distributions[1].keymap[(prev_topic, topic)]][2]

    def predict_vector(self, prev_topic, encoded=True):
        """
        Predict the probability vector for a given previous topic
        :param prev_topic: previous topic. if -1 then this is the first so give the initial probabilities
        :param encoded:
        :return: list of probabilities
        """
        if not encoded:
            prev_topic = self.encoder.transform(prev_topic)

        if prev_topic == -1:
            return list(self.mc.distributions[0].parameters[0].values())

        first = self.mc.distributions[1].keymap[(prev_topic, 0)]
        last = first + len(self.encoder.classes_)  # do not include the last one here

        return [p for _, _, p in self.mc.distributions[1].parameters[0][first:last]]

    def sample(self, k):
        """
        Sample a Markov chain of length k
        :param k:
        :return:
        """
        return self.mc.sample(k)


def make_chains(base_path, save=False):
    """
    Makes topic-chains from the SF data
    :param base_path:
    :param save: whether to save to file
    :return: list of topic-lists
    """
    encoder_path = base_path + '/models/xlnet-large-cased/'
    encoder = joblib.load(encoder_path + "label_encoder.pkl")  # this does not have a "begin" and "end" label

    docs_path = base_path + '/data/docs/'
    with open(docs_path + "data2.json", 'r') as infile:
        data = json.load(infile)

    topic_lists = [encoder.transform(np.ravel([_d[2] for _d in d])).tolist() for _, d in data.items()]
    for l in topic_lists:
        for i in range(len(l)-1, 0, -1):
            if l[i] == l[i-1]:
                l.pop(i)

    if save:
        print(topic_lists)
        with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/chains.json', "w+") as outfile:
            json.dump(topic_lists, outfile)
        print(encoder.classes_)
        with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/topics.json', "w+") as outfile:
            json.dump(encoder.classes_.tolist(), outfile)

    return topic_lists


def get_topic_list(base_path):
    return [t for c in make_chains(base_path=base_path) for t in c]

# not used
def make_transition_matrix(base_path, out_name="models/transitions/mc.json"):
    # should we apply smoothing?

    chains = make_chains(base_path)
    mc = MarkovChain.from_samples(chains)

    with open(base_path + out_name, 'w+') as f:
        f.write(mc.to_json())

    return mc


class MCClusters:
    """
    Class for clustering as a mixture of Markov Chains
    """
    def __init__(self, k=10):
        self.k = k
        self.mcs = [None] * k
        self.sets = [None] * k  # each set is a list of chain indices
        self.chains = None
        self.id2set = []
        self.ll = None

        self.base_path='/cs/snapless/oabend/eitan.wagner/segmentation/'
        encoder_path = self.base_path + '/models/xlnet-large-cased/'
        self.encoder = joblib.load(encoder_path + "label_encoder.pkl")  # this does not have a "begin" and "end" label

    def _chain2topics(self, chain):
        return self.encoder.inverse_transform(chain)

    def save_sets(self, name=''):
        """
        Save sets of chains as a dict by testimony
        :param name:
        :return:
        """
        sets = {i: [self._chain2topics(self.chains[j]).tolist() for j in js] for i, js in enumerate(self.sets)}
        if name == '':
            name = f'topic_chains_{self.k}.json'
        with open(f'/cs/snapless/oabend/eitan.wagner/segmentation/data/'+name, "w+") as outfile:
            json.dump(sets, outfile)

    def load(self):
        """
        Load data from file
        :return:
        """
        name = f'topic_chains_{self.k}.json'
        with open(f'/cs/snapless/oabend/eitan.wagner/segmentation/data/'+name, "r") as infile:
            dict = json.load(infile)
            self.sets = [[self.encoder.transform(l) for l in s] for s in dict.values()]
        with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/chains.json', "r") as infile:
            self.chains = json.load(infile)
        self._fit_on_sets(self.sets, weighted=True)
        return self

    def _init_sets(self, X):
        """
        initialize the clustering with k-means on the count vector
        :param X:
        :return:
        """
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(X)
        for i, l in enumerate(kmeans.labels_):
            if self.sets[int(l)] is None:
                self.sets[int(l)] = list()
            self.sets[int(l)].append(i)
            self.id2set.append(int(l))

        #return a copy
        return [list(s) for s in self.sets]

    def _fit_on_sets(self, sets, weighted=False, idxs=None):
        """
        Fit the MCs with a given set of sets
        :param sets:
        :param weighted:
        :param idxs:
        :return: the likelihood for this partition (after fitting)
        """
        if idxs is not None:
            for i in idxs:
                if len(sets[i]) >= 0:
                    self.mcs[i] = MC().fit([self.chains[s] for s in sets[i]])
            sets = [sets[i] for i in idxs]
            mcs = [self.mcs[i] for i in idxs]
            chains = [self.chains[j] for s in sets for j in s]

        else:
            for i in range(len(sets)):
                if len(sets[i]) >= 0:
                    self.mcs[i] = MC().fit([self.chains[s] for s in sets[i]])
            mcs = self.mcs
            chains = self.chains

        if weighted:
            # log_weights = np.log([len(s) / len(self.chains) for s in sets])
            weights = np.array([len(s) / len(self.chains) for s in sets])
            # logging.info(weights)
            # log_probs = [np.array([mc.mc.log_probability(chain) for mc in self.mcs]) for chain in self.chains]
            log_probs = [logsumexp(a=np.array([mc.mc.log_probability(chain) for mc in mcs]), b=weights) for chain in chains]
            # logging.info(log_probs)
            return sum(log_probs)
            # return sum([log_weights @ np.array([mc.mc.log_probability(chain) for mc in self.mcs]) for chain in self.chains])

        log_prob = sum([self.mcs[i].mc.log_probability(self.chains[j]) for i, s in enumerate(sets) for j in s])
        # log_prob = sum([self.mcs[i].mc.log_probability(self.chains[c]) for i in self.id2set for c in sets(i)])
        return log_prob

    def fit(self, chains, iterations=20, weighted=False):
        """
        Fit the mixture model
        :param chains:
        :param iterations:
        :param weighted:
        :return: self
        """
        self.chains = chains

        # initial clustering using k-means
        X = np.zeros((len(chains), len(self.encoder.classes_)))
        for i, chain in enumerate(chains):
            for j, l in enumerate(chain):
                X[i, l] += 1
        sets = self._init_sets(X)

        ll = self._fit_on_sets(sets, weighted=weighted)
        for t in range(iterations):  # iterations
            logging.info(f"ll: {ll}")
            logging.info(f"**************************************************** {t} ")
            # fit mcs
            # js = list(range(0, len(chains), 2))  # these are chain-index pairs
            js = list(range(len(chains)))  # these are chain-index pairs
            set_idxs = list(range(self.k))  # these are chain-index pairs
            np.random.shuffle(js)

            # predict with mcs
            # we need to check with each cluster if it's better to change!!!!
            for j in js:
                # for j1, j2 in zip(js1, js2):
                for s_i in set_idxs:
                    if self.id2set[j] == s_i:
                        continue

                    # score for the 2 sets considered, before swapping. This is NOT GOOD!!
                    # ll2sets = self._fit_on_sets(sets, idxs=[self.id2set[j], s_i], weighted=weighted)

                    sets[self.id2set[j]].remove(j)
                    sets[s_i].append(j)

                    ll2 = self._fit_on_sets(sets, weighted=weighted)
                    # ll2sets2 = self._fit_on_sets(sets, idxs=[self.id2set[j], s_i], weighted=weighted)

                    if ll2 > ll:
                        ll = ll2
                    # if ll2sets2 > ll2sets:
                        ll = ll2
                        # print(f"ll: {ll}")
                        self.id2set[j] = s_i

                    else:
                        sets[s_i].remove(j)
                        sets[self.id2set[j]].append(j)

        logging.info(f"ll: {ll}")
        self.ll = ll
        self.sets = sets
        return self

    def predict(self, chain):
        """
        predict best cluster (markov chain)
        :param chain:
        :return: index of best cluster for a new chain
        """
        log_probs = [mc.mc.log_probability(chain) if self.mcs is not None else -np.inf for mc in self.mcs]
        m = np.argmax(log_probs)
        return m

    def sample(self, k):
        """
        Sample a length k chain of topic. First choose a chain and then sample from it.
        :param k:
        :return:
        """
        weights = np.array([len(s) / len(self.chains) for s in self.sets])  # should sum up to 1
        mc = np.random.choice(self.mcs, 1, p=weights).item()
        return mc.sample(k)

    def predict_vector(self, prev_topic):
        """
        Calculate the probability from t1 to all other topics.
        We calculate the probability for each MC and then average by weight/
        :param t1:
        :return:
        """
        if self.sets is None or self.mcs is None:
            return

        weights = np.array([len(s) / len(self.chains) for s in self.sets])  # should sum up to 1
        vectors = [mc.predict_vector(prev_topic=prev_topic) for mc in self.mcs]
        return np.average(vectors, axis=0, weights=weights)
        # return np.sum(weights[:, np.newaxis] * vectors, axis=0)



if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'

    # make_transition_matrix(base_path=base_path)
    # # p = mc.predict(topic=4, prev_topic=5)
    # # p2 = mc.predict(topic=14, prev_topic=35)
    # # p3 = mc.predict(topic=14, prev_topic=14)
    # p4 = mc.predict(topic=25, prev_topic=20)
    # ps = mc.predict_vector(20)
    # chains = make_chains(base_path, save=True)
    t_l = get_topic_list(base_path)
    print("done")
    # mc = MC(name='').fit(chains, out_name="models/transitions/mc_iner1")

    # chains = [[1,2,3,4,5],[2,3,4,5,1],[4,3,4,3,2], [3,4,5,1,2], [5,4,3,2,1]]
    # logging.info("Using max cluster")
    # lls = []
    # for k in [2, 3, 5, 7, 10, 15, 20]:
    #     logging.info(f"k: {k}")
    #     mcc = MCClusters(k=k).fit(chains)
    #     lls.append(mcc.ll)
    # logging.info(f"likelikhoods for {k}:")
    # logging.info(lls)
    #
    # logging.info("Using weighted probability")
    # # for k in [2, 3, 5, 7, 10, 15, 20]:
    # lls = []
    # for k in [5]:
    #     logging.info(f"k: {k}")
    #     mcc = MCClusters(k=k).fit(chains, weighted=True, iterations=15)
    #     lls.append(mcc.ll)
    #     # mcc.save_sets(name=f'topic_chains_{k}_kmeans.json')
    # logging.info(f"likelkhoods for {k}:")
    # logging.info(lls)
    # joblib.dump(mcc, base_path + 'models/transitions/mcc5_iner5_iter15.pkl')
    # x=3
    # print(mcc)