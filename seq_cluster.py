
import pomegranate
from pomegranate import MarkovChain
from pomegranate import GeneralMixtureModel
from pomegranate import MultivariateGaussianDistribution
import numpy as np

def cluster(sequences,k):
    mm = GeneralMixtureModel.from_samples(distributions=MultivariateGaussianDistribution, n_components=k, X=sequences)
    # mc = MarkovChain.from_samples(sequences)
    return mm

if __name__ == "__main__":
    sequences = np.array([[1,2,3,4,5], [3,4,5,1,2], [4,5,1,2,3], [3,2,1,5,4], [4,3,2,1,5]])
    mm = cluster(sequences=sequences, k=2)
    print("done")