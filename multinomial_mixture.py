

# from GPyM_TM.GSDMM import DMM

def fit_dmm(chains, k):
    from GPyM_TM import GSDMM
    corpus = [" ".join([_c for _c in c]) for c in chains]
    dmm = GSDMM.DMM(corpus=chains, nTopics=k)
    dmm.topicAssigmentInitialise()
    dmm.inference()
    finalAssignments = dmm.writeTopicAssignments()