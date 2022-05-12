import spacy
import json
import numpy as np
import re
import pandas as pd

# import enum
# class Times(enum.Enum):
BEFORE = 0
AFTER_RISE = 1
BEFORE_INVASION = 2  # but after the war broke out
BEFORE_EXTERMINATION = 3  # but after the invasion to this country
DURING_WAR = 4
AFTER_WAR = 5

def make_time_category(terms):
    """
    Makes time category given a list of terms
    :param terms:
    :return:
    """
    def _make_time(str):
        """
        Converts sf label to time category (if relevant)
        :param label:
        :return:
        """
        # we need to add location data.
        # for now, we ignore classes 2 and 3
        years = re.findall(r'\d{4}', str)
        if len(years) == 0:
            return None
        elif len(years) >= 2:
            if int(years[1]) <= 1939:
                return BEFORE
            if int(years[1]) <= 1945:
                return DURING_WAR
            else:
                return AFTER_WAR
        else:
            if int(years[0]) < 1939:
                return BEFORE
            if 1939 < int(years[0]) < 1945:
                return DURING_WAR
            if int(years[0]) > 1945:
                return AFTER_WAR
        return None

    times = [_make_time(t) for t in terms]
    times = set([t for t in times if t is not None])
    # for t in terms:
    #     m_t = _make_time(t)
    #     if m_t is not None:
    #         return m_t
    if len(times) == 1:  # if no time or if two different times, return None
        return times.pop()
    return None

def make_time_data(data_path):
    with open(data_path + 'sf_all.json', 'r') as infile:
        data = json.load(infile)

    time_data = {}
    for t, d in data.items():
        time_data[t] = [[_d['text'], [int(make_time_category(_d['terms']))]] for _d in d if make_time_category(_d['terms']) is not None]

    # all_data = [t_d for t, d in time_data for t_d in d if t_d[1] is not None]
    with open(data_path + 'time_all.json', 'w') as outfile:
        json.dump(time_data, outfile)

def get_year(span):
    for t in span:
        _t = t.text
        while _t[-1] == "-":
            _t = _t[:-1]
            if len(_t) == 0:
                return None
        if t.is_digit:
            if len(t.text) == 4 and t.text[:2] == "19":
                return int(t.text)
        elif _t[-1] == "s" and len(_t) > 1:
            _t = _t[:-1]
        elif _t[-2:] in ["th", "rd", "nd"] and len(_t) > 2:
            _t = _t[:-2]

        if _t.isnumeric() and len(_t[1:]) == 2 and (_t[0] == "3" or _t[0] == "4"):
            return 1900 + int(_t)
    return None

def plot(x, y, t="", t_len=None, save=False, t_num=None):
    import matplotlib.pyplot as plt
    if t != "":
        plt.title(t)
    plt.plot(x, y, "ro")
    top = 800
    if t_len is not None:
        top = t_len + 10
    plt.axis([0, t_len, 1900, 1980])
    if save:
        plt.savefig(f"C:/Users/Eitan/nlplab/time_figs/{t_num}.png")
    plt.show()


def is_visit(terms, terms_df):
    for t in terms:
        row = terms_df[terms_df['Label'] == t].squeeze().to_dict()
        if row['Sub-Type'].find("returns and visits") >= 0:
            return True
    return False

def extract_loc(terms, terms_df):
    # TODO what if there are more than one?
    _terms = []
    loc_cats = ["cities in", "kibbutzim", "German concentration camps in", "German death camps in", "displaced persons camps or",
                "refugee camps", "ghettos in"]
    for t in terms:

        # for bad characters
        end = t.find("?")
        end2 = end + t[end+1:].find("?") + 1
        end3 = end2 + t[end2+1:].rfind("?") + 1
        # if end2 == 0:  # starts with ??
        #     print("bad term!!!!!")
        if end == -1:  # no '?'
            row = terms_df[terms_df['Label'] == t]
        elif end2 == end:  # only one '?'
            row = terms_df[pd.Series([_t[:end] + _t[end+1:] for _t in terms_df['Label']])
                           == t[:end] + t[end+1:]]
        elif end3 == end2:  # two '?'
            row = terms_df[pd.Series([_t[:end] + _t[end+1:] + _t[end2+1:] for _t in terms_df['Label']])
                           == t[:end] + t[end+1:] + t[end2+1:]]
        else:  # more than two '?'
            row = terms_df[pd.Series([_t[:end] + _t[end+1:end2] + _t[end3+1:] + str(_t.count("?")) for _t in terms_df['Label']])
                           == t[:end] + t[end+1:end2] + t[end3+1:] + str(t.count("?"))]
        if row.size == 0:
            row = row.squeeze().to_dict()
        else:
            row = row.iloc[0].squeeze().to_dict()

        if len(row['Label']) > 0:
            for _c in loc_cats:
                if row['Sub-Type'][:len(_c)] == _c:
                    _terms.append(row['Label'])
                    # print("!!!!!" + row['Label'])
        # print(row['Sub-Type'])
        # print(row['Label'])

    # returns only first
    if len(_terms) > 0:
        return _terms[0]
    return None

def make_loc_data(data_path, use_segments=True):
    print("Starting")
    terms_df = pd.read_csv(data_path + "All indexing terms 4 April 2022.xlsx - vhitm_data (14).csv", encoding='utf-8')
    # terms_df = pd.read_csv(data_path + "All indexing terms 4 April 2022.xlsx - vhitm_data (14).csv", header=1)

    with open(data_path + 'sf_all.json', 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    with open(data_path + 'sf_unused5.json', 'r') as infile:
        unused = json.load(infile) + [45064]
        unused = []

    last_loc = ""
    last_seg = ''
    loc_data = {}
    visit = False
    for t, d in data.items():
        print(t)
        if int(t) not in unused:
            loc_data[t] = []
            if not use_segments:
                for s in d:
                    current_loc = extract_loc(s['terms'], terms_df=terms_df)
                    if current_loc is not None and current_loc != last_loc:
                        loc_data[t].append([last_seg, last_loc])
                        last_loc = current_loc
                        last_seg = ""
                    elif visit:
                        current_loc = loc_data[t][-1][-1]  # last location added before the visit
                        loc_data[t].append([last_seg, last_loc])
                        last_loc = current_loc
                        last_seg = ""
                    visit = is_visit(s['terms'], terms_df=terms_df)
                    last_seg = last_seg + s['text']
                loc_data[t].append([last_seg, last_loc])
            else:
                for s in d:
                    current_loc = extract_loc(s['terms'], terms_df=terms_df)
                    if current_loc is None:
                        current_loc = ""
                    loc_data[t].append([s['text'], current_loc, "visit" if is_visit(s['terms'], terms_df=terms_df) else ""])

    if use_segments:
        with open(data_path + 'locs_segments.json', 'w') as outfile:
            json.dump(loc_data, outfile)
    else:
        with open(data_path + 'locs.json', 'w') as outfile:
            json.dump(loc_data, outfile)


if __name__ == "__main__":
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    data_path = base_path + 'data/'
    make_loc_data(data_path, use_segments=True)
    make_loc_data(data_path, use_segments=False)
    # make_time_data(data_path)
    pass
    # with open("C:/Users/Eitan/nlplab/raw_text.json", "r") as infile:
    #     texts = json.load(infile)
    # nlp2 = spacy.load("en_core_web_trf")
    # # t = "103"
    # ts = list(texts.keys())[:]
    # # ts = range(144, 155)
    # events = []
    # all_X, all_Y = [], []
    # all_X_n = np.array([])
    # for _t in ts:
    #     t = str(_t)
    #     doc2_2 = nlp2(texts[t])
    #
    #     sent2i = {s.text: i for i, s in enumerate(doc2_2.sents)}
    #
    #     X, Y = [], []
    #     for e in doc2_2.ents:
    #         if e.label_ == "EVENT":
    #             events.append((e.text, t, e.start))
    #         if e.label_ == "DATE":
    #             y = get_year(e)
    #             if y is not None:
    #                 # print(y, e.start)
    #                 Y.append(y)
    #                 # X.append(e.start)
    #                 X.append(sent2i[e.sent.text])
    #                 # print(e.text)
    #                 # print(e.start)
    #     print(f"\nT: {t}")
    #     # print(len(X))
    #     # print(X)
    #     # print(Y)
    #     t_len = len([s for s in doc2_2.sents])
    #     all_X = all_X + X
    #     all_Y = all_Y + Y
    #     all_X_n = np.concatenate([all_X_n, 100 * np.array(X) / t_len])
    #     # print("Sents: ", t_len)
    #     plot(X, Y, t=f"Testimony: {t}, ents: {len(X)}", t_len=t_len, save=True, t_num=_t)
    #
    # print(events)
    # plot(all_X, all_Y, t=f"All", t_len=3000, save=True, t_num="all")
    # plot(all_X_n, all_Y, t=f"All_n", t_len=100, save=True, t_num="all_n")