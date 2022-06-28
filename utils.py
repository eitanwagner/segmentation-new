
def merge_locs(l1, l2):
    """
    Find locations of l2 within l1. Both arrays are assumed to be sorted
    :param l1:
    :param l2:
    :return: list of locations of l2 in l1 (represented by the number of elements before)
    """
    locs = []
    i = j = 0
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            i += 1
        else:
            locs.append(i)
            j += 1

    # add elements at the end
    while j < len(l2):
        locs.append(i)
        j += 1
    return locs