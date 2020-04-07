import json
import numpy as np
import os

def els_to_amts(ordered_els, fstructure):
    """
    Args:
        ordered_els (list) - list of els (str) in the order they appear in the structure
        fstructure (path) - path to structure file to parse
    Returns:
        dictionary of {element (str) : number in calculated structure (int)}
    """
    els = ordered_els
    with open(fstructure) as f:
        count = 0
        for line in f:
            if count <= 4:
                count += 1
            if count == 5:
                stuff = [v for v in line.split(' ') if v != '']
                try:
                    amts = [int(v) for v in stuff]
                    els_to_amts = dict(zip(els, amts))
                    break
                except:
                    continue
                    
    return els_to_amts

def idxs_to_els(ordered_els, els_to_amts, nsites):
    """
    Args:
        ordered_els (list) - list of els (str) in the order they apepar in a structure
        els_to_amts (dict) - {el (str) : number of that el in structure (int)}
        nsites (int) - number (int) of ions in calculated structure
    Returns:
        dictionary of index in structure (int) to the element (str) at that index
    """
    els = ordered_els
    els_to_sites = {}
    idx = 0
    for el in els:
        start, stop = idx, els_to_amts[el]+idx
        els_to_sites[el] = (start, stop)
        idx = stop
    els_to_idxs = {el : range(els_to_sites[el][0], els_to_sites[el][1]) for el in els_to_sites}
    idxs_to_els = {}
    for idx in range(nsites):
        for el in els_to_idxs:
            if idx in els_to_idxs[el]:
                idxs_to_els[idx] = el
    return idxs_to_els

def els_to_idxs(idxs_to_els):
    """
    Args:
        idxs_to_els (dict) - {index in structure (int) : el on that site (str)}
    Returns:
        {el (str) : [idxs (int) with that el]}
    """
    els = sorted(list(set(idxs_to_els.values())))
    els_to_idxs = {el : [] for el in els}
    for idx in idxs_to_els:
        els_to_idxs[idxs_to_els[idx]].append(idx)
    return els_to_idxs

def nsites(els_to_amts):
    """
    Args:
        els_to_amts (dict) - {el (str) : number of that el in structure (int)}
    Returns:
        number (int) of ions in calculated structure
    """
    return np.sum(list(els_to_amts.values()))

def replace_line(file_name, line_num, text):
        
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def read_json(fjson):
    """
    Args:
        fjson (str) - file name of json to read
    Returns:
        dictionary stored in fjson
    """
    with open(fjson) as f:
        return json.load(f)

def write_json(d, fjson):
    """
    Args:
        d (dict) - dictionary to write
        fjson (str) - file name of json to write
    Returns:
        written dictionary
    """        
    with open(fjson, 'w') as f:
        json.dump(d, f)
    return d  

def gcd(a,b):
    """
    Args:
        a (float, int) - some number
        b (float, int) - another number
    
    Returns:
        greatest common denominator (int) of a and b
    """
    while b:
        a, b = b, a%b
    return a    

def atomic_valences_data():
    DATA_PATH = '/global/homes/y/yychoi/CODES/myVASPhandler/atomic_valence.json'
    with open(DATA_PATH) as f:
        return json.load(f)

def atomic_electronegativities_data():
    DATA_PATH = '/global/homes/y/yychoi/CODES/myVASPhandler/atomic_electronegativities.json'
    with open(DATA_PATH) as f:
        return json.load(f)
