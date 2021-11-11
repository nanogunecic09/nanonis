#to get a list of measurement filenames given:
# path, staticname eg.: 'S211026_', indexes: (1,200) and estension (.dat default) 
def getfnames(path,staticname, idx,extension='.dat'):
    idxn = [f"{i:03}" for i in range(idx[0],idx[1]+1)]
    fnames = []
    for idx in idxn:
        fnames.append(path + staticname + idx + extension)
    return fnames