import os
NUM_CLASSES = (len(os.listdir('../data/Training/')))

STR_2_INT = dict([(s,i) for i,s in enumerate(os.listdir('../data/Training/'))])
