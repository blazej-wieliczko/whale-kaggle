import numpy as np

outfile = '/Users/blazej/Projects/whale-kaggle/submission/submission-merge.csv'

infiles = ['/Users/blazej/Projects/whale-kaggle/submission/old/submission7-best.csv',
           '/Users/blazej/Projects/whale-kaggle/submission/submission-99400submission-nn-mel-scale-nodupes.csv.csv', #200 nodes in the random forest
           '/Users/blazej/Projects/whale-kaggle/submission/submission-submission-nn-big-fastlearning-fullset-4layer-0.8train-mel-scale-nodupes.csv.csv', #200 nodes in the random forest
           ]

weights = [0.95257, 0.95527292639, 0.954]

labels_array = None

for i, labels_path in enumerate(infiles):
    new_labels = np.loadtxt(labels_path, dtype='float32', delimiter=",", skiprows=0)
    if labels_array is not None:
        labels_array += new_labels * weights[i]
    else:
        labels_array = new_labels * weights[i]

labels_array /= sum(weights)

output = '\n'.join([str(p) for p in labels_array])

f = open(outfile, 'w')
f.write(output)
f.close()

print sum(labels_array)/50000




