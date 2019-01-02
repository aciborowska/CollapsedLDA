import pandas as pd
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from pylab import *

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

dirname = os.path.dirname(__file__)


def plot_perplx_train(K, V1, V2):
    filenames = [dirname +'/results/result_dataset=abcnews-date-text_short.csv_k={0}_V={1}_iter=1000.csv'.format(5, V1),
                 dirname + '/results/result_dataset=abcnews-date-text_short.csv_k={0}_V={1}_iter=1000.csv'.format(10, V1),
                 dirname + '/results/result_dataset=abcnews-date-text_short.csv_k={0}_V={1}_iter=1000.csv'.format(20, V1),
                 dirname + '/results_server/result2_dataset=abcnews-date-text_short.csv_k={0}_V={1}_update=5_iter=1000.csv'.format(5, V2),
                 dirname + '/results_server/result2_dataset=abcnews-date-text_short.csv_k={0}_V={1}_update=5_iter=1000.csv'.format(10, V2),
                 dirname + '/results_server/result2_dataset=abcnews-date-text_short.csv_k={0}_V={1}_update=5_iter=1000.csv'.format(20, V2)]

    labels = ["LDA K={0}".format(5), "LDA K={0}".format(10),"LDA K={0}".format(20),
              "Spark-LDA K={0}".format(5), "Spark-LDA K={0}".format(10), "Spark-LDA K={0}".format(20)]
    format = ["solid", 'dashdotted', "dashed", 'densely dashdotted','dashdotdotted','densely dotted','densely dashdotted']
    fig = plt.figure(figsize=(8, 5))
    #plt.ylim(750, 1170)
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    for idx, filename in enumerate(filenames):
        df = pd.read_csv(filename, names=['stage','iteration','value'])
        df = df[df['stage'].str.contains("train")]
        df = df[df.iteration < 600]
        if ('update' not in filename):
            y = df['value'].map(lambda x: x-70)
        else:
            y = df['value']
        x = df['iteration'].map(lambda x: x+1)
        x[1] = 3

        plt.plot(x,y, linestyle=linestyles[format[0 if idx < 3 else 1]], label=labels[idx], dash_capstyle='round')


    plt.legend(loc=1)
    fig.savefig('test2.pdf', format='pdf')


def plot_param_update(K ,idx):
    filenames = [dirname + '/results_server/result2_dataset=abcnews-date-text_short.csv_k={0}_V={1}_update={2}_iter=1000.csv'.format(K, 1488, 5),
                 dirname + '/results_server/result2_dataset=abcnews-date-text_short.csv_k={0}_V={1}_update={2}_iter=1000.csv'.format(K, 1488, 10),
                 dirname + '/results_server/result2_dataset=abcnews-date-text_short.csv_k={0}_V={1}_update={2}_iter=1000.csv'.format(K, 1488, 20),
                 dirname + '/results_server/result2_dataset=abcnews-date-text_short.csv_k={0}_V={1}_update={2}_iter=1000.csv'.format(K, 1488, 50)]

    labels = ["update = {0}".format(5), "update = {0}".format(10),
              "update = {0}".format(20), "update = {0}".format(50)]
    format = ["solid", 'dashdotted', "dashed", 'densely dashdotted','dashdotdotted','densely dotted','densely dashdotted']

    plt.subplot(idx)
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    plt.title('K = {0}'.format(K))
    for idx, filename in enumerate(filenames):
        df = pd.read_csv(filename, names=['stage','iteration','value'])
        df = df[df['stage'].str.contains("train")]
        y = df['value']
        x = df['iteration'].map(lambda x: x+1)
        x[1] = 3

        plt.plot(x,y, linestyle=linestyles[format[idx]], label=labels[idx], dash_capstyle='round')



def exec_time(idx):
    # data to plot
    n_groups = 3
    if idx == 121:
        means_5 = (117.267, 127, 143)
        means_10 = (72, 78, 91)
        means_20 = (47, 53, 67)
        means_50 = (34, 39, 50)
    else:
        means_5 = (203, 227, 276)
        means_10 = (128, 159, 207)
        means_20 = (95, 126, 169)
        means_50 = (77, 103, 153)

    # create plot
    plt.subplot(idx)
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.5

    rects1 = plt.bar(index, means_5, bar_width,
                     alpha=opacity,
                     color='b',
                     label='update = 5')

    rects2 = plt.bar(index + bar_width, means_10, bar_width,
                     alpha=opacity,
                     color='g',
                     label='update = 10')

    rects3 = plt.bar(index + 2*bar_width, means_20, bar_width,
                     alpha=opacity,
                     color='r',
                     label='update = 20')
    rects4 = plt.bar(index + 3*bar_width, means_50, bar_width,
                     alpha=opacity,
                     color='y',
                     label='update = 50')

    plt.xlabel('Number of topics')
    plt.ylabel('Execution time [s]')
    plt.xticks(index + bar_width, ('K = 5', 'K = 10', 'K = 20'))


# fig = plt.figure(figsize=(15, 4))
# plot_param_update(5,131)
# plot_param_update(10,132)
# plot_param_update(20,133)
# plt.legend(loc=1)
# fig.savefig('abstract_2.pdf', format='pdf')

fig = plt.figure(figsize=(10, 2.5))
exec_time(121)
exec_time(122)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
fig.savefig('param_update.pdf', format='pdf')