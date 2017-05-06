import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
from library.metrics.classification import confusion_matrix


def plot_confusion_matrix(y_test, y_pred, classes=[], normalize=False, fig_size=(8,6),
                          title='Confusion matrix', cmap=plt.cm.Blues, plot_lib='matplotlib',
                          matplotlib_style='default'):
    cm = confusion_matrix(y_test, y_pred)
    if len(classes) == 0:
        classes = list(range(cm.shape[0]))
    print('Confusion matrix, without normalization')
    print(cm)
    if normalize:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        print(norm_cm)
    if plot_lib == 'matplotlib':
        plt.style.use(matplotlib_style)
        plt.figure(figsize=fig_size)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=60)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    elif plot_lib == 'seaborn':
        cm_df = pd.DataFrame(data=cm, index=classes, columns=classes)
        plt.figure(figsize=fig_size)
        ax = plt.axes()
        sns.heatmap(cm_df, ax=ax, annot=True, fmt='d')
        ax.set_title(title)
        plt.xticks(rotation=60)
        sns.plt.show()
    else:
        print('Unknown library specified')


def plot_variance(scores, means, stds, legend=['Legend1', 'Legend2'], colors=['blue', 'green'],
                  plot_title=['Title1', 'Title2'],
                  plot_xlabel=['X'], plot_ylabel=['Y'], plot_lib='matplotlib',
                  matplotlib_style='default', type='fill', fig_size=(8,6)):
    if plot_lib == 'matplotlib':
        plt.style.use(matplotlib_style)
        plt.rcParams['font.size'] = 12
        font = {'fontname': 'Ubuntu Mono'}
        plt.grid()
        for i in range(means.shape[1]):
            if type == 'errorbar':
                plt.errorbar(np.arange(len(means[:,i])), means[:,i], stds[:,i])
            if type == 'fill':
                plt.fill_between(np.arange(len(means[:,i])), means[:,i] - stds[:,i], means[:,i] + stds[:,i],
                                 alpha=0.1, color='b')
            plt.plot(np.arange(len(means[:,i])), means[:,i], 'o-', color='b', label='Training score')
        plt.title(plot_title, **font)
        plt.xlabel(plot_xlabel, **font)
        plt.ylabel(plot_ylabel, **font)
        plt.legend(loc='best')
        plt.axis('tight')
        plt.show()
    elif plot_lib == 'seaborn':
        fig = plt.figure(figsize=fig_size)
        plt.grid()
        sns_plot = sns.tsplot(data=scores, err_style=['ci_band', 'ci_bars'], marker='o', legend=True)
        sns_plot.set(xlabel=plot_xlabel, ylabel=plot_ylabel)
        sns.plt.title(plot_title)
        sns.plt.show()
    else:
        print('Unknown library specified')


def plot_accuracy(scores, legend=[], colors=[], plot_title='Title',
                  plot_xlabel='X', plot_ylabel='Y', plot_lib='matplotlib', matplotlib_style='default',
                  fig_size=(800,600), filename=''):
    if len(legend) == 0:
        for i in range(scores.shape[0]):
            legend.append('Legend ' + str(i))
    if len(colors) == 0:
        colors = ['blue', 'green', 'red', 'mediumvioletred', 'magenta', 'sienna', 'maroon', 'brown']
    if plot_lib == 'matplotlib':
        plt.style.use(matplotlib_style)
        plt.grid()
        for i in range(scores.shape[0]):
            plt.plot(np.arange(len(scores[i,:])), scores[i,:], '-', color=colors[i], label=legend[i])
        plt.title(plot_title)
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabel)
        plt.legend(loc='best')
        plt.axis('tight')
        if filename != '':
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        plt.clf()
    elif plot_lib == 'seaborn':
        fig = plt.figure(figsize=fig_size)
        plt.grid()
        sns_plot = sns.tsplot(data=scores, err_style=['ci_band', 'ci_bars'], marker='o', legend=True)
        sns_plot.set(xlabel=plot_xlabel, ylabel=plot_ylabel)
        sns.plt.title(plot_title)
        sns.plt.show()
    else:
        print('Unknown library specified')
