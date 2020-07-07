import matplotlib.pyplot as plt
import seaborn as sns

def trend_comparisons(df, target, comparitors, s=10, alpha=0.3):
    fig, axes = plt.subplots(len(comparitors), figsize=(20, 20))
    for i, c in enumerate(comparitors):
        axes[i].scatter(df[c], df[target], s=s, alpha=alpha)
        axes[i].set_title(c)
        axes[i].set_xticklabels(axes[i].get_xticklabels(),rotation=80)    

def mean_count_plot(df, col, target, rc={'figure.figsize':(15,10)}):
    sns.set(rc=rc)
    ax = sns.countplot(x=col, data=df)
    ax2 = ax.twinx()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=80)    
    ax2 = sns.pointplot(x=col, y=target, data=df, color='black', legend=False, errwidth=0.5)
    ax.grid(False)

def split_plot(data, x, y, compcol, aalpha=0.1, balpha=0.1, xlim=None, ylim=None):
    alphamap = {False: aalpha, True: balpha}
    colormap = {False: "tab:blue", True: "red"}
    
    for val in [False, True]:          
        plt.scatter(x, y, data=data[data[compcol] == val], alpha=alphamap[val], s=20, c=colormap[val])
    
    plt.xlabel(x)
    plt.ylabel(y)
    
    if xlim is not None:
        plt.xlim(xlim)
        
    if ylim is not None:
        plt.ylim(ylim)