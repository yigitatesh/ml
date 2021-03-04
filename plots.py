import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns

def knn_classifier_plot(clf, X_test, y_test, labels=None):
    plt.figure()
    
    # classes
    classes = list(set(y_test))
    
    # color maps
    colors_bold = ['#FF0000', '#0000FF', '#00FF00', '#00FFFF'][:len(classes)]
    colors_light = ['#FFAAAA', '#AAAAFF', '#AAFFAA', '#AAFFFF'][:len(classes)]
    cmap_light = ListedColormap(colors_light)
    
    # numpy arrays for region coloring
    x_min, x_max = X_test[:, 0].min(), X_test[:, 0].max()
    
    mesh_step_size = (x_max - x_min) / 100
    step = mesh_step_size * 10
    x_min, x_max = X_test[:, 0].min() - step, X_test[:, 0].max() + step
    y_min, y_max = X_test[:, 1].min() - step, X_test[:, 1].max() + step
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), 
                         np.arange(y_min, y_max, mesh_step_size))
    
    # predictions
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # plotting
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    if labels is not None:
        for i, cls in enumerate(classes):
            plt.scatter(X_test[y_test == cls, 0], X_test[y_test == cls, 1], c=colors_bold[i], alpha=0.8, label=labels[i])
    else:
        for i, cls in enumerate(classes):
            plt.scatter(X_test[y_test == cls, 0], X_test[y_test == cls, 1], c=colors_bold[i], alpha=0.8, label=str(cls))
            
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("KNN Classifier Regions")
    
    plt.legend()
    plt.show()

def plot_classifier(clf, X, y, X_test=None, y_test=None, title=None, target_names = None):
    numClasses = np.amax(y) + 1
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])

    step = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    xx, yy = np.meshgrid(np.arange(x_min-k, x_max+k, step), 
                         np.arange(y_min-k, y_max+k, step))

    P = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    P = P.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, P, cmap=cmap_light, alpha = 0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, 
                s=plot_symbol_size, edgecolor = 'black')
    plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
    plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)

    if (X_test is not None):
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, 
                    s=plot_symbol_size, marker='^', edgecolor = 'black')
        train_score = clf.score(X, y)
        test_score  = clf.score(X_test, y_test)
        title = title + "\nTrain score = {:.2f}, Test score = {:.2f}".format(train_score, test_score)

    if (target_names is not None):
        legend_handles = []
        for i in range(0, len(target_names)):
            patch = mpatches.Patch(color=color_list_bold[i], label=target_names[i])
            legend_handles.append(patch)
        plt.legend(loc=0, handles=legend_handles)

    if (title is not None):
        plt.title(title)
