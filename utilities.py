import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
from time import perf_counter
import pickle
import copy
import os
import itertools

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import confusion_matrix


def drawSketch(x, y = None, rows = 28, cols = 28, cmap='gray_r', ax = None, scale = None, savefig = False):
    """
    Draw the greyscale image specified by the 1D vector x on the axes ax
    
    :param x: A 1D numpy array of greyscale pixel data for a rectangular image of
              size (rows x cols)
              Row 0 is the topmost row, col 0 is the leftmost column.
    :param y: Optional annotation which will be added in the top left corner
    :param rows: Number of rows in the image
    :param cols: Number of columns in the image.
    :param ax: Axes object to draw to.  If none, one will be created
    :param cmap: Matplotlib-format colormap designation
    :param scale: If not None, multiply all pixels in X by scale
    :param savefig: If not False, save the figure generated here to savefig
    
    :return: Axes object
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # Reorganize to a 2D array
    x = x.reshape((rows, cols))

    if scale is not None:
        x = x * scale

    ax.imshow(x, cmap=cmap)
    if y is not None:
        ax.text(rows * 0.1, cols * 0.1, str(y), color='r')

    if savefig:
        ax.get_figure().savefig(savefig)
    return ax

def drawSketches(X, y = None, subplotShape = None, fig = None, savefig = False, **kwargs):
    """
    Draw an array of subplots of greyscale images defined by the 2D array X, optional annotated y
    
    :param X: a 2D numpy array of 1D vectors of greyscale pixel data for a rectangular image of 
              size (rows x cols)
              Row 0 is the topmost row, col 0 is the leftmost column.
    :param y: (Optional) List-like of annotations for figures (eg: their true values) 
    :param subplotShape: Tuple defining the shape of the subplots to be displayed (rows, cols)
                         If None, display will be approximately square
    :param savefig: If not False, save the figure generated here to savefig
    :param kwargs: All other arguments are passed to drawSketch()
    
    :return: Tuple of matplotlib figure and axes objects
    """
    if subplotShape is None:
        # Get subplotShape by the next round square
        temp = math.ceil(math.sqrt(X.shape[0]))
        subplotShape = (temp, temp)
    
    if fig is None:
        fig, axs = plt.subplots(nrows=subplotShape[0], ncols=subplotShape[1])
    else:
        axs = fig.subplots(nrows=subplotShape[0], ncols=subplotShape[1])

    for i in range(subplotShape[0]):
        for j in range(subplotShape[1]):
            k = i * subplotShape[1] + j
            try:
                thisy = int(y[k])
            except TypeError:
                thisy = None
            drawSketch(X[k, :], thisy, ax=axs[i, j], **kwargs)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    if savefig:
        ax.get_figure().savefig(savefig)
    return fig, axs

def plotCategoryExamples(names, df, dataCols, nImages = 5, randomSeed = 1, savefig = False):
    """
    Return a figure of examples of the specified Quick, Draw! examples from the dataframe given

    :param names: Names of the Quick, Draw! datasets to plot
    :param df: Dataframe of data to plot
    :param dataCols: Names of the pixel data columns in the dataframe
    :param nImages: Number of randomly drawn images to plot
    :param randomSeed: Random number seed
    :param savefig: If not False, save the figure generated here to savefig

    :return: Tuple of Matplotlib Figure and Axes (or list of axes) objects
    """
    nImages = 5
    rows = len(names)
    cols = nImages + 1
    subplotShape = (rows, cols)

    for name in names:
        mask = df.name == name
        avgPixel = df.loc[mask, dataCols].values.mean(axis=0)
        thisdf = df.loc[mask,  dataCols].sample(n=nImages, random_state=randomSeed)
        try:
            toPlot = np.concatenate([toPlot, avgPixel[None, :], thisdf.values])
        except NameError:
            toPlot = np.concatenate([avgPixel[None, :], thisdf.values])
    
    figsize = (subplotShape[1], subplotShape[0])
    fig = plt.figure(figsize=figsize)
    fig, axs = drawSketches(toPlot, subplotShape=subplotShape, fig=fig)

    for i, ax in enumerate(axs):
        ax[0].set_ylabel(names[i])
    axs[-1][0].set_xlabel("Mean Image")
        
    if savefig:
        fig.savefig(savefig)
    return fig, axs

def computeLearningCurve(estimator, XFull, yFull, testSize, scorer = None,
                     learningCurveFolds = 10, stratifyLearningCurveFolds=True, randomSeed = 1,
                     saveas = False, returnEstimators = False):
    """
    Returns a dictionary of results from a learning curve calculation for a given estimator
    
    Data is split into a validation set of size testSize and a training set (remaining data).  The training data 
    is then broken into folds of varying sizes (depending on input learningCurveFolds) training/validation scores
    are computed.  

    :param estimator: The estimator used to classify the data
    :param XFull: Training data set inputs (to be segmented for plotting learning curve)
    :param yFull: Training data set outputs 
    :param testSize: Number of data points (or fraction if <= 1) held out of X,y for test curve calculation
    :param scorer: If not None, specifies the scorer to be used in scoring the estimator.  Otherwise, use estimator.score()
    :param learningCurveFolds: Integer number of folds to segment the learning curve into, or a list of integer numbers
                               of points to use for each fold
    :param stratifyLearningCurveFolds: Boolean to specify whether learning curve folds should use stratified samples
    :param randomSeed: Seed used for all random functions to keep things random-ish
    :param saveas: If not False, output dict is also pickled in file named saveas (will overwrite if exists)
    :param returnEstimators: If True, return all estimators fitted during evaluation (may take up a lot of memory 
                             (/space if saved to disk))

    :return: Dict of:
        n_data: Number of data points for each step
        fit_time: Fit time for the estimator in each outer fold
        score_time: Score time for the estimator in each outer fold
        train_score: Train score for the estimator in each outer fold
        test_score: Test (validation) score for the estimator in each outer fold
        test_score: Score computed on a separate test data set 
                         (not included in inner cross validation) for the estimator in each outer fold
        best_params: Best params for each estimator trained
        cv_results: The cv_results_ attribute from each estimator trained
        estimator: If returnEstimators==True, the estimators trained
    """
    timeStart = perf_counter()
    np.random.seed(randomSeed) # Seed to make random choice repeatable

    # Split training data into train/validation split
    X, XTest, y, yTest = train_test_split(XFull, yFull, test_size=testSize, 
                                      random_state=randomSeed, stratify=yFull)


    # Initialize output:
    storedMetrics = {
        'fit_time': [],
        'score_time': [],
        'train_score': [],
        'test_score': [],
        'n_data': [],
        'convergence_iter': [],
        'best_params': [],
        'cv_results': [],
        'estimator': [],
        }

    # Split training data
    if stratifyLearningCurveFolds:
        skf = StratifiedKFold
    else:
        raise NotImplementedError("... Should be easy with KFold from sklearn though")

    if isinstance(learningCurveFolds, int):
        foldIndices = [[] for x in range(learningCurveFolds)]
        # Hacky way to split data into n equal groups
        skf = skf(n_splits=learningCurveFolds, random_state=randomSeed)
        # StratifiedKFold splits into 10 train/test groupings, where the test points in each are unique.  
        # Just use the test indices as our group indices
        # foldIndicesTemp = [iTest for iTrain, iTest in skf.split(X, y)]

        for i, data in enumerate(skf.split(X, y)):
            iTest = data[1]
            for j in range(i, len(foldIndices)):
                foldIndices[j].extend(iTest)
    else:
        foldIndices = [None] * len(learningCurveFolds)
        nMax = y.shape[0]
        indices = np.arange(nMax)
        for i, n in enumerate(learningCurveFolds):
            # Grab n random points as a training dataset
            # Use train_test_split, but get a test set of 0 elements and only keep the indices
            iTrain, _ = train_test_split(indices, random_state=randomSeed, stratify=y, train_size=n, test_size=None)
            foldIndices[i] = iTrain

    for i in range(len(foldIndices)):
        timeIterStart = perf_counter()
        # Update indices
        # Add this fold's training indices to previous indices
        iTrain = foldIndices[i]
        nData = len(iTrain)
        print(f'Indices included at fold {i}: {nData}')

        # Train model
        timeModelFitStart = perf_counter()
        estimator.fit(X[iTrain], y[iTrain])
        timeModelFitEnd = perf_counter()
        fit_time = timeModelFitEnd - timeModelFitStart
        print(f'\tModel trained in {fit_time:.2f}s')
        storedMetrics['fit_time'].append(fit_time)

        # Score model on Train and Test data
        timeScoreStart = perf_counter()
        if scorer is None:
            storedMetrics['train_score'].append(estimator.score(X[iTrain], y[iTrain]))
        else:
            yPred = estimator.predict(X[iTrain])
            storedMetrics['train_score'].append(scorer(y[iTrain], yPred))
        timeScoreEnd = perf_counter()
        print(f'\tModel Scored on Training Data in {timeScoreEnd - timeScoreStart:.2f}s')

        timeScoreStart = perf_counter()
        if scorer is None:
            storedMetrics['test_score'].append(estimator.score(XTest, yTest))
        else:
            yPred = estimator.predict(XTest)
            storedMetrics['test_score'].append(scorer(yTest, yPred))
        timeScoreEnd = perf_counter()
        score_time = timeScoreEnd - timeScoreStart
        print(f'\tModel Scored on Test Data in {score_time:.2f}s')
        storedMetrics['score_time'].append(score_time)

        storedMetrics['n_data'].append(nData)

        # Store best_params and cv_results_ if this was a GridSearchCV object
        try:
            storedMetrics['best_params'].append(estimator.best_params_)
            storedMetrics['cv_results_'].append(estimator.cv_results_)
        except AttributeError:
            pass

        # Store iteration number for items that have iterations (eg: neural network via MLP)
        try:
            storedMetrics['convergence_iter'].append(estimator.n_iter_)
        except AttributeError:
            pass

        if returnEstimators:
            storedMetrics['estimator'].append(copy.deepcopy(estimator))

        print(f'\tTotal time for this iteration = {perf_counter() - timeIterStart:.2f}s')

    if saveas:
        with open(saveas, 'wb') as fout:
            pickle.dump(storedMetrics, file=fout)
    return storedMetrics

def plotTimeCurve(lc, bottom = None, top = None, ax = None, savefig = False, legend = True, ls = ':'):
    """
    Plot a learning curve of training and testing time using the output from computeLearningCurve

    :param lc: Learning curve data
    :param bottom, top: (Optional) y-axis limits as specified by axes.set_ylim
    :param ax: (Optional) Axes object to plot curve to, 
    :param savefig: If not False, save the figure generated here to savefig
    :param legend: If true, include a legend on the figure
    :param ls: (Optional) linestyle for all plots, passed to matplotlib plotting routines

    :return: Matplotlib axes object
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    x = lc["n_data"]
    ax = plotLineWithError(x, lc["fit_time"], getattr(lc, "fit_time_std", None), ax=ax, label_mean='Training', marker='.', ls=ls)
    ax = plotLineWithError(x, lc["score_time"], getattr(lc, "score_time_std", None), ax=ax, label_mean='Scoring', marker='.', ls=ls)

    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlabel("Number of Data Points in Training Set")
    ax.set_ylabel("Time (s)")
    if legend:
        ax.legend()
    if savefig:
        ax.get_figure().savefig(savefig)
    return ax

def plotScoreCurve(lc, bottom = None, top = None, ax = None, savefig = False, legend = True, ylabel = "Score (Accuracy)", ls = ':'):
    """
    Plot a learning curve of training and testing error using the output from computeLearningCurve

    :param lc: Learning curve data
    :param bottom, top: (Optional) y-axis limits as specified by axes.set_ylim
    :param ax: (Optional) Axes object to plot curve to, 
    :param savefig: If not False, save the figure generated here to savefig
    :param legend: If true, include a legend on the figure
    :param ylabel: Label for the y-axes of the plot
    :param ls: (Optional) linestyle for all plots, passed to matplotlib plotting routines

    :return: Matplotlib axes object
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    
    x = lc["n_data"]

    ax = plotLineWithError(x, lc["train_score"], getattr(lc, "train_score_std", None), ax=ax, label_mean='Training Data', marker='.', ls=ls)
    try:
        ax = plotLineWithError(x, lc["cv_test_score"], getattr(lc, "cv_test_score_std", None), ax=ax, label_mean='Inner Cross Validation Data', marker='.', ls=ls)
    except KeyError:
        pass
    ax.plot(x, lc["test_score"], marker='.', label='Held Out Validation Data', ls=ls)
    
    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlabel("Number of Data Points in Training Set")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    if savefig:
        ax.get_figure().savefig(savefig)
    return ax

def plotIterationCurve(lc, bottom = None, top = None, ax = None, savefig = False, ls = ':'):
    """
    Plot a learning curve of the number of iterations required to train on the held out test set


    :param lc: Learning curve data
    :param bottom, top: (Optional) y-axis limits as specified by axes.set_ylim
    :param ax: (Optional) Axes object to plot curve to, 
    :param savefig: If not False, save the figure generated here to savefig
    :param ls: (Optional) linestyle for all plots, passed to matplotlib plotting routines

    :return: Matplotlib axes object
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    x = lc["n_data"]
    ax.plot(x, lc['convergence_iter'], marker='.', ls=ls)

    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlabel("Number of Data Points in Training Set")
    ax.set_ylabel("Iterations to Convergence in Training")
    if savefig:
        ax.get_figure().savefig(savefig)
    return ax

def plotLineWithError(x, y_mean, y_std, ax = None, label_mean = None, label_std = None, marker = '.', alpha = 0.5, savefig = False, ls='-'):
    """
    Helper plotting function to plot a line with optional standard deviation error bands
    """

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y_mean, marker=marker, label=label_mean, ls=ls)
    if y_std is not None:
        y_mean = np.asarray(y_mean)
        y_std = np.asarray(y_std)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, label=label_std, alpha=alpha)
    if savefig:
        ax.get_figure().savefig(savefig)
    return ax

def plotConfusionMatrix(yTrue, yPred, classes, cmap = 'gray_r', ax = None, rotation = 0, savefig = False):
    """
    Plot a confusion matrix.

    Derived from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    :param yTrue: 1D vector of true values of y (integers of 0 or 1)
    :param yPred: 1D vector of predicted values of y (integers of 0 or 1)
    :param classes: Class names (in same order as integer values of y)
    :param cmap: Passed to imshow
    :param ax: (OPTIONAL) axes object to draw to
    :param rotation: Rotation of the xticks, passed to set_xticklabels
    :param savefig: If not False, save the figure generated here to savefig

    :return: Matplotlib axes object
    :return: 
    """
    if ax is None:
        fig, ax = plt.subplots()

    cm = confusion_matrix(yTrue, yPred)
    ax.imshow(cm, interpolation='nearest', cmap=cmap)

    fmt = 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", verticalalignment='center',
                 color="red", fontsize='20')

    tick_marks = np.arange(len(classes))
    ax.set_xticklabels(classes, rotation=rotation)
    ax.set_yticklabels(classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    # ax.tight_layout()

    if savefig:
        ax.get_figure().savefig(savefig, bbox='tight')
    return ax

def structure2str(hidden_layer_sizes, string="", sep='-'):
    """
    Convert a tuple of hidden layer sizes into a string of sizes separated by sep
    """
    if len(hidden_layer_sizes) == 0:
        raise ValueError("Invalid hidden_layer_sizes, must be of at least length of 1")
    else:
        if len(string) == 0:
            string = str(hidden_layer_sizes[0])
        else:
            string = string + sep + str(hidden_layer_sizes[0])
        if len(hidden_layer_sizes) > 1:
            return structure2str(hidden_layer_sizes[1:], string=string, sep=sep)
        else:
            return string

def getDrawData(names = None, nSamples = 1000, randomSeed = 1, datapath = "./data/",
                datasetPrefix = "full_numpy_bitmap_", datasetExtension = ".npy" , saveas = False):
    """
    Returns a pandas DataFrame with the requested Quick, Draw! data.

    Note: Only has access to a subset specified in the dictionary below (didn't
    download everything...)

    :param names: Names of drawings to be pulled from data
    :param nSamples: Number of samples to be loaded per named drawing
    :param randomSeed: If not False, Random seed passed to np.random.seed
    :param datapath: Path to Quick Draw drawing data (in .npy format)
    :param datasetPrefix, datasetExtension: Prefix and extension on filenames.  Filenames to be constructed as:
                                            datasetPrefix + name + datasetExtension
    :param saveas: If not False, pickle the dataframe to a file names saveas (will be compressed if file has .zip
                   extension)

    :return: Dict of: 
        df: Pandas dataframe with all features plus columns of name (class name)
            and classNumber (unique integer )
        nameDict: dictionary relating class names to classNumber
    """
    datafiles = [datasetPrefix + name + datasetExtension for name in names]
    nameDict = {}
    
    if randomSeed:
        np.random.seed(randomSeed)

    data = pd.DataFrame()
    classNumber = 0
    for name, fname in zip(names, datafiles):
        nameDict[name] = classNumber
        fullpath = os.path.join(datapath, fname)
        print(f"Loading {nSamples} samples for {name} from {fullpath}")
        temp = np.load(fullpath)
        temp = temp[np.random.choice(temp.shape[0], nSamples, replace=False)]
        df = pd.DataFrame(temp)
        df['name'] = name
        df['classNumber'] = classNumber
        data = pd.concat([data, df], ignore_index=True)
        classNumber += 1

    if saveas:
        data.to_pickle(saveas)

    return {'df': data, 'nameDict': nameDict}

def getFraudData(datafile = "./data/creditcard.csv", nSamples = None, randomSeed = 1, saveas = False):
    """
    Get a subset of the credit card fraud dataset and return as a pandas dataframe
    
    Data from: https://www.kaggle.com/mlg-ulb/creditcardfraud/home
    
    :param datafile: Path to the file containing the data
    :param nSamples: Tuple of (number of positive (fraud) samples, number of negative (no fraud) samples)
                     If None, read all (492) positive and 5000 negative examples 
    :param randomSeed: Seed for random operations
    :param saveas: If true, save dataframe to file of this name
    
    :return: Pandas dataframe of data
    """
    if nSamples is None:
        nSamples = (492, 5000)

    print(f"Importing data with {nSamples[0]} fraudulent and {nSamples[1]} not fraudulent cases")
    df = pd.read_csv(datafile)
    fraud = df.loc[df.Class == 1].sample(n=nSamples[0], random_state=randomSeed)
    notFraud = df.loc[df.Class == 0].sample(n=nSamples[1], random_state=randomSeed)
    dfOut = pd.concat([notFraud, fraud], ignore_index=True)
    return dfOut

def drawIncorrectSketches(X, yTrue, yPred, classNumbers, names, n = 5, randomSeed = 1, savefig = False):
    """
    Returns a sketch of n members of each class that were incorrectly classified (false negatives)

    :param X: 2D array of examples of drawings defined by columns of pixel data
    :param yTrue: True y values corresponding to data in X
    :param yPred: Predicted y values
    :param classNumbers: Integer numbers for each class
    :param names: Names of each class
    :param n: Number of incorrect sketches to return for each class
    :param randomSeed: Random number seed
    :param savefig: If not False, save the figure to a file with this name

    :return: Matplotlib figure
    """
    if randomSeed:
        np.random.seed(randomSeed)

    for classNumber in classNumbers:
        #       Members of this class & Incorrectly classified (false negative)
        mask = (yTrue == classNumber) & (yTrue != yPred)
        Xsub = X[mask]
        thisX = Xsub[np.random.choice(Xsub.shape[0], n, replace=False)]
        try:
            toPlot = np.concatenate((toPlot, thisX))
        except NameError:
            toPlot = thisX

    fig, ax = drawSketches(toPlot, subplotShape=(len(classNumbers), n))
    fig.suptitle("Random Misclassified Objects (False Negatives)")
    for i, a in enumerate(ax):
        a[0].set_ylabel(names[i] + " (Truth)")

    if savefig:
        fig.savefig(savefig)

    return fig

def heatmap(data, xticklabels, yticklabels, xlabel = "", ylabel = "", textcolor = "red", fontsize = None, cmap = 'Greys', savefig = False, ax = None):
    """
    Return a basic labelled heatmap of the numberic data, with data[0,0] placed in the upper left
    
    :param data: Numpy array of data to plot
    :param xticklabels (yticklabels): Labels corresponding to row (column) element of data
    :param textcolor: color designation passed to ax.text()
    :param fontsize: fontsize parameter passed to ax.text()
    :param cmap: cmap designation passed to imshow()
    :param savefig: If not False, save the figure generated here to savefig
    
    :return: tuple of (matplotlib figure, matplotlib axes)
    """
    # Labels need to be padded by 1 element, as imshow will plot data starting centered at tick (1,1) but the
    # set_xticklabels/set_yticklabels sets labels starting at tick 0
    xticklabels = ("", ) + tuple(xticklabels)
    yticklabels = ("", ) + tuple(yticklabels)
    fig, ax = getax(ax)
    fmt = '.2f'
    for i, j in itertools.product(range(data.shape[0]), range(data.shape[1])):
        ax.text(j, i, format(data[i, j], fmt),
                 horizontalalignment="center", verticalalignment='center',
                 color=textcolor, fontsize=fontsize)
    ax.imshow(data, cmap=cmap)
    # Force a tick for every row and column
    ax.xaxis.set_major_locator(MaxNLocator(len(xticklabels), integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(len(yticklabels), integer=True))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if savefig:
        fig.savefig(savefig)

    return fig, ax

def df_to_heatmap(df, x_variable, y_variable, z_variable, **kwargs):
    """
    Slices a dataframe to get a heatmap using the unique x and y variable combos, returning heatmap of z given x,y

    Assumes you have a grid (all x,y combinations)
    """
    x_labels = df.loc[:, x_variable].unique()
    y_labels = df.loc[:, y_variable].unique()

    toPlot = np.zeros((len(y_labels), len(x_labels)))
    for y_i, y_val in enumerate(y_labels):
        for x_i, x_val in enumerate(x_labels):
            mask = (df.loc[:, x_variable] == x_val)  & (df.loc[:, y_variable] == y_val)
            toPlot[y_i, x_i] = df.loc[mask, z_variable]
    heatmap(toPlot, x_labels, y_labels, **kwargs)

def reconstruction_error(X, transformer):
    X_transformed = transformer.transform(X)
    X_reconstructed = transformer.inverse_transform(X_transformed)
    mse = ((X - X_reconstructed)**2).mean()
    return mse

def get_pca_components_for_var(pca, var):
    """
    Return the minimum number of PCA components that exceeds the requested variance explained, and its explained variance
    
    Note that the index of the component that obtains this variance is return - 1

    :return: (n_components, variance_explained_by_n_components)
    """
    #    cumulative variance > target variance ==> array of False False ... False True ... True True 
    #    (FF...FT...TT).argmax() gives index of first True
    evrcs = pca.explained_variance_ratio_.cumsum()
    i = (evrcs > var).argmax()
    actual_var = evrcs[i]
    return i + 1, actual_var

def plot_pca_components(x, coefficients=None, pca=None, scaler=None, cmap='gray_r',
                        imshape=(28, 28), fontsize=12, plot_step=1, last_component=None,
                        show_mean=True):
    """
    Modified from: https://jakevdp.github.io/PythonDataScienceHandbook/06.00-figure-code.html#Digits-PCA-Components
    """
    # mean = pca.mean_
    mean = scaler.mean_

    components = pca.components_
    if last_component is None:
        last_component = components.shape[0] - 1

    if isinstance(plot_step, int):
        to_plot = [x for x in range(last_component) if x % plot_step == 0]
    else:
        to_plot = plot_step
        
    if coefficients is None:
        coefficients = x
        
    if components is None:
        components = np.eye(len(coefficients), len(x))
        
    mean = np.zeros_like(x) + mean

    n_plots = len(to_plot)
    fig = plt.figure(figsize=(1.2 * (5 + n_plots), 1.2 * 2))
    g = plt.GridSpec(2, 4 + bool(show_mean) + n_plots, hspace=0.3)

    def show(i, j, x, title=None, xlabel=None, ylabel=None):
        ax = fig.add_subplot(g[i, j], xticks=[], yticks=[])
        ax.imshow(x.reshape(imshape), interpolation='nearest', cmap=cmap)
        if title:
            ax.set_title(**title)
        if xlabel:
            ax.set_xlabel(**xlabel)
        if ylabel:
            ax.set_ylabel(**ylabel)

    if scaler:
        x = scaler.inverse_transform(x)

    show(slice(2), slice(2), x, title={'label': "True", 'fontsize': fontsize})
    
    approx = mean.copy()
    
    counter = 2
    if show_mean:
        show(0, 2, np.zeros_like(x), title={'label': r'$\mu$', 'fontsize': fontsize})
        show(1, 2, approx, title={'label': r'$Var_{Explained}=$', 'fontsize': fontsize})
        counter += 1

        
    evrc = np.cumsum(pca.explained_variance_ratio_)
    i_plot = 0
    first_plot = True
    for i in range(last_component):
        temp = coefficients[i] * components[i]
        approx = approx + temp
        if i in to_plot:
            if first_plot:
                ylabel_top = "Component"
                ylabel_bot = "Cumulative"
                xlabel_bot = "Mean Image"
                first_plot = False
            else:
                ylabel_top = None
                ylabel_bot = None
                xlabel_bot = None

            this_components = components[i].copy()
            if scaler:
                this_components = scaler.inverse_transform(this_components)
                this_approx = scaler.inverse_transform(approx)
            else:
                this_approx = approx

            show(0, i_plot + counter, components[i], 
                title={'label': f'$c_{{{i}}}$', 'fontsize': fontsize}, 
                ylabel={'ylabel': ylabel_top, 'fontsize': fontsize - 1})
            show(1, i_plot + counter, this_approx,
                 title={'label': f'{evrc[i]:.2f}', 'fontsize': fontsize},
                 ylabel={'ylabel': ylabel_bot, 'fontsize': fontsize - 1},
                 xlabel={'xlabel': xlabel_bot, 'fontsize': fontsize - 1},
                 )
#                  f'${coefficients[i]:.2f} \cdot c_{{{i}}}$')
#             if show_mean or i > 0:
#                 plt.gca().text(0, 1.05, '$+$', ha='right', va='bottom',
#                                transform=plt.gca().transAxes, fontsize=fontsize)
            i_plot += 1
    if scaler:
        approx = scaler.inverse_transform(approx)
    show(slice(2), slice(-2, None), approx, title={'label': "Approx", 'fontsize': fontsize})
    return fig

def getax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax

def get_reconstruction_error(X, transformer):
    """
    Returns the reconstruction error computed for data X using a transformer

    Routine uses transformer to transform and then inverse transform X, then computes mse
    """
    X_transformed = transformer.transform(X)
    X_reconstructed = transformer.inverse_transform(X_transformed)
    mse = ((X - X_reconstructed)**2).mean()
    return mse

def get_max_along_df_axis(df, variable_to_max, variable_to_max_along):
    """
    Returns the slice of a dataframe that has the maximum value of a variable for each instance of a second variable

    eg: [max(y) for all y where x=x1, max(y) for all y where x=x2, ...]

    :return: Numpy array with columns of x_val, max(y|x_val)
    """
    x_labels = df.loc[:, variable_to_max_along].unique()
    scores = np.zeros((len(x_labels), 2))
    for x_i, x_val in enumerate(sorted(x_labels)):
        mask = df.loc[:, variable_to_max_along] == x_val
        scores[x_i] = (x_val, df.loc[mask, variable_to_max].max())
    return scores
