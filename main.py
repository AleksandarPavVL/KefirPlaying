import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def main():
    dataPath = 'C:/All_Projects/Kefir data/data/'
    df = pd.read_csv(dataPath + 'Kefir_Titanium_data.csv')
    print(df.head(), '\n')
    df['date and time'] = pd.to_datetime(df['date and time'])
    print(df.head(), '\n')

    # signal = df['pH']
    # l = len(signal)  # duzina signala
    # w = round(l / 20)  # sirina prozora = 5% od duzine signala
    # if w % 2 != 0:
    #     w += 1
    # print (l, ' ', w)

    # # print(len(df.columns)) # len = 5
    names = list(df.columns) # date and time, Gray (mV) MPS1, Red (mV) MPS2, pH, m(seed)

    print('\nNumber of missing data:\n', df.isnull().sum(), '\n') # nema nedostajucih podataka

    # PRINT AND DRAW ELEMENTARY STATISTICS
    means, medians, stds = statistics(names, df) # prikazuje min, max, mean, median, std i indekse s vrednostima signala
    # koje odgovaraju minimumu i maksimumu tih signala
    plotting(names, df, means, medians, stds) # plotuje signale svih feature-a (osim date and time) i njihove histograme

    # FIND AND REMOVE OUTLIERS
    threshCoeff = 2.5 # coefficient to multiply standard deviation with for calculating the outlier threshold
    startIdxs, stopIdxs = findOutliers(names, df, means, stds, threshCoeff)
    print('Outlier start indices: ', startIdxs, '\n')
    print('Outlier stop indices: ', stopIdxs, '\n')
    df = removeOutliers(names, df, startIdxs, stopIdxs)

    # CORRELATION
    print(df.corr())
    plt.figure()
    # full screen prikaz
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    sns.heatmap(df.corr(), annot = True, cmap = 'YlGnBu')
    plt.title('Correlation between features')

    # PAIRS
    sns.pairplot(df)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    # Z-NORMALIZATION
    scaler = StandardScaler()
    dfScaled = pd.DataFrame(scaler.fit_transform(df.drop('date and time', axis = 1)), columns = df.columns.drop('date and time'))
    dfScaled['date and time'] = df['date and time']
    print('\nZ-normalized data:\n', dfScaled.head())

    # TRAINING - LINEAR REGRESSION
    # X = dfScaled.drop(['pH', 'date and time'], axis = 1)
    X = dfScaled.drop(['pH', 'date and time', 'm(seed)'], axis = 1)
    y = dfScaled['pH']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    print('\nCoefficients:\n', lm.coef_)
    predictions = lm.predict(X_test)
    plt.figure()
    plt.scatter(y_test, predictions)
    plt.xlabel('Y test')
    plt.ylabel('Predicted Y')
    meansS, mediansS, stdsS = statistics(names, dfScaled)
    plotting(names, dfScaled, meansS, mediansS, stdsS, ' scaled')

    plt.show()


def statistics(names, dataframe):
    means = []
    medians = []
    stds = []
    for i in range(len(names)):
        name = names[i]
        if ('date' or 'time') not in name.lower():
            signal = dataframe[names[i]]
            print('\n', name, ' min = ', np.min(signal), ', ', name, ' max = ', np.max(signal))
            print('Row min: ', signal[signal == np.min(signal)]) # indeks i vrednost signala pri minimalnoj vrednosti
            print('Row max: ', signal[signal == np.max(signal)]) # indeks i vrednost signala pri maksimalnoj vrednosti
            print(name, ' mean = ', np.mean(signal))
            print(name, ' median = ', np.median(signal))
            print(name, ' std = ', np.std(signal), '\n')
            means.append(np.mean(signal))
            medians.append(np.median(signal))
            stds.append(np.std(signal))
        else: # popunjavanje indikatorskom vrednoscu, da bi duzina bila ista kao duzina od names
            means.append('NaN')
            medians.append('NaN')
            stds.append('NaN')

    return means, medians, stds


def plotting(names, dataframe, means, medians, stds, type = ''):
    for i in range(len(names)):
        name = names[i]
        if ('date' or 'time') not in name.lower():
            signal = dataframe[names[i]]
            plt.figure()
            plt.plot(range(len(signal)), signal)
            plt.title(name + type + ' signal')
            plt.xlabel('Timestamp')
            # full screen prikaz
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            if means[i] != 'NaN':
                plt.plot(range(len(signal)), np.ones(len(signal)) * means[i], label = 'mean = ' + str(round(means[i], 2)))
                plt.plot(range(len(signal)), np.ones(len(signal)) * medians[i], label = 'median = ' + str(round(medians[i], 2)))
                plt.plot(range(len(signal)), np.ones(len(signal)) * stds[i], label = 'std = ' + str(round(stds[i], 2)))
            plt.legend()

            # histogram
            plt.figure()
            n, bins, patches = plt.hist(x = signal, bins = 'auto', color = '#0504aa', alpha = 0.7, rwidth = 0.85)
            plt.grid(axis = 'y', alpha = 0.75)
            plt.title('Histogram of ' + name + type + ' values')
            plt.xlabel(name + ' values')
            plt.ylabel('Frequency')
            maxFreq = n.max()
            plt.ylim(ymax = np.ceil(maxFreq / 10) * 10 if maxFreq % 10 else maxFreq + 10)
            # full screen prikaz
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')


def findOutliers(names, dataframe, means, stds, threshCoeff): # threshCoeff = 2.5
    startIdxs = {}
    stopIdxs = {}
    for i in range(len(names)):
        name = names[i]
        startIdxs[name] = []
        stopIdxs[name] = []
        if ('date' or 'time') not in name.lower(): # prihvatljiv uslov bi bio i da means[i] != 'NaN'
            signal = dataframe[name]
            upperLimit = means[i] + threshCoeff * stds[i]
            lowerLimit = means[i] - threshCoeff * stds[i]
            for t in range(len(signal)):
                if (t != 0) and (t != len(signal)-1) and (isBetween(signal[t-1], upperLimit, lowerLimit) and (not isBetween(signal[t], upperLimit, lowerLimit))):
                    startIdxs[name].append(t)
                elif (t != 0) and (t != len(signal)-1) and (not isBetween(signal[t-1], upperLimit, lowerLimit) and (isBetween(signal[t], upperLimit, lowerLimit))):
                    stopIdxs[name].append(t)
        if len(startIdxs[name]) > len(stopIdxs[name]): # ako ima viste start indeksa nego stop indeksa, to znaci da se u
            # nekom trenutku desio outlier koji se do kraja signala nije zavrsio
            startIdxs[name].pop()
        elif len(stopIdxs[name]) > len(startIdxs[name]): # ako ima vise stop indeksa nego start indeksa, to znaci da se
            # na pocetku desio outlier, tj. da je signal zapoceo outlier-om
            stopIdxs[name].pop(0)

    return startIdxs, stopIdxs


def isBetween(value, limitUp, limitDown):
    if (value <= limitUp) and (value >= limitDown):
        return True
    else:
        return False


def removeOutliers(names, dataframe, startIdxs, stopIdxs):
    for i in range(len(names)):
        name = names[i]
        signal = dataframe[name]
        for j in range(len(startIdxs[name])): # startIdxs i stopIdxs osigurano da su iste duzine
            # interpoliraj signal izmedju startIdxs[j] i stopIdxs[j]
            signal = interpolate(signal, startIdxs[name][j], stopIdxs[name][j])

            plt.figure()
            plt.plot(range(len(signal)), signal)
            plt.title(name + ' interpolated signal')
            plt.xlabel('Timestamp')
            # full screen prikaz
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')

        dataframe[name] = signal

    return dataframe


def interpolate(signal, start, stop):
    diff = signal[stop] - signal[start-1]
    step = stop - (start-1)
    for i in range(start, stop):
        signal[i] = signal[i-1] + diff/step

    return signal


if __name__ == '__main__':
    main()