import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.metrics import classification_report, confusion_matrix


def load_fashion_data(train_size=0):

    # Ref: https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
    # Ref: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    # Data source (Fashion MNIST): https://www.kaggle.com/zalando-research/fashionmnist

    # Note: Fashion MNIST data is already split between training and test sets, in separate files

    # Load training data
    train_data = pd.read_csv('./data/fashion-mnist_train.csv')
    # print train_data.isnull().any()  # no null values

    # Shuffle training data
    # Ref: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    # Split features (X) from labels (y)
    X = train_data.drop('label', axis=1)
    y = train_data['label']

    # Check class distribution before splitting train data
    # hist = y.hist()
    # plt.show()

    # Sample data if specified (stratified sampling to maintain proportionality of classes)
    if train_size != 0:
        X_train, X_unused, y_train, y_unused = train_test_split(X, y, train_size=train_size, random_state=42,
                                                                stratify=y)
    else:
        X_train, y_train = X, y

    # Check class distribution after splitting train data
    # hist = y_train.hist()
    # plt.show()

    # Load test data
    test_data = pd.read_csv('./data/fashion-mnist_test.csv')
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    return X_train, X_test, y_train, y_test


def load_adult_data(train_size=0):

    # Data source (Adult): https://archive.ics.uci.edu/ml/datasets/Adult

    # Note: Adult data is provided already split between training and test sets, in separate files.  These are merged
    # here, pre-processed as a single file, then split back into new Train / Test sets.

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'label']

    # Load and merge data
    train_data = pd.read_csv('./data/adult.data', header=None, names=column_names, skipinitialspace=True)
    test_data = pd.read_csv('./data/adult.test', header=None, names=column_names, skipinitialspace=True, skiprows=[0])
    all_data = train_data.append(test_data, ignore_index=True).reset_index(drop=True)

    print "train_data shape:", train_data.shape
    print "test_data shape:", test_data.shape
    print "all_data shape:", all_data.shape
    print

    # Segregate numerical from categorical features (dropping captital-gain and capital-loss as it's 0 for most
    numerical = ['age', 'fnlwgt', 'education-num', 'hours-per-week']
    categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'native-country']
    labels = ['label']
    selection = numerical + categorical + labels
    select_all_data = all_data[selection]

    # Drop instances with missing / poor data
    # print "Empty / null values:\n", select_all_data.isnull().any()  # no null values
    # print select_all_data[select_all_data.sex.isin(['Male'])]
    # print select_all_data[select_all_data.age.isnull()]
    # Remove inconsistent '.' from label
    # https://stackoverflow.com/questions/13682044/remove-unwanted-parts-from-strings-in-a-column
    select_all_data['label'] = select_all_data['label'].map(lambda x: x.rstrip('.'))
    # Map '<=50K' to 0, '>50K' to 1
    select_all_data['label'] = select_all_data['label'].replace('<=50K', 0)
    select_all_data['label'] = select_all_data['label'].replace('>50K', 1)
    clean_data = select_all_data.dropna(axis=0).reset_index(drop=True)

    # clean_data.groupby('label').hist()
    # plt.show()

    print '\nclean_data shape:', clean_data.shape
    print "\nduplicate rows count:", len(all_data[all_data.duplicated(selection)])
    # print "\nduplicate rows:", all_data[all_data.duplicated(selection)]

    # Extract features and labels
    X = clean_data.drop('label', axis=1)
    y = clean_data['label']
    print '\nX shape:', X.shape
    print 'y shape:', y.shape

    # Perform one-hot-encoding for categorical features
    encoded_X = pd.get_dummies(X, columns=categorical)
    print "encoded_X shape:", encoded_X.shape
    print

    # Split into Train and Test sets

    # Check class distribution before splitting train data
    # print 'y describe:\n', y.describe()
    # print
    # hist = y.hist()
    # plt.show()

    # Split data into training and testing sets (shuffled and stratified)
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.25, random_state=42, stratify=y,
                                                        shuffle=True)

    # Check class distribution after splitting train data
    # hist = y_train.hist()
    # plt.show()

    # Further sample Train if specified (stratified to maintain proportionality of classes)

    # Check class distribution before splitting train data
    # hist = y_test.hist()
    # plt.show()

    if train_size != 0:
        X_train_sampled, X_train_unused, y_train_sampled, y_train_unused = train_test_split(X_train, y_train,
                                                                                            train_size=train_size,
                                                                                            random_state=42,
                                                                                            stratify=y_train,
                                                                                            shuffle=True)
    else:
        X_train_sampled, y_train_sampled = X_train, y_train

    # Check train set sizes after sampling
    print 'X_train_sampled shape:', X_train_sampled.shape
    print 'y_train_sampled shape:', y_train_sampled.shape

    # Check class distribution after splitting train data
    # hist = y_train_sampled.hist()
    # plt.show()

    # Scale numerical features
    # https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
    # https://stackoverflow.com/questions/38420847/apply-standardscaler-on-a-partial-part-of-a-data-set
    if len(numerical) > 0:
        X_train_scaled = X_train_sampled.copy()
        X_test_scaled = X_test.copy()
        X_train_numerical = X_train_scaled[numerical]
        X_test_numerical = X_test_scaled[numerical]
        scaler = preprocessing.StandardScaler().fit(X_train_numerical)  # Fit using only Train data
        numerical_X_train = scaler.transform(X_train_numerical)
        numerical_X_test = scaler.transform(X_test_numerical)  # transform X_test with same scaler as X_train
        X_train_scaled[numerical] = numerical_X_train
        X_test_scaled[numerical] = numerical_X_test
    else:
        X_train_scaled = X_train_sampled
        X_test_scaled = X_test

    print "\nX_train_scaled shape:", X_train_scaled.shape
    print "X_test_scaled shape:", X_test_scaled.shape
    print "y_train shape:", y_train.shape
    print "y_test shape:", y_test.shape

    # Select important features based on correlation analysis
    # plot_correlation(X_train_scaled, y_train)
    # Features to keep: Corr. > 0.03 and Corr. < -0.05 (to start with), based on correlation plot.
    # And dropping 'sex_Female' given inverse correlation with 'sex_Male'
    # Current...
    features_to_keep = ['marital-status_Married-civ-spouse', 'relationship_Husband', 'education-num', 'hours-per-week',
                        'age', 'sex_Male', 'occupation_Exec-managerial', 'occupation_Prof-specialty',
                        'education_Bachelors', 'education_Masters', 'education_Prof-school', 'workclass_Self-emp-inc',
                        'education_Doctorate', 'relationship_Wife', 'race_White', 'workclass_Federal-gov',
                        'workclass_Local-gov', 'native-country_United-States', 'education_9th',
                        'occupation_Farming-fishing', 'education_Some-college', 'education_7th-8th',
                        'native-country_Mexico', 'marital-status_Widowed', 'education_10th',
                        'occupation_Machine-op-inspct', 'marital-status_Separated', 'workclass_Private',
                        'workclass_?', 'occupation_?', 'occupation_Adm-clerical', 'occupation_Handlers-cleaners',
                        'education_11th', 'relationship_Other-relative', 'race_Black', 'marital-status_Divorced',
                        'education_HS-grad', 'relationship_Unmarried', 'occupation_Other-service',
                        'relationship_Not-in-family', 'relationship_Own-child',
                        'marital-status_Never-married']

    # # Trying less... Top 10: Corr. > 0.18.  Bottom 11: Corr. < 0.086 (going from 43 to 21)
    # features_to_keep = ['marital-status_Married-civ-spouse', 'relationship_Husband', 'education-num', 'hours-per-week',
    #                     'age', 'sex_Male', 'occupation_Exec-managerial', 'occupation_Prof-specialty',
    #                     'education_Bachelors', 'education_Masters',
    #                     'education_11th', 'relationship_Other-relative', 'race_Black', 'marital-status_Divorced',
    #                     'education_HS-grad', 'relationship_Unmarried', 'occupation_Other-service',
    #                     'relationship_Not-in-family', 'relationship_Own-child',
    #                     'marital-status_Never-married']

    # # Trying even less... Corr. > 0.2.  Corr. < 10.2 (going from 43 to 9)
    # features_to_keep = ['marital-status_Married-civ-spouse', 'relationship_Husband', 'education-num', 'hours-per-week',
    #                     'age', 'sex_Male', 'occupation_Exec-managerial', 'relationship_Own-child',
    #                     'marital-status_Never-married']

    # # Try top 10 positive Corr. only
    # features_to_keep = ['marital-status_Married-civ-spouse', 'relationship_Husband', 'education-num', 'hours-per-week',
    #                     'age', 'sex_Male', 'occupation_Exec-managerial', 'occupation_Prof-specialty',
    #                     'education_Bachelors', 'education_Masters']

    final_X_train = X_train_scaled[features_to_keep]
    final_X_test = X_test_scaled[features_to_keep]

    # plot_correlation(final_X_train, y_train_sampled)

    print "\nfinal_X_train shape:", final_X_train.shape
    print "final_X_test shape:", final_X_test.shape
    print "y_train shape:", y_train.shape
    print "y_test shape:", y_test.shape
    print

    return final_X_train, final_X_test, y_train_sampled, y_test


def plot_correlation(X, y):
    # https://likegeeks.com/seaborn-heatmap-tutorial/
    # https://medium.com/@chrisshaw982/seaborn-correlation-heatmaps-customized-10246f4f7f4b
    # https://github.com/mwaskom/seaborn/issues/430

    all_processed_data = pd.concat([X, y], axis=1)
    correlation_matrix = all_processed_data.corr()
    # 1. Filtering for only correlations with 'label'
    plt.figure(figsize=(20, 50))
    heat_map = sb.heatmap(correlation_matrix[['label']].sort_values(by=['label'], ascending=False),
                          vmin=-1, cmap='coolwarm', annot=True, annot_kws={"size": 20})
    # 2. Full correlation matrix
    # plt.figure(figsize=(50, 50))
    # heat_map = sb.heatmap(correlation_matrix,
    #                       vmin=-1, cmap='coolwarm')
    plt.tick_params(axis='both', labelsize=20)
    plt.grid()
    plt.show()


def mean_scores(train_scores, validation_scores, index_values):
    # Calculate mean results from k-folds cross-validation
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)
    # print 'Mean training scores\n\n', pd.Series(train_scores_mean, index=index_values)
    # print '\n', '-' * 20  # separator
    # print'\nMean validation scores\n\n', pd.Series(validation_scores_mean, index=index_values)
    return train_scores_mean, validation_scores_mean


def plot_learning_curves(train_sizes, train_scores_mean, validation_scores_mean, chart_title):
    plt.plot(train_sizes, train_scores_mean, label='Training score')
    plt.plot(train_sizes, validation_scores_mean, label='Validation score')
    plt.ylabel('Score', fontsize=26)
    plt.xlabel('# Training Examples', fontsize=26)
    plt.title(chart_title, fontsize=28, y=1.03)
    # Ref: https://stackoverflow.com/questions/7125009/how-to-change-legend-size-with-matplotlib-pyplot
    plt.legend(loc='best', prop={'size': 26})
    plt.ylim(0, 1.02)
    # Ref: https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller/11386056#11386056
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid()
    plt.show()


def learning_curves(X_train, y_train):

    # Ref: https://www.dataquest.io/blog/learning-curves-machine-learning/
    # Note: uncomment classifier to be used

    t0 = time.time()
    print 'Learning curves time started:', time.strftime('%X %x %Z')

    # Decision Tree
    # classifier = DecisionTreeClassifier(criterion='entropy', max_depth=13, min_samples_leaf=1)
    # train_sizes = [1, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 10000, 40000]
    # cv_value = 3
    # chart_title = 'Decision Tree Classifier'

    # KNN
    classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2, n_jobs=-1)
    # classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2, weights='distance', n_jobs=-1)
    # classifier = KNeighborsClassifier(n_neighbors=7, metric='hamming', n_jobs=-1)
    train_sizes = [15, 200, 400, 1000, 2000, 10000, 24420]
    cv_value = 3
    chart_title = 'kNN Classifier'

    # ANN
    # # classifier = MLPClassifier(random_state=42, alpha=1, max_iter=200)
    # # Defaults
    # # classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.19,
    # #                            learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=42)
    # classifier = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', alpha=1.3e-1,
    #                            learning_rate='adaptive', learning_rate_init=0.001, max_iter=200, random_state=42)
    # # classifier = MLPClassifier(hidden_layer_sizes=(1000, 1000), activation='relu', solver='adam', alpha=0.19,
    # #                            learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=42)
    #
    # # classifier = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', solver='sgd', alpha=100,
    # #                            learning_rate='adaptive', learning_rate_init=0.001, max_iter=200, random_state=42)
    # # train_sizes = [1, 200, 400, 1000, 10000, 40000]
    # train_sizes = [1, 200, 400, 1000, 10000]
    # cv_value = 3
    # chart_title = 'Neural Network Classifier'

    # SVM
    # # Final, for 'rbf' kernel
    # classifier = SVC(C=1.0, kernel='rbf', gamma=3.7e-01, cache_size=10000)
    # # classifier = SVC(C=0.1, kernel='linear', cache_size=10000)
    # # classifier = SVC(C=1.0, kernel='sigmoid', gamma=0.03, cache_size=10000)
    # train_sizes = [20, 200, 1000, 2000, 10000]
    # cv_value = 3
    # chart_title = 'SVM Classifier'

    # Boosting
    # classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7),
    #                                 n_estimators=10, learning_rate=0.7)
    # train_sizes = [1, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 10000, 24420]
    # cv_value = 3
    # chart_title = 'Boosted Decision Tree Classifier'

    # Set scoring metric
    # scoring_metric = 'accuracy'
    scoring_metric = 'f1_weighted'

    # Get learning curves
    train_sizes, train_scores, validation_scores = learning_curve(classifier, X_train, y_train,
                                                                  train_sizes=train_sizes, cv=cv_value, shuffle=True,
                                                                  scoring=scoring_metric)

    print 'Learning curves time ended:', time.strftime('%X %x %Z')
    print 'Done in %0.3fs' % (time.time() - t0)
    print

    # Calculate mean results from k-folds cross-validation
    train_scores_mean, validation_scores_mean = mean_scores(train_scores, validation_scores, train_sizes)

    # Plot learning curves
    plot_learning_curves(train_sizes, train_scores_mean, validation_scores_mean, chart_title)

    # Output results to CSV file
    ds_train_scores = pd.Series(train_scores_mean, index=train_sizes, name='train_scores')
    ds_validation_scores = pd.Series(validation_scores_mean, index=train_sizes, name='validation_scores')
    df_results = pd.concat([ds_train_scores, ds_validation_scores], axis=1)

    print df_results

    # df_results.to_csv('learning_curves.csv')


def plot_validation_curves(train_scores_mean, validation_scores_mean, parameter, param_range, plot_type, chart_title):
    plt.title(chart_title, fontsize=28, y=1.03)
    plt.xlabel(parameter, fontsize=26)
    plt.ylabel("Score", fontsize=26)
    plt.ylim(0.0, 1.02)

    # param_range = ['5', '10', '20', '50', '100', '500', '1000', '5000']
    # param_range = ['1', '3', '5', '10', '20', '30']
    # param_range = ['1', '2', '3', '4', '5']
    # param_range = ['1', '2', '3', '4', '5', '6', '7', '8']
    # param_range = ['1', '2', '3']

    # Ref: https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller/11386056#11386056
    plt.tick_params(axis='both', which='major', labelsize=24)
    if plot_type == 'log_scale':
        plt.semilogx(param_range, train_scores_mean, label="Training score")
        plt.semilogx(param_range, validation_scores_mean, label="Validation score")
    if plot_type == 'linear_scale':
        plt.plot(param_range, train_scores_mean, label="Training score")
        plt.plot(param_range, validation_scores_mean, label="Validation score")
    # Ref: https://stackoverflow.com/questions/7125009/how-to-change-legend-size-with-matplotlib-pyplot
    plt.legend(loc='best', prop={'size': 26})
    plt.grid()
    plt.show()


def validation_curves(X, y):

    # Ref: http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
    # Ref: https://scikit-learn.org/stable/modules/learning_curve.html
    # Note: uncomment classifier to be used

    t0 = time.time()
    print 'Validation curves time started:', time.strftime('%X %x %Z')
    print 'X shape:', X.shape
    print 'Y shape:', y.shape

    # # Decision Tree
    # classifier = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_leaf=1)
    # # parameter = 'max_depth'  # Parameter to be used in validation curve
    # # param_range = np.arange(1, 25, 3)
    # parameter = 'min_samples_leaf'  # Parameter to be used in validation curve
    # param_range = np.arange(1, 30, 3)
    # cv_value = 3
    # chart_title = 'Decision Tree Classifier'
    # plot_type = 'linear_scale'

    # KNN
    # classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2, weights='distance', n_jobs=-1)
    # # classifier = KNeighborsClassifier(n_neighbors=7, metric='hamming', n_jobs=-1)
    # parameter = 'n_neighbors'  # Parameter to be used in validation curve
    # param_range = np.arange(1, 15, 3)
    # # parameter = 'p'
    # # param_range = np.arange(1,10,1)
    # cv_value = 3
    # chart_title = 'kNN Classifier'
    # plot_type = 'linear_scale'

    # ANN
    # # Default
    # # classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
    # #                            learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=42)
    # # classifier = MLPClassifier(hidden_layer_sizes=(1000, 1000), activation='relu', solver='adam', alpha=0.19,
    # #                            learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=42)
    # classifier = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', alpha=1.3e-1,
    #                            learning_rate='adaptive', learning_rate_init=0.001, max_iter=200, random_state=42)
    # # parameter = 'alpha'  # Parameter to be used in validation curve
    # # param_range = np.logspace(-5, 5, 10)
    # # parameter = 'hidden_layer_sizes'  # Parameter to be used in validation curve
    # # param_range = ((5, 5), (10, 10), (20, 20), (50, 50), (100, 100), (200, 200), (500, 500), (1000, 1000))
    # # param_range = ((100,), (100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100))
    # # param_range = ((3,), (3, 3), (3, 3, 3), (3, 3, 3, 3), (3, 3, 3, 3, 3))
    # # param_range = ((1,), (1, 1), (1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1, 1))
    # # param_range = ((1000,), (1000, 1000), (1000, 1000, 1000))
    # # param_range = ((5,), (10,), (20,), (50,), (100,), (500,), (1000,), (5000,))
    # parameter = 'max_iter'  # Parameter to be used in validation curve
    # param_range = (10, 50, 100, 150, 200, 500)
    # cv_value = 3
    # chart_title = 'Neural Network Classifier'
    # plot_type = 'linear_scale'

    # SVM
    # # classifier = SVC(C=1.0, kernel='rbf', gamma=3.7e-01, cache_size=10000)
    # # classifier = SVC(C=1.0, kernel='linear', cache_size=10000)
    # classifier = SVC(C=1.0, kernel='sigmoid', gamma='auto', cache_size=10000)
    # # parameter = 'gamma'  # Parameter to be used in validation curve
    # # param_range = np.logspace(-3, 3, 5)
    # parameter = 'C'  # Parameter to be used in validation curve
    # param_range = np.logspace(-3, 3, 5)
    # cv_value = 3
    # chart_title = 'SVM Classifier'
    # plot_type = 'log_scale'

    # Boosting
    classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42, max_depth=20),
                                    n_estimators=36, learning_rate=1.0)
    parameter = 'learning_rate'  # Parameter to be used in validation curve
    param_range = np.arange(0.1, 1.1, 0.1)
    # parameter = 'n_estimators'  # Parameter to be used in validation curve
    # param_range = np.arange(1, 51, 10)
    # parameter = 'base_estimator'   # Parameter to be used in validation curve
    # param_range = (DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3),
    #                DecisionTreeClassifier(max_depth=5), DecisionTreeClassifier(max_depth=10),
    #                DecisionTreeClassifier(max_depth=20), DecisionTreeClassifier(max_depth=30))
    cv_value = 3
    chart_title = 'Boosted Decision Tree Classifier'
    plot_type = 'linear_scale'

    # Set scoring metric
    # scoring_metric = 'accuracy'
    scoring_metric = 'f1_weighted'

    # Get validation curves
    train_scores, validation_scores = validation_curve(
        classifier, X, y, param_name=parameter, param_range=param_range,
        cv=cv_value, scoring=scoring_metric, n_jobs=1)

    print 'Validation curves time ended:', time.strftime('%X %x %Z')
    print 'Done in %0.3fs' % (time.time() - t0)
    print

    # Calculate mean results from k-folds cross-validation
    train_scores_mean, validation_scores_mean = mean_scores(train_scores, validation_scores, param_range)

    # Plot validation curves
    plot_validation_curves(train_scores_mean, validation_scores_mean, parameter, param_range, plot_type, chart_title)

    # Output results to CSV file
    ds_train_scores = pd.Series(train_scores_mean, index=param_range, name='train_scores')
    ds_validation_scores = pd.Series(validation_scores_mean, index=param_range, name='validation_scores')
    df_results = pd.concat([ds_train_scores, ds_validation_scores], axis=1)

    print df_results

    # df_results.to_csv('validation_curves.csv')


def prediction(X_train, y_train, X_test, y_test):

    # Note: insert relevant optimal parameters based on learning and validation curve analysis
    # Note: manually iterate for increase training_size size to generate Test learning curve.

    # class_labels = ['0 T-shirt/top', '1 Trouser', '2 Pullover', '3 Dress', '4 Coat', '5 Sandal', '6 Shirt',
    #                 '7 Sneaker', '8 Bag', '9 Ankle boot']

    class_labels = ['0 <=50K', '1 >50K']

    # Check train and test data set sizes
    print 'X_train shape:', X_train.shape
    print 'y_train shape:', y_train.shape
    print 'X_test shape:', X_test.shape
    print 'y_test shape:', y_test.shape

    # Decision Tree
    # classifier = DecisionTreeClassifier(criterion='entropy', max_depth=13, min_samples_leaf=1)

    # KNN
    # classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)

    # ANN
    # classifier = MLPClassifier(hidden_layer_sizes=(1000,), activation='logistic', solver='sgd', alpha=100,
    #                            learning_rate='adaptive', learning_rate_init=0.001, max_iter=100, random_state=42)

    # ANN temp
    # classifier = MLPClassifier(hidden_layer_sizes=(1000, 1000), activation='relu', solver='adam', alpha=0.19,
    #                            learning_rate='constant', learning_rate_init=0.001, max_iter=200, random_state=42)

    # SVM
    # classifier = SVC(C=1, kernel='rbf', gamma=3.7e-01, cache_size=10000)

    # Boosting
    classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7),
                                    n_estimators=10, learning_rate=0.7)

    # Fit
    t0 = time.time()
    print '\nTime fit started:', time.strftime('%X %x %Z')
    classifier.fit(X_train, y_train)
    print 'Time fit ended:', time.strftime('%X %x %Z')
    print 'Fit done in %0.3fs' % (time.time() - t0)

    # Predict
    t0 = time.time()
    print '\nTime predict started:', time.strftime('%X %x %Z')
    y_pred = classifier.predict(X_test)
    print 'Time predict ended:', time.strftime('%X %x %Z')
    print 'Predict done in %0.3fs' % (time.time() - t0)

    # Print results
    print "\nClassification report:\n", classification_report(y_test, y_pred, target_names=class_labels)
    print "Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(2))


if __name__ == "__main__":

    # To run this, make sure the data files are in a sub-directory "./data".  Refer to README.txt for instructions.

    # Load and pre-process Fashion MNIST data
    # training_size = 20000  # Set to 0 for Learning Curves
    # X_train, X_test, y_train, y_test = load_fashion_data(training_size)

    # Load and pre-process Adult data
    training_size = 0  # Set to 0 for Learning Curves
    X_train, X_test, y_train, y_test = load_adult_data(training_size)

    # Learning curves
    learning_curves(X_train, y_train)

    # Validation curves
    # validation_curves(X_train, y_train)

    # Prediction
    # prediction(X_train, y_train, X_test, y_test)
