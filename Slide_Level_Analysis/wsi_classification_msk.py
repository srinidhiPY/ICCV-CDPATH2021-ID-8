import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
import utils as utils
from auc_delong_xu import auc_ci_Delong, delong_roc_test
from sklearn.metrics import confusion_matrix


FEATURE_START_INDEX = 0    # Best 0 - All features

####
def export_tree(forest):
    i_tree = 0
    for tree_in_forest in forest.estimators_:
        with open('trees/tree_' + str(i_tree) + '.dot', 'w') as my_file:
            tree.export_graphviz(tree_in_forest, out_file=my_file,
                                 feature_names=utils.heatmap_feature_names[:len(utils.heatmap_feature_names) - 1])
        i_tree += 1


def plot_roc(gt_y, prob_predicted_y, ROC_Value_Store, ROC_id):

    predictions = prob_predicted_y[:, 1]
    fpr, tpr, _ = roc_curve(np.array(gt_y).astype(int), predictions)
    roc_auc = auc(fpr, tpr)

    # Save tpr/fpr to plot ROC later
    ROC_Value = np.array([[fpr], [tpr]]).tolist()
    np.save(os.path.join(ROC_Value_Store, 'ROC' + '_' + str(ROC_id)), ROC_Value)

    plt.figure(0).clf()
    plt.plot(fpr, tpr, 'b', label=r'$AUC_{Proposed} = %0.4f$' % roc_auc)
    plt.title('ROC curves comparision')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    lg = plt.legend(loc='lower right', borderaxespad=1.)
    lg.get_frame().set_edgecolor('k')
    plt.grid(True, linestyle='-')
    plt.show()

    # ROC with confidence interval using De_Longs test
    auc_ver2, auc_var_ver2, ci = auc_ci_Delong(y_true=gt_y, y_scores=predictions)
    print('ROC AUC ver2 : %s' % auc_ver2)
    print('ROC AUC ver1 : %s' % roc_auc)
    print('Confidence Interval: %s (95%% confidence)' % str(ci))

    return gt_y, predictions

def validate(x, gt_y, clf, subset):

    predicted_y = clf.predict(x)
    prob_predicted_y = clf.predict_proba(x)

    tn, fp, fn, tp = confusion_matrix(gt_y, predicted_y).ravel()
    print('tp=', tp)
    print('fp=', fp)
    print('tn=', tn)
    print('fn=', fn)

    print('%s confusion matrix:' % subset)
    conf_matrix = pd.crosstab(gt_y, predicted_y, rownames=['Actual'], colnames=['Predicted'])

    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]
    fp = conf_matrix[1][0]
    fn = conf_matrix[0][1]

    acc = (tp + tn) / (tp + tn + fp + fn)
    print(pd.crosstab(gt_y, predicted_y, rownames=['Actual'], colnames=['Predicted']))
    print('accuracy', acc)
    return predicted_y, prob_predicted_y


def train(x, y):

    ####### Different classifiers  ###################

    # clf = svm.SVC(kernel='rbf', C=1000, probability=True)
    # clf.fit(x, y)

    # clf = XGBClassifier(n_estimators=50)
    # clf.fit(x, y)

    clf = RandomForestClassifier(n_estimators=100, random_state=12345)      # n_estimators=100, random_state=12345
    clf.fit(x, y)

    # clf = GaussianNB()
    # clf.fit(x, y)

    # clf = DecisionTreeClassifier(min_samples_split=100, random_state=99)
    # clf.fit(x, y)

    # clf = LogisticRegression()
    # clf.fit(x, y)

    # clf = KNeighborsClassifier(n_neighbors=5)
    # clf.fit(x, y)

    # clf = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=1, max_iter=1000)
    # clf.fit(x, y)

    return clf

##########
def load_train_validation_data(f_train, f_validation):

    df_train = pd.read_csv(f_train)
    df_validation = pd.read_csv(f_validation)

    n_columns = len(df_train.columns)

    feature_column_names = df_train.columns[FEATURE_START_INDEX:n_columns - 1]
    label_column_name = df_train.columns[n_columns - 1]

    return df_train[feature_column_names], df_train[label_column_name], df_validation[feature_column_names], df_validation[label_column_name]

#########
def load_test_data(f_test):

    df_test = pd.read_csv(f_test)

    n_columns = len(df_test.columns)

    feature_column_names = df_test.columns[FEATURE_START_INDEX:n_columns - 1]
    label_column_name = df_test.columns[n_columns - 1]

    return df_test[feature_column_names], df_test[label_column_name]


if __name__ == '__main__':

    ######## Train ML model  ##########
    train_x, train_label, validation_x, validation_label = load_train_validation_data(utils.TRAIN_HEATMAP_FEATURE_CSV, utils.VAL_HEATMAP_FEATURE_CSV)
    model = train(train_x, train_label)

    ######## Validate ML model  ##########
    predict_y, prob_predict_y = validate(validation_x, validation_label, model, 'Validation')
    plot_roc(validation_label, prob_predict_y, utils.ROC_Value_Store, utils.ROC_id)

    # export_tree(model)   # Save ML model

    ######## Test ML model  ##########
    test_x, test_label = load_test_data(utils.TEST_HEATMAP_FEATURE_CSV)
    predict_label, prob_predict_label = validate(test_x, test_label, model, 'Test')
    gt_label, predictions = plot_roc(test_label, prob_predict_label, utils.ROC_Value_Store, utils.ROC_id)


    ###### Test ML model 2 #############
    test_x, test_y = load_test_data(utils.TEST2_HEATMAP_FEATURE_CSV)
    predict_y, prob_predict_y = validate(test_x, test_y, model, 'Test')
    gt_y, predictions2 = plot_roc(test_y, prob_predict_y, utils.ROC_Value_Store, utils.ROC_id)

    # Calculate statistical significance using Delong P-Value
    z_value, p_value, log_pvalue = delong_roc_test(gt_y, predictions1, predictions2, sample_weight=None)
    print('log(pvalue) : %s' % log_pvalue)
    print('p_value : %s' % p_value)
    print('z_value : %s' % z_value)