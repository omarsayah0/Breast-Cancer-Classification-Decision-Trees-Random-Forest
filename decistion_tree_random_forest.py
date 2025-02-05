from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix , ConfusionMatrixDisplay , roc_auc_score , roc_curve
import numpy as np

def load_data():

    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)

    df['target'] = data.target

    x = df.iloc[:, :-1]

    y = df['target']

    return (x, y)

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(

        x, y ,test_size = 0.2, random_state = 42

    )

    return (x_train, x_test, y_train, y_test)

def set_model_tree(x_train, y_train, x_test):

    param = {

        'criterion':['gini', 'entropy'],

        'max_depth':[None, 5, 10, 15],

        'min_samples_split':[2, 5, 10]
    
    }

    model_tree = GridSearchCV(

        DecisionTreeClassifier(random_state = 42),

        param_grid = param,

        cv = 5 ,

        scoring = 'accuracy',

        n_jobs = -1

        )

    model_tree.fit(x_train, y_train)

    model_tree = model_tree.best_estimator_

    y_pred_tree = model_tree.predict(x_test) 

    y_pred_tree = pd.DataFrame(data=y_pred_tree)

    return (model_tree , y_pred_tree)

def class_report(y_test, y_pred_tree, y_pred_forest):

    report = classification_report(y_test, y_pred_tree, output_dict = True)

    report_forest = classification_report(y_test, y_pred_forest, output_dict = True)

    report = pd.DataFrame(report)

    report_forest = pd.DataFrame(report_forest)

    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(report.iloc[: -1 , :-2] , annot= True , fmt=".2f" , cmap="Blues" ,ax=axes[0])

    axes[0].set_title("Decision Tree Classification Report")

    sns.heatmap(report_forest.iloc[: -1 , :-2], annot= True, fmt=".2f", cmap="Blues",ax=axes[1])

    axes[1].set_title("Random Forest Classification Report")

    plt.xlabel("Metrics")

    plt.ylabel("Classes")

    plt.show()

def confu_matrix(y_test, y_pred_tree, y_pred_forest):

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    conf_tree = confusion_matrix(y_test, y_pred_tree)

    conf_forest = confusion_matrix(y_test, y_pred_forest)

    disp_tree = ConfusionMatrixDisplay(confusion_matrix=conf_tree)

    disp_forest = ConfusionMatrixDisplay(confusion_matrix=conf_forest)

    disp_tree.plot(cmap="Blues", ax=axes[0])

    axes[0].set_title("Decision Tree Confusion Matrix")

    disp_forest.plot(cmap="Blues", ax=axes[1])

    axes[1].set_title("Random Forest Confusion Matrix")

    plt.tight_layout()

    plt.show()

def roc(model_tree, model_forest, x_test, y_test):

    plt.figure(figsize=(6, 4))

    fpr_tree, tpr_tree, _ = roc_curve(y_test, model_tree.predict_proba(x_test)[:, 1])

    fpr_forest ,tpr_forest, _ = roc_curve(y_test, model_forest.predict_proba(x_test)[:, 1])

    plt.plot(fpr_tree, tpr_tree, label =f"Decision Tree AUC = {roc_auc_score(y_test, model_tree.predict_proba(x_test)[:, 1]):.2f}")

    plt.plot(fpr_forest , tpr_forest, label =f"Rnadom Forest AUC = {roc_auc_score(y_test, model_forest.predict_proba(x_test)[:, 1]):.2f}")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("Roc Curve")

    plt.legend()

    plt.show()

def show_tree(model_tree, x):

    plt.figure(figsize=(15, 10))

    plot_tree(model_tree ,
              
             feature_names = x.columns,

             class_names = ['Malignant', 'Benign'] ,

             filled = True,

             rounded = True,

             fontsize = 10 ,

             precision = 2

             )

    plt.title("Decision Tree Visualization", fontsize=15, fontweight='bold')

    plt.tight_layout()

    plt.show()

def show_forest(model_forest, x):

    plt.figure(figsize=(15, 10))

    plot_tree(model_forest.estimators_[0] ,
              
             feature_names = x.columns,

             class_names = ['Malignant', 'Benign'] ,

             filled = True,

             rounded = True,

             fontsize = 10 ,

             precision = 2

             )

    plt.title("Random Forest (Single Tree) Visualization", fontsize=15, fontweight='bold')

    plt.tight_layout()

    plt.show()

def set_model_forest(x_train, y_train, x_test):

    param = {

        'n_estimators':[50, 100, 200],

        'max_depth':[None, 5, 10],

        'criterion':['gini', 'entropy'],

        'min_samples_split':[2, 5]

    }

    model_forest = GridSearchCV(

        RandomForestClassifier(random_state = 42),

        param_grid = param,

        cv = 5, scoring = 'accuracy',

        n_jobs = -1

        )

    model_forest.fit(x_train, y_train)

    model_forest = model_forest.best_estimator_

    y_pred_forest = model_forest.predict(x_test)

    y_pred_forest = pd.DataFrame(data=y_pred_forest)

    return (model_forest, y_pred_forest)

def show_importances(x, model_forest):

    importances = model_forest.feature_importances_

    feature_names = x.columns

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))

    sns.barplot(x = importances[indices], y= feature_names[indices])

    plt.title("Feature Importances (Best RF)")

    plt.show()

def main():
    x, y = load_data()

    x_train, x_test, y_train, y_test = split_data(x, y)

    model_tree, y_pred_tree = set_model_tree(x_train, y_train, x_test)

    model_forest, y_pred_forest = set_model_forest(x_train, y_train, x_test)

    class_report(y_test, y_pred_tree, y_pred_forest)

    confu_matrix(y_test, y_pred_tree, y_pred_forest)

    roc(model_tree, model_forest, x_test, y_test)

    show_tree(model_tree, x)

    show_forest(model_forest, x)

    show_importances(x, model_forest)





if __name__ == "__main__":
    main()