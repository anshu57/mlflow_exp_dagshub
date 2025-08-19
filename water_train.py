import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn

mlflow.set_experiment("water_exp2")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = pd.read_csv('/Users/anshugangwar/Desktop/Anshu/exp_mlflow/Data/water_potability.csv')
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)




def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)


from sklearn.ensemble import GradientBoostingClassifier
import pickle

X_train = train_processed_data.iloc[:,0:-1].values
y_train = train_processed_data.iloc[:,-1].values

n_estimators = 500

with mlflow.start_run():

    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    # save
    pickle.dump(clf, open('water_potability_model.pkl', 'wb'))

    X_test = test_processed_data.iloc[:,0:-1].values
    y_test = test_processed_data.iloc[:,-1].values

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    model = pickle.load(open('water_potability_model.pkl', 'rb'))
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)\
    
    mlflow.log_param("n_estimators", n_estimators)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(clf, "GradientBoostingClassifier")

    mlflow.log_artifact(__file__)

    mlflow.set_tag("author", "Anshu")
    mlflow.set_tag("model", "GB")



    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")    