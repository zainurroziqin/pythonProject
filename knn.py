import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from  sklearn.metrics import classification_report, confusion_matrix

# membaca csv
data = pd.read_csv("data.csv")
#memisah atribute dan class
x = data.drop(["hasil"], axis=1)
y = data["hasil"]

#membagi data traning dan data testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=1)

#iniasialisasi KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# prediksi traning data traing dan data testing
knn_pred_tr = knn.predict(x_train)
knn_pred_te = knn.predict(x_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, knn_pred_te)
print(cm)

# Menghitung akurasi training
print('----- Evaluation on Training Data -----')
score_tr = knn.score(x_train, y_train)
print('Accuracy Score: ', score_tr)
# Menghitung akurasi testing
print(classification_report(y_train, knn_pred_tr))
print('--------------------------------------------------------')
print('----- Evaluation on Test Data -----')
score_te = knn.score(x_test, y_test)
print('Accuracy Score: ', score_te)
# Look at classification report to evaluate the model
print(classification_report(y_test, knn_pred_te))
print('--------------------------------------------------------')

