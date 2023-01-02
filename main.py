import  pandas as pd
from  knn import  knn

coba = pd.read_csv("cobalagi.csv")

x_coba = coba.drop(["hasil"], axis=1)
prediksi = knn.predict(x_coba)
print(prediksi)