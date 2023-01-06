import  pandas as pd
from  knn import  knn

coba = pd.read_csv("cobalagi.csv")

prediksi = knn.predict(coba)
print(prediksi)