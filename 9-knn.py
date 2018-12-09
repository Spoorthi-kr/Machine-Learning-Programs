from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris_dataset=load_iris()
print("DATA SET:",iris_dataset)
print("DATA :",iris_dataset.data)
print("TARGET NAMES:",iris_dataset.target_names)
print("TARGET :",iris_dataset.target)

'''print("\n IRIS DATA :\n",iris_dataset["data"])
print("\n Target :\n",iris_dataset["target"])'''


'''
print("\n IRIS FEATURES \ TARGET NAMES:")
for i in range(len(iris_dataset.target_names)):
    print("\n[{0}]:[{1}]".format(i,iris_dataset.target_names[i]))'''



X_train, X_test = train_test_split(iris_dataset.data, random_state=0)
y_train, y_test = train_test_split(iris_dataset.target, random_state=0)

print("\n X TRAIN \n", X_train)
print("\n X TEST \n", X_test)
print("\n Y TRAIN \n", y_train)
print("\n Y TEST \n", y_test)

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(X_train, y_train)
prediction = kn.predict(X_test)

for i in range(len(X_test)):
    print("\n Actual : {0} {1}, Predicted :{2} {3}".format(y_test[i], iris_dataset["target_names"][y_test[i]],
                                                           prediction[i], iris_dataset["target_names"][prediction][i]))
print("\n TEST SCORE[ACCURACY]: {:.2f}\n".format(kn.score(X_test, y_test)))
