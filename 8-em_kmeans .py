from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import sklearn.metrics as sm
'''scikit-learn, an estimator for classification is a Python object that implements the methods fit(X, y) and predict(T)'''
'''scikit-learn comes with a few standard datasets, for instance the iris and digits datasets for classification and the boston house prices dataset for regression.'''


list1 = [0, 1, 2]  #?
def rename(s):                          #to alter mixed vlues
	list2 = []
	for i in s:
		if i not in list2:
			list2.append(i)
	for i in range(len(s)):
		pos = list2.index(s[i])
		s[i] = list1[pos]
	return s

iris_dataset = load_iris()
X = iris_dataset["data"]
y = iris_dataset["target"]              #actual output
#print(X)
print(y)


# k-means
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)                                   #train with data x- unsupervised learning-without considering target
y_kmeans = kmeans.predict(X)
print(y_kmeans)
plt.scatter(X[:, 0], X[:, 1],c=y_kmeans s=40, cmap='viridis') #scatter(x axis(all rows,1st column),y axis(all rows ,2nd column),content,size,type of map)
print(y_kmeans)
km = rename(y_kmeans)
print(km)
print("Accuracy KM : ", sm.accuracy_score(y, km))
plt.show()


# EM part
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_kmeans = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=40, cmap='viridis')
print(y_kmeans)
em = rename(y_kmeans)
print(em)
print("Accuracy EM : ", sm.accuracy_score(y, em))
plt.show()
