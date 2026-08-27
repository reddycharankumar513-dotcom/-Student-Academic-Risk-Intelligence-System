DRIVE READY 8.0

ML: ML TERMENOLOGY

DL : KNN, SVM, ANN, BACKPROPODATION, CNN, RNN,LSTM, TRANSFER LEARNING, NLP, OPENCV, LLM, recommendation engine

POWER BI : LR,LOGISTIC R, NB, BAGGING, BOOSTING, K MEANS,

PAYTHON : PANDAS, NUMPY, SKLEARN, MATPLOTLIB,TENSORFLOW,KERAS





DATA IS SYLLUBUS And brain is machine learning

DATA IS SYLLUBUS And brain is deep learning





**supervised learning** --> classification, regression

**unsupervised learning** --> clustering



knn, dt, rf, svm for both classification and regression

mb, logistic regression -> classification



logistic regression -> regression



**kNN Algorithm** means --> K nearest neighbour for this we use the formula of distance



knn regression averages

knn classification voting



averages

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

|(x2 - x1)2 + (y2 - y1)2 + (z2 - z1)2



mikeside                      |x1 - x2|

evident

multiatom

hamming

cosine similarity

Jaccard

lavishing

mb, logistic regression -> classification





**Data Type**

data collection --> data understanding --> data visualization  --> data preprocessing --> model building --> model revelation --> result prediction





**Linear regression :** it is of tow types

1\. **single linear regression formula** : y = mx + c

2\. **multiple linear regression**  **formula** :y =  m1x1 + m2x2 + ................ mnxn + c

to overcome the error we use the

**lasso regression**: cut the weak players entirely, keep only the strong ones it removes the unwanted features

**ridge regression**:  shrink everyone a bit, keep everyone in the team  it reduces the unwanted features





**logistic regression**: Logistic regression predicts the probability of a class (like yes/no) by squashing a linear combination of inputs through a sigmoid function into a range b/w 0 and 1

Entre Deep Learning is based on the sigmoid and softmax function



**Evaluation Metrix:** An Evaluation Metric is a quantitative measure used to assess how well a model, system, algorithm, or project performs against its intended objective. It provides an objective way to compare results, identify strengths and weaknesses, and determine whether the desired performance has been achieved



split the test into training and testing, In general random 80% training and 20% testing



**Confusion Metrix:** A Confusion Matrix is a table used to evaluate the performance of a classification model by comparing the model's predicted labels with the actual labels. It shows not only how many predictions are correct but also the types of errors the model makes.



| Actual / Predicted | Positive           | Negative            |

| ------------------ | ------------------ | ------------------- |

| Positive           | True Positive (TP) | False Negative (FN) |

| Negative           | False Positive (FP | True Negative (TN)  |





Components

* True Positive (TP): The model correctly predicts a positive class.

Example: A patient has a disease, and the model predicts "Disease."

* True Negative (TN): The model correctly predicts a negative class.

Example: A patient does not have a disease, and the model predicts "No Disease."

* False Positive (FP): The model incorrectly predicts a positive class (Type I Error).

Example: A healthy patient is predicted to have the disease.

* False Negative (FN): The model incorrectly predicts a negative class (Type II Error).

Example: A patient with the disease is predicted to be healthy.





Metrics Derived from a Confusion Matrix



* **Accuracy**: (TP+TN) /(TP+TN+FP+FN)

&#x09;​

* **Precision**:  TP / (TP+FP)

&#x09;​

* **Recall (Sensitivity)**: TP / (TP+FN)

&#x09;​

* **Specificity**: TN /(TN+FP)

&#x09;​

* **F1-Score**:  (2×Precision×Recall) / (Precision + Recall)



**K MEANS ALGORITHM:**   K-means is a popular unsupervised machine learning algorithm that partitions data into k clusters by minimizing the distance between points and their cluster centroids. It’s fast, scalable, and widely used for pattern recognition, customer segmentation, and anomaly detection.



HOW IT WORKS:

* Select number of clusters (k):

Decide how many groups you want to divide the data into.



* Initialize centroids:

Randomly choose k points from the dataset (or use smarter methods like K-means++) as the initial cluster centers.



* Assign points to clusters:

For each data point, calculate the distance to all centroids and assign it to the nearest one.



Common distance metric: Euclidean distance.



* Update centroids:

For each cluster, compute the mean of all points assigned to it. This mean becomes the new centroid.



* Repeat assignment and update:

Reassign points to the nearest centroid and update centroids again. Continue until:



Centroids stop changing significantly, or



A maximum number of iterations is reached.



**DISADVANTAGES**



Misclassification → happens near cluster boundaries.



Outliers → distort centroids and cluster quality.



Random centroids → cause unstable or suboptimal results.











