#Artifical Neural Networks

#Part 1 - Data preprocessing
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])
labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])
# No order to categorical to variables .. no precedence
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating ANN
# Importing Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initializing the ANN
classifier = Sequential()

# Adding the input layer & the first hidden layer with dropout
# (input + final output )/2 .. good recommendation for output dimensions
classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1))
# Adding the second hidden layer with dropout dropping 10% of neurons at each iteration
classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))
classifier.add(Dropout(rate = 0.1))
# Adding the final output layer
# Can use softmax function for more than 2 categories of outputs
classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))

# Compiling the ANN
# categorical_crossentropy loss in case of 3 or more categories of outputs
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting ANN to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting for a customer if they're gonna leave the bank
new_pred = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)
# Making predictions and evaluating model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Improving the ANN
# Adding dropout regularization

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu', input_dim = 11))
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation='relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# Creating a dictionary for hyperparams that need to be optimized
parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

