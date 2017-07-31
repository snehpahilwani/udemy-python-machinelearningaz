#Importing the dataset

dataset = read.csv('50_Startups.csv')
#dataset = dataset[, 2:3]

#Splitting data into training and test set
#install.packages('caTools')

dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))


library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature scaling
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting Multiple Linear Regression to the training set
#dot means all the independent variables
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)
summary(regressor)
#Removing variables with high P-values and keeping relevant variables
regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)
summary(regressor)
#Predicting the test set results
y_pred = predict(regressor, newdata = test_set)