#SVR

#Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
#Splitting data into training and test set
#install.packages('caTools')
#library(caTools)
#set.seed(123)
#split = sample.split(dataset$Salary, SplitRatio = 2/3)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

#Feature scaling
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting SVR to dataset
#install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~ .,
                data = dataset, 
                type = 'eps-regression')


#Predicting a result with SVR Regression
y_pred = predict(regressor, data.frame(Level = 6.5))

#Visualizing poly reg (high resolution)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y =  predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') + 
  ggtitle('Truth or Bluff(Regression Model)') +
  xlab('Level') + 
  ylab('Salary')

