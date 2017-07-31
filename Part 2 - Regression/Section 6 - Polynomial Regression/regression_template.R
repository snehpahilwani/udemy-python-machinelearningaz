#Regression Template
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

#Fitting Regression Model to dataset
#Create regressor here


#Predicting a result with Poly Regression
y_pred = predict(poly_reg, data.frame(Level = 6.5))

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
