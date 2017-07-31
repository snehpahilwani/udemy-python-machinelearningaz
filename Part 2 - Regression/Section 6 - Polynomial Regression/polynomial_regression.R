#Polynomial Regression

#Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#Small dataset -- NO splitting required
  
#Fitting Linear Regression to dataset
lin_reg = lm(formula = Salary ~ .,
             data = dataset)


#Fitting Poly Regression to dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
             data = dataset)

#Visualizing linear reg
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y =  predict(lin_reg, newdata = dataset)),
            colour = 'blue') + 
  ggtitle('Truth or Bluff(Linear Regression Results)') +
  xlab('Level') + 
  ylab('Salary')


#Visualizing poly reg
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y =  predict(poly_reg, newdata = dataset)),
            colour = 'blue') + 
  ggtitle('Truth or Bluff(Linear Regression Results)') +
  xlab('Level') + 
  ylab('Salary')

#Predicting a result with Linear Regression
y_pred = predict(lin_reg, data.frame(Level = 6.5))

#Predicting a result with Poly Regression
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))