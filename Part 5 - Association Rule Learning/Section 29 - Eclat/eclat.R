#Eclat

#Data Preprocessing
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions(file='Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 100)

#Training apriori on the dataset
rules = eclat(data = dataset, 
                parameter = list(support = 0.003, minlen = 2))

#Visualizing the results
inspect(sort(rules, by = 'support')[1:10])

