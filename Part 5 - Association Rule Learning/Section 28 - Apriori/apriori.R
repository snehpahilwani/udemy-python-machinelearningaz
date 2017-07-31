#Apriori

#Data Preprocessing
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions(file='Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 100)

#Training apriori on the dataset
rules = apriori(data = dataset, 
                parameter = list(support = 0.004, confidence = 0.4))

#Visualizing the results
inspect(sort(rules, by = 'lift')[1:10])

