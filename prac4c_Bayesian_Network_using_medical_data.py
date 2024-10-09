import numpy as np 
import pandas as pd 
from pgmpy.models import BayesianNetwork 
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination 
import networkx as nx 
import matplotlib.pyplot as plt 

# Create sample medical data (heart disease example) 
data = pd.DataFrame({'Age': [30, 40, 50, 60, 70], 
                     'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'], 
                     'ChestPain': ['Typical', 'Atypical', 'Typical', 'Atypical', 'Typical'], 
                     'HeartDisease': ['Yes', 'No', 'Yes', 'No', 'Yes']}) 

# Create a Bayesian Network model 
model = BayesianNetwork([('Age', 'HeartDisease'), 
                         ('Gender', 'HeartDisease'), 
                         ('ChestPain', 'HeartDisease')]) 

# Fit the model to the data using Maximum Likelihood Estimation 
model.fit(data, estimator=MaximumLikelihoodEstimator) 

# Plot the Bayesian network 
pos = nx.circular_layout(model) 
nx.draw(model, pos, with_labels=True, node_size=5000, node_color="skyblue", font_size=12, font_color="black") 
plt.title("Bayesian Network Structure") 
plt.show() 

# Print Conditional Probability Distributions (CPDs) 
for cpd in model.get_cpds(): 
    print("CPD of", cpd.variable) 
    print(cpd) 

# Inference and diagnosis 
inference = VariableElimination(model) 
query = inference.query(variables=['HeartDisease'], evidence={'Age': 50, 'Gender': 'Male', 'ChestPain': 'Typical'}) 
print(query) 
