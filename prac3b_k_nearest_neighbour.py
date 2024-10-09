import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from mpl_toolkits.mplot3d import Axes3D 

# Load and display the sample data 
data = { 
    'Age': [19, 21, 20, 23, 31, 22, 35, 25, 23, 64, 30, 67, 35, 58, 24], 
    'Annual Income (k$)': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22], 
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 79, 65, 76, 76, 94], 
    'Segment': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]  # 0: Low-value, 1: High-value 
} 

df = pd.DataFrame(data) 
print("Sample Data:") 
print(df.head()) 

# Data Preprocessing 
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] 
y = df['Segment'] 

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 

# Apply KNN Algorithm 
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test) 

# Evaluation 
print("\nConfusion Matrix:") 
print(confusion_matrix(y_test, y_pred)) 
print("\nClassification Report:") 
print(classification_report(y_test, y_pred)) 
print("\nAccuracy Score:") 
print(accuracy_score(y_test, y_pred)) 

# Classify new user input 
new_user_data = {'Age': [27], 'Annual Income (k$)': [23], 'Spending Score (1-100)': [60]} 
new_user_df = pd.DataFrame(new_user_data) 
new_user_scaled = scaler.transform(new_user_df) 

new_user_segment = knn.predict(new_user_scaled) 
new_user_df['Segment'] = new_user_segment 
print("\nNew User Data Prediction:") 
print(new_user_df) 

# Visualization: Scatter plot of the customer segments 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Segment', data=df, palette='Set1', marker='o') 
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Segment', data=new_user_df, palette='Set2', marker='X', s=200) 
plt.title('Customer Segments with New User Input') 
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)') 
plt.legend() 
plt.show() 

