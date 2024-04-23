import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv(r"C:\Users\USER\Desktop\Final_Assignment - Copy\House Price Project\data.csv")

# Selecting columns excluding 'sqft_living'
columns = ['bedrooms', 'bathrooms', 'floors', 'price']
df = df[columns]

# Separating features (X) and target variable (y)
X = df.iloc[:, :-1]  # Excluding the last column, which is 'price'
y = df['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Training the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Save the trained model
pickle.dump(lr, open('model.pkl', 'wb'))
