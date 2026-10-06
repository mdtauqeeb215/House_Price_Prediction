import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load dataset
df = pd.read_csv("house_data.csv")

print("Dataset loaded successfully!")
print(df.head())


# 2. Basic data analysis
print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
df.info()

print("\nAverage Price:", df["price"].mean())
print("Maximum Price:", df["price"].max())
print("Minimum Price:", df["price"].min())
print("Average Area:", df["area"].mean())

print("\nLocation-wise Average Price:")
print(df.groupby("location")["price"].mean())


# 3. Separate features and target
X = df[["area", "bedrooms", "bathrooms", "location"]]
y = df["price"]


# 4. Convert location into numerical values
X = pd.get_dummies(X, columns=["location"], dtype=int)


# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# 7. Make predictions
y_pred = model.predict(X_test)


# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error:", round(mae))
print("R2 Score:", round(r2, 4))


# 9. Actual vs Predicted Price graph
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price (₹)")
plt.ylabel("Predicted Price (₹)")
plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

os.makedirs("results", exist_ok=True)
plt.savefig("results/actual_vs_predicted.png")
plt.show()


# 10. Take input from the user
area = int(input("\nEnter house area (sq.ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
location = input("Enter location (Kolkata/Patna): ")


# 11. Create input DataFrame
new_house = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "location": [location]
})


# 12. Encode location
new_house = pd.get_dummies(
    new_house,
    columns=["location"],
    dtype=int
)


# Make sure input columns match training columns
new_house = new_house.reindex(
    columns=X.columns,
    fill_value=0
)


# 13. Predict house price
predicted_price = model.predict(new_house)


# 14. Display prediction
print("\nPredicted House Price: ₹", round(predicted_price[0]))
print(
    "Predicted Price in Lakhs: ₹",
    round(predicted_price[0] / 100000, 2),
    "Lakh"
)