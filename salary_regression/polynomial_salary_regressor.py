import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


class PolynomialSalaryRegressor:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = None
        self.poly_features = None

    def load_data(self, filepath):
        """Load CSV data and split into X and y."""
        df = pd.read_csv(filepath)
        self.X = df[["YearsExperience"]].values  # 2D array
        self.y = df["Salary"].values

    def train(self):
        """Train the polynomial regression model."""
        self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False, interaction_only=False)
        X_poly = self.poly_features.fit_transform(self.X)
        self.model = LinearRegression()
        self.model.fit(X_poly, self.y)

    def predict(self, years_exp):
        """Predict salary for given years of experience (float or list of floats)."""
        years_exp = np.array(years_exp).reshape(-1, 1)
        X_poly = self.poly_features.transform(years_exp)
        return self.model.predict(X_poly)

    def plot(self):
        """Plot the original data and the polynomial regression curve."""
        plt.scatter(self.X, self.y, color="red", label="Actual")
        X_grid = np.linspace(min(self.X), max(self.X), 100).reshape(-1, 1)
        X_grid_poly = self.poly_features.transform(X_grid)
        y_pred_grid = self.model.predict(X_grid_poly)
        plt.plot(X_grid, y_pred_grid, color="blue", label="Polynomial Fit")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title(f"Polynomial Regression (Degree {self.degree})")
        plt.legend()
        plt.show()

    def evaluate(self):
        """Return the mean squared error on training data."""
        X_poly = self.poly_features.transform(self.X)
        y_pred = self.model.predict(X_poly)
        return mean_squared_error(self.y, y_pred)



if __name__ == "__main__":
    regressor = PolynomialSalaryRegressor(degree=3)
    regressor.load_data("resources/Salary_data.csv")
    regressor.train()
    regressor.predict([5.5]) # Predict salary for 5.5 years of experience
    print("MSE on training data:", regressor.evaluate())
    regressor.plot()
    #While loop to allow multiple predictions
    while True:
        try:
            user_input = input("Enter years of experience to predict salary (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            years_exp = float(user_input)
            if years_exp < 0:
                print("Years of experience cannot be negative. Please try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a valid number for years of experience.")
            continue

        years_exp = np.array([[years_exp]])  # Reshape for prediction
        predictions = regressor.predict(years_exp)
        # format decimal to 2 places
        predictions = np.round(predictions, 2)
        print("Predicted salary: ${}".format(predictions[0]))
