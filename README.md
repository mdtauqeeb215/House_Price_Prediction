# House Price Prediction

A Machine Learning project that predicts house prices based on area, number of bedrooms, bathrooms, and location.

## Project Overview

This project uses Linear Regression to predict the approximate price of a house.

The model is trained using a dataset containing house information from Kolkata and Patna.

## Features Used

- Area (sq.ft)
- Number of Bedrooms
- Number of Bathrooms
- Location

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Machine Learning Model

Linear Regression

## Model Performance

- Mean Absolute Error: ₹2,00,752
- R² Score: 0.9873

## Example Prediction

For a house with:

- Area: 1600 sq.ft
- Bedrooms: 3
- Bathrooms: 2
- Location: Kolkata

Predicted Price:

₹67.33 Lakh

## Project Structure

House_Price_Prediction/

├── main.py  
├── house_data.csv  
└── README.md

## How to Run

1. Install Python.
2. Install required libraries.
3. Open the project folder in VS Code.
4. Run:

```bash
python main.py