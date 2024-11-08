# Insurance Prediction App

This project is a machine learning-based **Insurance Price Prediction** web application that allows users to predict insurance charges based on factors such as BMI, age, gender, number of children, and smoking status. The app is built as a REST API using **Flask** and is enhanced with an interactive user interface built with **HTML, CSS, and JavaScript**. Various regression models, including **Lasso**, **OLS**, **ElasticNet**, and **Ridge**, have been trained on the data, with the best model selected based on thorough analysis. The application is dockerized for easy deployment.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive Interface](#interactive-interface)
- [Model Training](#model-training)
- [Dockerization](#dockerization)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Insurance Prediction app allows users to input several variables, and it uses a trained machine learning model to predict the insurance charges. This prediction is based on historical data that includes variables such as BMI, age, gender, children, and smoking status. The app is designed to serve as both a backend REST API and a user-friendly web interface.

The machine learning model was trained using multiple regression techniques (Lasso, OLS, ElasticNet, and Ridge). After evaluating the models based on performance metrics, the best-performing model was selected and integrated into the Flask API.

## Features

- **REST API**: A backend API built with Flask to handle prediction requests.
- **Machine Learning Models**: Trained using Lasso, OLS, ElasticNet, and Ridge regression methods to predict insurance charges.
- **Interactive Frontend**: A simple, interactive interface where users can enter input values for BMI, age, gender, children, and smoking status.
- **Dockerized App**: Docker is used for easy packaging, deployment, and scaling of the application.
- **Prediction Rendering**: Once the user submits the input values, the application provides an estimated insurance price.

## Technologies Used

- **Python**: The backend is developed using Python.
- **Flask**: Lightweight web framework to build the REST API.
- **HTML, CSS, JavaScript**: Used for the interactive frontend to collect user input and display predictions.
- **Machine Learning Models**: 
  - **Lasso Regression**
  - **OLS (Ordinary Least Squares) Regression**
  - **ElasticNet Regression**
  - **Ridge Regression**
- **Docker**: Containerized the app to simplify deployment.
- **VS Code**: Development environment used for building and testing the application.

## Installation

To run this project locally, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/insurance-prediction.git

