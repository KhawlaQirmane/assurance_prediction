# Insurance Prediction App

This project is a machine learning-based **Insurance Price Prediction** web application that allows users to predict insurance charges based on factors such as BMI, age, gender, number of children, and smoking status. The app is built as a REST API using **Flask** and is enhanced with an interactive user interface built with **HTML, CSS, and JavaScript**. Various regression models, including **Lasso**, **OLS**, **ElasticNet**, and **Ridge**, have been trained on the data, with the best model selected based on thorough analysis. The application is dockerized for easy deployment.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Interactive Interface](#interactive-interface)
- [Model Training](#model-training)
- [Dockerization](#dockerization)
- [Contributing](#contributing)
- [License](#license)


## Features

- **REST API**: A backend API built with Flask to handle prediction requests.
- **Machine Learning Models**: Trained using Lasso, OLS, ElasticNet, and Ridge regression methods to predict insurance charges.
- **Interactive Frontend**: A simple, interactive interface(A form filled in by the user). where users can enter input values for BMI, age, gender, children, and 
    smoking status and the application provides an estimated insurance price.
- **Dockerized App**: Docker is used for easy packaging, deployment, and scaling of the application.

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

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/insurance-prediction-app.git
    cd insurance-prediction-app
    ```
2.Create avirtual envirement: 
   for Linux/macOS:
   
     ```bash
     source .venv/bin/activate
     ```
   for windows: 
   
     ```bash
    .venv\Scripts\activate
     ```
3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Flask API:

    ```bash
    python app.py
    ```

5. Docker (optional):

    Build and run the Docker container:

    ```bash
    docker build -t insurance-prediction-app .
    docker run -p 5000:5000 insurance-prediction-app
    ```
