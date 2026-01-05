# Titanic Survival Prediction ML

A machine learning project for predicting passenger survival on the Titanic using Random Forest classification with MLflow integration and automated CI/CD pipeline.

## Features

- **Model Training**: Random Forest classifier with optimized hyperparameters
- **MLflow Integration**: Track experiments, log metrics, and manage model versions
- **FastAPI Server**: REST API for making predictions
- **GitHub Actions Pipeline**: Automated train, test, and deploy workflows
- **Docker Support**: Containerized deployment to Docker Hub


## Model

- **Algorithm**: Random Forest Classifier
- **n_estimators**: 200
- **max_depth**: 15
- **class_weight**: balanced
- **min_samples_split**: 2
- **min_samples_leaf**: 1

## CI/CD Pipeline

The GitHub Actions workflow automates three main stages:

1. **Train**: Trains the model on the training dataset
2. **Test**: Validates model performance against test thresholds
3. **Deploy**: Builds and pushes Docker image to Docker Hub

Pipeline triggers on changes to data, model training/testing code, API server code, or container configuration.

## Test Thresholds

The model must meet these performance criteria:
- **Minimum Accuracy**: 75%
- **Minimum F1 Score**: 70%
