Comparative Analysis of Machine Learning Models in High-Frequency Trading and Sentiment Analysis
Project Overview
This dissertation project presents a comprehensive comparative analysis of various machine learning models applied in the domain of high-frequency trading (HFT) and sentiment analysis. The primary objective is to ascertain the effectiveness and performance of different ML algorithms in predicting financial market movements and analyzing sentiment data from financial news and social media. This analysis is crucial for developing automated trading strategies that can adapt to rapid market changes and leverage sentiment as a predictive indicator.

Repository Structure
The repository is meticulously organized to facilitate easy navigation and comprehension of the project's components. Below is an outline of the directory structure and the purpose of each category of scripts:

Data Acquisition
Market Data & Technical Indicators: Scripts dedicated to fetching high-frequency market data and various technical indicators from the Alpha Vantage API.
Sentiment Data: Scripts for extracting sentiment-related data from financial news and social media platforms.
Data Preparation
Cleaning Scripts: A series of scripts for cleaning and preprocessing the acquired data to ensure it is ready for analysis and modeling.
Sentiment Score Extraction: Utilizes Natural Language Processing (NLP) techniques to evaluate and score sentiment data, preparing it for integration with market data.
Model Implementation
Training Scripts: Separate scripts for training machine learning models on cleaned market data and sentiment scores. Each script is tailored to a specific ML algorithm, including advanced models like BERT for NLP tasks.
Testing Scripts: Corresponding testing scripts for evaluating the trained models' performance, focusing on predictive accuracy and the potential for generating actionable trading signals.
Backtesting
Strategy Simulation: Scripts that simulate trading strategies based on predictions from the trained models. These scripts are critical for assessing the practical applicability of each model in a real-world trading context.
Models Evaluated
The project evaluates an array of machine learning models to cover a broad spectrum of predictive analytics and sentiment analysis:

Convolutional Neural Networks (CNN)
Long Short-Term Memory Networks (LSTM)
Gradient Boosting Machines (GBM)
Random Forests
Support Vector Machines (SVM)
XGBoost
BERT (for sentiment analysis)
Each model is chosen for its unique ability to capture patterns and relationships within the data, with a particular emphasis on BERT's capacity for understanding the nuances of language in sentiment analysis.

Dependencies
The project leverages several Python libraries to process data, perform analyses, and model training. Key dependencies include:

pandas for data manipulation
numpy for numerical operations
scikit-learn for machine learning algorithms
tensorflow/keras for deep learning models
matplotlib for data visualization
nltk and transformers for natural language processing
alpha_vantage for API interactions
