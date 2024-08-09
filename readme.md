LSTM Model for Stock Market Prediction
Welcome to the repository for our LSTM model designed to predict stock market trends. This project to see the strengths of LSTM architectures to analyze time-series data and extract features, aiming to outperform traditional methods in stock market prediction.

Table of Contents  
Introduction  
Project Structure  
Installation  
Usage  
Model Architecture  
Data Preparation  
Training and Evaluation  
Results  

Introduction  
This project leverages a LSTM model to predict stock prices. Our goal is to develop a robust model that can effectively predict stock market trends and potentially beat the market. Given the resource out there 

Project Structure
.  
├── data  
│   ├── data_util.py
├── model
│   ├── lstm.py  
├── training and testing
│   ├── train_model.py   
├── notebooks  
├── requirements.txt  
Installation
Clone the repository and install the necessary dependencies.

Copy Repo:  
```
git clone https://github.gatech.edu/asutherland9/Deep-Learning-CS7643.git  
conda env create -f environment.yaml 
```

Report Link:  
https://www.overleaf.com/project/669f0c77533db67300b566b7

Data Link:  
https://gtvault-my.sharepoint.com/my?id=%2Fpersonal%2Fasutherland9%5Fgatech%5Fedu%2FDocuments%2FDeep%20Learning%20CS7643%2FProject


LSTM: Handles sequential data to learn temporal patterns.
CNN: Extracts local features and patterns in the data.
Dense Layers: Combine features learned by LSTM and CNN for final prediction.
Data Preparation

The data preprocessing involves:  
1.Handling missing values.  
2.Normalizing the data.  
3.Creating sequences of data for LSTM input.  
4.Splitting data into training, validation, and test sets.  
5.Training and Evaluation  

The training process involves:  
Configuring hyperparameters through some config file (file type TBD)  
Training the model on the processed data.  
Saving the best model based on validation performance.  

The evaluation process involves:  
Loading the trained model.  
Predicting stock prices on the test set.  
Comparing predicted values with actual values using performance metrics.  
Results
The results of our experiments, including performance metrics and visualizations, are documented in the results directory.
