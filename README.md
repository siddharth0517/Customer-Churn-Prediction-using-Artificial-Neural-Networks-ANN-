
# Customer Churn Prediction using Artificial Neural Networks (ANN)

This project demonstrates the use of an Artificial Neural Network (ANN) for predicting customer churn based on various features. The model is implemented using TensorFlow and Keras, and the dataset is preprocessed with scikit-learn.

## Dataset
The dataset used in this project contains customer information, including demographics and transaction history, which is used to predict whether or not a customer will churn. The dataset includes:
- Features like customer geography, gender, age, balance, and more.
- The target variable indicates whether the customer has churned.

## Project Workflow

1. **Data Preprocessing**
    - Load the dataset (`data.csv`).
    - Perform label encoding for categorical variables like gender.
    - Apply OneHotEncoding for the country column.
    - Split the dataset into training and testing sets (80%-20%).

2. **Feature Scaling**
    - Feature scaling is applied to normalize the input features.

3. **Building the ANN**
    - A sequential model with input, hidden, and output layers is constructed.
    - Hidden layers use the ReLU activation function.
    - The output layer uses the sigmoid activation function for binary classification.

4. **Training the ANN**
    - The ANN is compiled with the Adam optimizer and binary cross-entropy loss.
    - The model is trained on the training set with 100 epochs.

5. **Model Evaluation**
    - Predictions are made on the test set.
    - A confusion matrix and accuracy score are used to evaluate the performance of the model.

## Libraries Used
- `numpy`
- `pandas`
- `tensorflow`
- `sklearn`

## How to Run
1. Clone this repository.
2. Ensure you have all required libraries installed:
    ```bash
    pip install numpy pandas tensorflow scikit-learn
    ```
3. Run the Python script:
    ```bash
    python ann_for_customer_churn.py
    ```

## Results
- The model's performance is evaluated using a confusion matrix and accuracy score.
- Adjust the architecture or parameters to improve model accuracy and reduce overfitting.

## Future Improvements
- Experiment with different ANN architectures (e.g., more hidden layers, neurons, or regularization techniques).
- Use more advanced techniques for handling imbalanced datasets if applicable.
