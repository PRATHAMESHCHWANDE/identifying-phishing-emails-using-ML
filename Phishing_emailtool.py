# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and clean the dataset
def load_and_clean_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Drop irrelevant columns
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    
    # Drop rows with missing values 
    data = data.dropna(subset=['Email Text'])
    
    return data

# Function to train the model
def train_model(data):
    # Preprocessing the data: Vectorizing the email text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(data['Email Text'])

    # Defining the target variable (1: Phishing Email, 0: Safe Email)
    y = data['Email Type'].apply(lambda x: 1 if x == "Phishing Email" else 0)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluating the model on the test set
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Safe Email", "Phishing Email"]))

    # Plotting the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Safe Email", "Phishing Email"], 
                yticklabels=["Safe Email", "Phishing Email"])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return model, vectorizer

# Function to predict if a given email is safe or phishing
def predict_email(model, vectorizer, email_text):
    email_transformed = vectorizer.transform([email_text])
    prediction = model.predict(email_transformed)
    return "Phishing Email" if prediction[0] == 1 else "Safe Email"

# Main script
if __name__ == "__main__":
    # Load and clean the dataset
    file_path = 'C:/Users/Siddhant/Downloads/Phishing_Email.csv'  # This is your file path
    data = load_and_clean_data(file_path)

    # Train the model
    model, vectorizer = train_model(data)

    # Input an email to classify
    email_to_predict = input("Enter the email text to check if it's phishing or safe: ")
    result = predict_email(model, vectorizer, email_to_predict)
    print(f"The email is classified as: {result}")
