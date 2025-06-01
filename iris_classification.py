import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the CSV from the “data” directory
#    Adjust the path if needed (e.g., if your script lives in a different folder)
df = pd.read_csv("data/iris1.csv")

# 2. Inspect the first few rows (optional)
# print(df.head())

# 3. Separate features (X) and target (y)
#    Assuming the CSV has the standard Iris columns in this order:
#      sepal_length, sepal_width, petal_length, petal_width, species
X = df.iloc[:, :-1]   # all columns except the last one
y = df.iloc[:, -1]    # only the last column (“species”)

# 4. Encode the target labels as integers (Iris-setosa → 0, Iris-versicolor → 1, Iris-virginica → 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Split into training and test sets (80% train, 20% test, stratified by class)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_encoded, 
    test_size=0.20, 
    random_state=42, 
    stratify=y_encoded
)

# 6. Initialize and train a Random Forest classifier
clf = RandomForestClassifier(
    n_estimators=100,      # number of trees
    random_state=42,       # for reproducibility
    n_jobs=-1              # use all CPU cores
)
clf.fit(X_train, y_train)

# 7. Make predictions on the test set
y_pred = clf.predict(X_test)

# 8. Evaluate performance
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}\n")

# 9. Detailed classification report (precision, recall, f1-score per class)
print("Classification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=le.classes_
))
