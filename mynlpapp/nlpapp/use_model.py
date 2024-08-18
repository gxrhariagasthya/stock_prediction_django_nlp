"""import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the data
df = pd.read_csv('Data.csv', encoding='ISO-8859-1')

# Data preprocessing
train = df[df['Date'] < '20150101']
test = df[df['Date'] >= '20150101']
data = train.iloc[:, 2:27]
data.replace('[^a-zA-Z]', " ", regex=True, inplace=True)
new_column_names = [str(i) for i in range(25)]
data.columns = new_column_names

for col in new_column_names:
    data[col] = data[col].str.lower()

headlines = [' '.join(str(x) for x in data.iloc[row, 0:25]) for row in range(len(data))]

# Load the CountVectorizer and model
countvector = CountVectorizer(ngram_range=(2, 2))
traindataset = countvector.fit_transform(headlines)
model = joblib.load('my_model_stock_price.pkl')

# Prepare test data
test_data = test.iloc[:, 2:27]
test_data.replace('[^a-zA-Z]', " ", regex=True, inplace=True)
test_data.columns = new_column_names  # Ensure test_data has the same columns

for col in new_column_names:
    test_data[col] = test_data[col].str.lower()

test_transform = [' '.join(str(x) for x in test_data.iloc[row, 0:25]) for row in range(len(test_data))]
test_dataset = countvector.transform(test_transform)

# Predictions and evaluation
predictions = model.predict(test_dataset)
print("Confusion Matrix:")
print(confusion_matrix(test['Label'], predictions))
print("\nAccuracy Score:", accuracy_score(test['Label'], predictions))
print("\nClassification Report:")
print(classification_report(test['Label'], predictions))
"""