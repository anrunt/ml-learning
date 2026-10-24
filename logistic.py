from typing import cast

from pandas import DataFrame, Series
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Przygotowuje dane
X, y = cast(
    tuple[DataFrame, Series],
    load_iris(return_X_y=True, as_frame=True),
)

print(X.head())
print(y)
print(y.value_counts())

# Dziele dane na treningowe i testowe
X_train, X_test, y_train, y_test = cast(
    tuple[DataFrame, DataFrame, Series, Series],
    train_test_split(X, y, test_size=0.2, random_state=42, stratify=y),
)

# Trenuje logistic regression
l_reg = LogisticRegression()
l_reg.fit(X_train, y_train)

# Generuje predykcje dla danych X_test oraz licze accuracy
y_pred_test = l_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print("Accuracy: ", accuracy)

# Confusion matrix oraz classification report
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
print(classification_report(y_test, y_pred_test))
