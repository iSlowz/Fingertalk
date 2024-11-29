import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = pickle.load(open('dataset.pkl', 'rb'))

X = dataset['X']
Y = dataset['Y']
# split data
X_train, X_test, y_train, y_test = train_test_split(np.array(X), Y, test_size=0.15, random_state=22, shuffle=True)

# model
try:
    model = pickle.load(open('random_forest_model.pkl', 'rb'))
except FileNotFoundError:
    model = RandomForestClassifier(random_state=22)
    model.fit(X_train, y_train)
    # save model
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)


def test_model():
    test_predictions = model.predict(X_test)
    acc = accuracy_score(y_test, test_predictions)
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    test_model()

