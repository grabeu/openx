import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# 1. Load the Covertype Data Set
column_names = [
    'Elevation',
    'Aspect',
    'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area1',
    'Wilderness_Area2',
    'Wilderness_Area3',
    'Wilderness_Area4',
    'Soil_Type1',
    'Soil_Type2',
    'Soil_Type3',
    'Soil_Type4',
    'Soil_Type5',
    'Soil_Type6',
    'Soil_Type7',
    'Soil_Type8',
    'Soil_Type9',
    'Soil_Type10',
    'Soil_Type11',
    'Soil_Type12',
    'Soil_Type13',
    'Soil_Type14',
    'Soil_Type15',
    'Soil_Type16',
    'Soil_Type17',
    'Soil_Type18',
    'Soil_Type19',
    'Soil_Type20',
    'Soil_Type21',
    'Soil_Type22',
    'Soil_Type23',
    'Soil_Type24',
    'Soil_Type25',
    'Soil_Type26',
    'Soil_Type27',
    'Soil_Type28',
    'Soil_Type29',
    'Soil_Type30',
    'Soil_Type31',
    'Soil_Type32',
    'Soil_Type33',
    'Soil_Type34',
    'Soil_Type35',
    'Soil_Type36',
    'Soil_Type37',
    'Soil_Type38',
    'Soil_Type39',
    'Soil_Type40',
    'Cover_Type'
]

data = pd.read_csv('covtype.data', header=None, names=column_names)

# Feature Engineering
data['Euclidean_Distance_To_Hydrology'] = np.sqrt(data['Horizontal_Distance_To_Hydrology']**2 + data['Vertical_Distance_To_Hydrology']**2)
data['Average_Hillshade'] = (data['Hillshade_9am'] + data['Hillshade_Noon'] + data['Hillshade_3pm']) / 3

X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Implement a very simple heuristic that will classify the data

def simple_heuristic(X):
    return np.random.randint(1, 8, size=(len(X),))

# 3. Use Scikit-learn library to train two simple Machine Learning models
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)

# 4. Use TensorFlow library to train a neural network that will classify the data
def build_nn(hidden_layers=1, neurons_per_layer=64, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons_per_layer, input_dim=X_train.shape[1], activation='relu'))
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def find_hyperparameters():
    nn = KerasClassifier(build_nn, epochs=100, batch_size=256, verbose=0)
    param_dist = {
        'hidden_layers': [1, 2, 3],
        'neurons_per_layer': [32, 64, 128],
        'learning_rate': [0.001, 0.01, 0.1],
    }
    search = RandomizedSearchCV(nn, param_distributions=param_dist, n_iter=5, cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_params_

best_hyperparams = find_hyperparameters()
nn = build_nn(**best_hyperparams)
history = nn.fit(X_train, y_train, epochs=100, batch_size=256, validation_split=0.1, verbose=0)

# Plot training curves for the best hyperparameters
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 5. Evaluate your neural network and other models
y_pred_heuristic = simple_heuristic(X_test)
y_pred_tree = tree.predict(X_test)
y_pred_logreg = logreg.predict(X_test)
y_pred_nn = np.argmax(nn.predict(X_test), axis=1)

print("Heuristic accuracy:", accuracy_score(y_test, y_pred_heuristic))
print("Decision Tree accuracy:", accuracy_score(y_test, y_pred_tree))
print("Logistic Regression accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Neural Network accuracy:", accuracy_score(y_test, y_pred_nn))

# 6. Create a very simple REST API that will serve your models
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    model_choice = request.form["model"]
    input_data = np.array(request.form.getlist("input_data"), dtype=float).reshape(1, -1)

    if model_choice == "heuristic":
        prediction = simple_heuristic(input_data)
    elif model_choice == "tree":
        prediction = tree.predict(input_data)
    elif model_choice == "logreg":
        prediction = logreg.predict(input_data)
    elif model_choice == "nn":
        prediction = np.argmax(nn.predict(input_data), axis=1)
    else:
        return jsonify({"error": "Invalid model choice"}), 400

    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
