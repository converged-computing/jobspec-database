from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import swish
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
import sys

if len(sys.argv) > 1:
    DATADIR = Path(sys.argv[1])
else:
    DATADIR = Path("/project/dfreedman/colmt/UChicago-AI-in-Science-Hackathon/stellar-paleontology-data/")

def load_classification_data_sample(datadir=DATADIR, fraction=1.0, metallicity="all"):
    """
    Load the data from the prepared pickle file and optionally sample a fraction of it.

    Parameters:
    - datadir: Path to the directory containing the pickle file.
    - fraction: Fraction of the data to load (default is 1.0, i.e., all data).
    - metallicity: Filter by metallicity if required, currently implemented as "all".

    Returns:
    - DataFrame with the requested fraction of data.

    Notes:
    - `ZAMS` (zero-age main sequence) marks when the stars form. Any quantities defined at this time are inputs to the simulation.
    - The kick refers to momentum that is lost during supernova explosions, technically occurring later in the simulation but chosen based on some prescription.

    We also retain our target, `Merges_Hubble_Time`.
    """
    ignore = ["Mass(1)", "Mass(2)", "Eccentricity@DCO", "SemiMajorAxis@DCO", "Coalescence_Time"]

    # Load the full dataset from a pickle file
    double_compact_objects = pd.read_pickle(datadir / "compas-data.pkl")

    # Remove specified columns from the dataset
    double_compact_objects = double_compact_objects.drop(columns=[key for key in ignore if key in double_compact_objects.columns], errors='ignore')

    # Sample the requested fraction of the data
    if fraction < 1.0:
        double_compact_objects = double_compact_objects.sample(frac=fraction, random_state=42)  # random_state for reproducibility

    return double_compact_objects

def load_classification_data(datadir=DATADIR, metallicity="all"):
    """
    Load the data
    
    We read the data from the prepepared pickle file.
    
    We have retained parameters that have `ZAMS`$^1$ or `Kick`$^2$ in the name along with our classification target.

        1. `ZAMS` (zero-age main sequence) marks when the stars form. Any quantities that are defined at this time are inputs to the simulation.
        2. The kick refers to momentum that is lost during supernova explosions. Technically, these occur later in the simulation, but in practice they are chosen randomly based on some phenomenological prescription.

    We also retain our target, `Merges_Hubble_Time`"
    """
    ignore = ["Mass(1)", "Mass(2)", "Eccentricity@DCO", "SemiMajorAxis@DCO", "Coalescence_Time"]

    double_compact_objects = pd.read_pickle(datadir / "compas-data.pkl")
    for key in ignore:
        if key in double_compact_objects:
            del double_compact_objects[key]
    return double_compact_objects

def statistics(confusion: np.ndarray, verbose=True) -> float:
    """
    Compute summary statistics based on the confusion matrix.
    For more details see https://en.wikipedia.org/wiki/Confusion_matrix.
    As we are interested in both reducing false positives and negatives, quantitative
    evaluation of the model will be based on the balanced accuracy.
    
    Parameters
    ==========
    confusion: np.ndarray
        The two-by-two confusion matrix of predicted class vs actual class
        
    Returns
    =======
    balanced_accuracy: float
        The balanced accuracy of the predictions
    """
    confusion = confusion / confusion.sum(axis=0)
    true_positive_rate = confusion[1, 1]
    true_negative_rate = confusion[0, 0]
    false_positive_rate = confusion[1, 0]
    false_negative_rate = confusion[0, 1]
    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
    if verbose:
        print(f"True positive rate: {true_positive_rate:.4f}")
        print(f"False positive rate: {false_positive_rate:.4f}")
        print(f"True negative rate: {true_negative_rate:.4f}")
        print(f"False negative rate: {false_negative_rate:.4f}")
        print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    return balanced_accuracy


stars = load_classification_data_sample(fraction=1)

X = stars.copy()
y = X.pop('Merges_Hubble_Time')

features_num = [
    "Kick_Magnitude_Random", "Kick_Magnitude_Random(1)",
    "Kick_Magnitude_Random(2)", "Kick_Mean_Anomaly(1)",
    "Kick_Mean_Anomaly(2)", "Kick_Phi(1)", "Kick_Phi(2)", "Kick_Theta(1)",
    "Kick_Theta(2)", "Mass@ZAMS(1)",
    "Mass@ZAMS(2)", "Metallicity@ZAMS(1)",
    "SemiMajorAxis@ZAMS",
]

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    StandardScaler(),
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
)

# stratify - make sure classes are evenlly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.75)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]

model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1,activation = 'sigmoid'),
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5, 
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=1024,
    epochs=150,
    callbacks=[early_stopping],
)
test_results = model.evaluate(X_valid, y_valid)
print(f"Test Loss: {test_results[0]}, Test Accuracy: {test_results[1]}")

y_pred_prob = model.predict(X_valid)

# Convert the probabilities to predicted classes (0 or 1)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(y_valid, y_pred)

# Calculate the balanced accuracy
balanced_acc = statistics(cm)

print("Confusion Matrix:")
print(cm)
print(f"Balanced Accuracy: {balanced_acc:.4f}")

first_layer_weights = model.layers[0].get_weights()[0]

# Print the feature weights
print("Feature Weights:")
for feature, weight in zip(features_num, first_layer_weights):
    print(f"{feature}: {weight:.4f}")

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
'''
def prepare_and_predict(test_data_path):
    test_data = pickle.load(open(test_data_path, 'rb'))
    X_new = test_data.copy()
    
    X_new.pop('Merges_Hubble_Time')
    
    X_new_transformed = preprocessor.transform(X_new)
    
    probabilities = model.predict(X_new_transformed)
    
    predicted_classes = (probabilities > 0.5).astype(int)
    
    labels = ["not merge", "merge"]
    interpreted_results = [labels[pred] for pred in predicted_classes.flatten()]

    return interpreted_results

test_data_path = 'path_to_new_data.pickle'
interpreted_predictions = prepare_and_predict(test_data_path)
print(interpreted_predictions)
'''
