"""Rebuild the penguin classifier using the notebook's preprocessing and split."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "penguins.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "penguin_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"

NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "year",
]
CATEGORICAL_FEATURES = ["island", "sex"]
ENCODED_FEATURES = [
    "island_Dream",
    "island_Torgersen",
    "sex_male",
]
MODEL_FEATURES = NUMERIC_FEATURES + ENCODED_FEATURES
ENCODER_CATEGORIES = [
    ["Biscoe", "Dream", "Torgersen"],
    ["female", "male"],
]


def main() -> None:
    """Train and save the scaled KNN model, preserving the notebook's scaler."""
    data = pd.read_csv(DATA_PATH).dropna().drop_duplicates()
    raw_features = data.drop(columns=["species"])
    target = data["species"]

    notebook_features = pd.get_dummies(
        raw_features, drop_first=True, dtype=int
    )
    train_features, test_features, y_train, y_test = train_test_split(
        notebook_features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    encoder = OneHotEncoder(
        categories=ENCODER_CATEGORIES,
        drop="first",
        handle_unknown="error",
        sparse_output=False,
    )
    encoder.fit(
        raw_features.loc[train_features.index, CATEGORICAL_FEATURES]
    )
    encoded_values = encoder.transform(raw_features[CATEGORICAL_FEATURES])
    encoded_features = pd.DataFrame(
        encoded_values,
        columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES),
        index=raw_features.index,
    )
    features = pd.concat(
        [raw_features[NUMERIC_FEATURES], encoded_features], axis=1
    ).loc[:, MODEL_FEATURES]

    if not np.array_equal(
        features.loc[train_features.index].to_numpy(),
        train_features.loc[:, MODEL_FEATURES].to_numpy(),
    ):
        raise ValueError(
            "The saved encoder does not reproduce the notebook's one-hot "
            "features. Check the dataset categories before training."
        )

    x_train = features.loc[train_features.index]
    x_test = features.loc[test_features.index]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        if list(scaler.feature_names_in_) != MODEL_FEATURES:
            raise ValueError(
                "The saved scaler feature order does not match the notebook."
            )
        reference_scaler = StandardScaler().fit(x_train)
        if not (
            np.allclose(scaler.mean_, reference_scaler.mean_)
            and np.allclose(scaler.scale_, reference_scaler.scale_)
        ):
            raise ValueError(
                "The saved scaler does not match the notebook's training "
                "split; it was not overwritten."
            )
    else:
        scaler = StandardScaler().fit(x_train)
        joblib.dump(scaler, SCALER_PATH)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(scaler.transform(x_train), y_train)
    predictions = model.predict(scaler.transform(x_test))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print(f"Training records: {len(x_train)}")
    print(f"Test records: {len(x_test)}")
    print(f"Features: {MODEL_FEATURES}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4%}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved encoder: {ENCODER_PATH}")
    print(f"Reused scaler: {SCALER_PATH}")


if __name__ == "__main__":
    main()
