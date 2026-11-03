"""Streamlit application for the ML-011 Penguin Species Classification project."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "penguins.csv"
MODEL_PATH = ROOT / "models" / "penguin_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
ENCODER_PATH = ROOT / "models" / "encoder.pkl"

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
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]

SPECIES_INFO = {
    "Adelie": {
        "islands": "Torgersen, Dream & Biscoe",
        "description": (
            "A compact, adaptable penguin found across the archipelago. "
            "Adelies tend to have shorter flippers and lighter builds."
        ),
        "accent": "#55a8ce",
        "bill": "#e9a23b",
    },
    "Chinstrap": {
        "islands": "Dream Island",
        "description": (
            "Named for the narrow dark band beneath its bill. Chinstraps "
            "have long bills and a distinctive profile."
        ),
        "accent": "#8f9db8",
        "bill": "#eea14d",
    },
    "Gentoo": {
        "islands": "Biscoe Island",
        "description": (
            "The largest and fastest-swimming of these three species. "
            "Gentoos stand out for their longer flippers and heavier build."
        ),
        "accent": "#50ad9d",
        "bill": "#ed8e49",
    },
}


st.set_page_config(
    page_title="Penguin Species Classification",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --blue: #176b9b;
        --navy: #17324b;
        --sky: #eaf3f9;
        --teal: #117c78;
        --muted: #637b8d;
        --line: #dce6ee;
      }
      html, body, [class*="css"] { font-family: "Segoe UI", Arial, sans-serif; }
      .stApp { background: #f3f7fb; color: var(--navy); }
      [data-testid="stHeader"] { background: rgb(243 247 251 / 88%); }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123650 0%, #174e6c 100%);
        border-right: 1px solid #123650;
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #e4eef5;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
        border-color: rgb(255 255 255 / 18%);
      }
      h1, h2, h3 { font-family: "Segoe UI", Arial, sans-serif !important; color: var(--navy); letter-spacing: -.035em; }
      .brand { color: #d2e8f4; font-family: "Segoe UI", Arial, sans-serif; font-size: 13px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
      .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid #1d5e80;
        border-radius: 22px;
        padding: 42px 44px;
        background: linear-gradient(118deg, #123650 0%, #176b9b 72%, #18817f 120%);
        box-shadow: 0 18px 42px rgb(23 70 98 / 16%);
      }
      .hero::after {
        position: absolute;
        right: -52px;
        bottom: -132px;
        width: 280px;
        height: 280px;
        border: 1px solid rgb(255 255 255 / 17%);
        border-radius: 50%;
        box-shadow: 0 0 0 34px rgb(255 255 255 / 5%), 0 0 0 68px rgb(255 255 255 / 4%);
        content: "";
        pointer-events: none;
      }
      .eyebrow { color: #a9e3dc; font-size: 11px; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
      .hero-title { position: relative; z-index: 1; margin: 10px 0; color: #fff; font-family: "Segoe UI",Arial,sans-serif; font-size: clamp(32px,4.2vw,55px); font-weight: 800; letter-spacing: -.055em; line-height: 1.08; }
      .hero-copy { position: relative; z-index: 1; max-width: 700px; color: #d4e4ee; font-size: 15px; line-height: 1.8; }
      .section-label { margin: 26px 0 8px; color: var(--blue); font-size: 11px; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
      .species-card { min-height: 365px; overflow: hidden; border: 1px solid var(--line); border-radius: 17px; background: #fff; box-shadow: 0 10px 28px rgb(23 70 98 / 7%); transition: transform .18s ease, box-shadow .18s ease; }
      .species-card:hover { transform: translateY(-3px); box-shadow: 0 16px 34px rgb(23 70 98 / 12%); }
      .species-card-copy { padding: 5px 20px 19px; }
      .species-card-copy h3 { margin: 4px 0 6px; font-size: 21px; }
      .species-card-copy p { margin: 0; color: #648096; font-size: 12px; line-height: 1.7; }
      .species-card-meta { margin-top: 12px; color: var(--teal); font-size: 10px; font-weight: 700; letter-spacing: .04em; }
      .penguin-illustration { display: grid; height: 165px; place-items: center; margin: 13px 13px 0; overflow: hidden; border-radius: 12px; background: linear-gradient(145deg,#e9f4f8 0%,#dcecf2 58%,#d9efea 100%); }
      .result-card { padding: 22px; border: 1px solid #d9e8f1; border-radius: 16px; background: linear-gradient(135deg,#fff,#f3f9fc); box-shadow: 0 10px 30px rgb(27 81 113 / 8%); }
      .metric-card { min-height: 102px; padding: 17px 18px; border: 1px solid var(--line); border-top: 3px solid #4196b7; border-radius: 4px 4px 13px 13px; background: #fff; box-shadow: 0 7px 20px rgb(23 70 98 / 5%); }
      .metric-label { color: #658095; font-size: 11px; font-weight: 600; }
      .metric-value { margin-top: 5px; color: var(--blue); font-family: "Segoe UI",Arial,sans-serif; font-size: clamp(19px,1.8vw,26px); font-weight: 800; white-space: nowrap; }
      .subtle { color: var(--muted); font-size: 12px; line-height: 1.75; }
      .info-box { padding: 14px 17px; border-left: 3px solid var(--teal); border-radius: 5px; color: #3e6172; background: #e8f4f2; font-size: 12px; line-height: 1.7; }
      .footer { margin-top: 45px; padding: 17px 0 5px; border-top: 1px solid var(--line); color: #8297a5; font-size: 10px; }
      div.stButton > button[kind="primary"] { border-radius: 9px; color: #fff; background: linear-gradient(105deg,#176b9b,#117c78); border-color: #176b9b; box-shadow: 0 5px 14px rgb(23 107 155 / 18%); }
      div.stButton > button[kind="primary"]:hover { border-color: #117c78; background: #117c78; }
      div[data-testid="stForm"] { padding: 20px; border: 1px solid var(--line); border-radius: 15px; background: white; box-shadow: 0 8px 24px rgb(23 70 98 / 5%); }
      div[data-baseweb="tab-list"] { gap: 8px; }
      button[data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
      [data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg,#176b9b,#117c78); }
      @media (max-width: 700px) {
        .hero { padding: 25px 22px; border-radius: 16px; }
        .hero-copy { font-size: 13px; }
        .species-card { min-height: unset; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load the project model, the notebook scaler, and its fitted encoder."""
    missing = [
        path.name
        for path in (MODEL_PATH, SCALER_PATH, ENCODER_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s): {', '.join(missing)}. "
            "From the project folder, run `python train_model.py`."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    if list(scaler.feature_names_in_) != MODEL_FEATURES:
        raise ValueError(
            "The saved scaler feature order does not match the app's "
            "prediction inputs."
        )
    if list(encoder.get_feature_names_out(CATEGORICAL_FEATURES)) != (
        ENCODED_FEATURES
    ):
        raise ValueError(
            "The saved encoder output does not match the trained model."
        )
    return model, scaler, encoder


def build_features(raw_features, encoder):
    """Apply the notebook's one-hot feature encoding in the trained order."""
    encoded_values = encoder.transform(
        raw_features[CATEGORICAL_FEATURES]
    )
    encoded = pd.DataFrame(
        encoded_values,
        columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES),
        index=raw_features.index,
    )
    features = pd.concat(
        [raw_features[NUMERIC_FEATURES], encoded], axis=1
    ).loc[:, MODEL_FEATURES]
    return features


@st.cache_data(show_spinner=False)
def evaluate_models():
    """Recreate the notebook's held-out evaluation and return display data."""
    data = pd.read_csv(DATA_PATH).dropna().drop_duplicates()
    raw_features = data.drop(columns=["species"])
    target = data["species"]
    notebook_features = pd.get_dummies(
        raw_features, drop_first=True, dtype=int
    )
    x_train, x_test, y_train, y_test = train_test_split(
        notebook_features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model, scaler, encoder = load_artifacts()
    features = build_features(raw_features, encoder)
    x_train = features.loc[x_train.index]
    x_test = features.loc[x_test.index]
    predictions = model.predict(scaler.transform(x_test))

    decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    decision_tree.fit(x_train, y_train)
    tree_predictions = decision_tree.predict(x_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    knn_predictions = knn.predict(x_test)

    report = classification_report(
        y_test,
        predictions,
        labels=SPECIES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": report,
        "confusion": confusion_matrix(y_test, predictions, labels=SPECIES),
        "classes": SPECIES,
        "test_count": len(y_test),
        "accuracies": {
            "Scaled KNN": accuracy_score(y_test, predictions),
            "Decision Tree": accuracy_score(y_test, tree_predictions),
            "KNN": accuracy_score(y_test, knn_predictions),
        },
    }


def penguin_illustration(species):
    """Return a self-contained SVG illustration for a species card."""
    info = SPECIES_INFO[species]
    illustration = f"""
    <svg viewBox="0 0 260 170" width="260" height="170" role="img"
         aria-label="{species} penguin illustration"
         xmlns="http://www.w3.org/2000/svg">
      <circle cx="130" cy="82" r="67" fill="#d6eaf4"/>
      <path d="M37 139c30-13 57-10 87 0s58 10 99-2v33H37z" fill="#b9dce9"/>
      <path d="M97 125c-10-25-10-61 0-84 7-16 20-25 34-25 19 0 34 14 40 37 6 25 4 54-7 75l17 14h-36l-11-16h-19l-10 16H72z" fill="#193b55"/>
      <path d="M121 35c10-9 24-7 32 4 10 13 13 34 9 54-4 18-12 30-25 31-15 1-25-13-29-33-4-22 0-46 13-56z" fill="#fbfdff"/>
      <path d="M124 36c5-12 20-17 30-8 9 7 11 19 8 28-4 10-12 14-23 13-10-1-17-7-18-17z" fill="#193b55"/>
      <circle cx="151" cy="49" r="2.6" fill="#fff"/>
      <path d="m160 56 24 6-22 8z" fill="{info['bill']}"/>
      <path d="m101 76-24 43 23-5 17-28M161 92l28 36-25-5-17-24" fill="#193b55"/>
      <path d="m106 142-2 7h27l2-7m22 0 2 7h27l-7-8" fill="{info['bill']}"/>
      <path d="M133 26h8" stroke="{info['accent']}" stroke-width="6" stroke-linecap="round"/>
      <circle cx="57" cy="53" r="4" fill="{info['accent']}" opacity=".75"/>
      <circle cx="205" cy="83" r="3" fill="{info['accent']}" opacity=".8"/>
    </svg>
    """
    return "".join(line.strip() for line in illustration.splitlines())


def render_species_cards():
    """Render the three project species as responsive Streamlit cards."""
    columns = st.columns(3)
    for column, species in zip(columns, SPECIES):
        info = SPECIES_INFO[species]
        with column:
            card = (
                '<div class="species-card"><div class="penguin-illustration">'
                f"{penguin_illustration(species)}"
                '</div><div class="species-card-copy">'
                f"<h3>{species}</h3>"
                f"<p>{info['description']}</p>"
                '<div class="species-card-meta">'
                f"ISLANDS · {info['islands'].upper()}"
                "</div></div></div>"
            )
            st.markdown(
                card,
                unsafe_allow_html=True,
            )


def render_home(metrics):
    """Show the project introduction, species, and model highlights."""
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">ML-011 &nbsp; · &nbsp; Wildlife classification</div>
          <div class="hero-title">A field guide to<br>penguin classification.</div>
          <div class="hero-copy">
            Explore the Palmer Penguins dataset, compare three Antarctic
            species, and see how a scaled machine-learning model identifies
            patterns in their measurements.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">The study at a glance</div>', unsafe_allow_html=True)
    metric_columns = st.columns(4)
    metric_values = [
        ("333", "clean records"),
        ("3", "penguin species"),
        (f"{metrics['accuracy']:.1%}", "test accuracy · scaled KNN"),
        (str(metrics["test_count"]), "held-out test records"),
    ]
    for column, (value, label) in zip(metric_columns, metric_values):
        with column:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">Meet the species</div>', unsafe_allow_html=True)
    render_species_cards()

    st.markdown('<div class="section-label">What the model learned</div>', unsafe_allow_html=True)
    distribution_tab, comparison_tab, correlation_tab = st.tabs(
        ["Species distribution", "Model comparison", "Feature relationships"]
    )
    with distribution_tab:
        st.image(
            str(ROOT / "images" / "species_distribution.png"),
            caption="Species counts in the cleaned Palmer Penguins dataset.",
            width="stretch",
        )
    with comparison_tab:
        accuracy_frame = pd.DataFrame(
            list(metrics["accuracies"].items()),
            columns=["Model", "Test accuracy"],
        )
        st.bar_chart(
            accuracy_frame,
            x="Model",
            y="Test accuracy",
            x_label="Accuracy",
            y_label="Model",
            horizontal=True,
            height=300,
        )
        st.markdown(
            '<div class="subtle">The notebook uses a stratified 80/20 split '
            'with random state 42. Evaluation scores describe that held-out '
            'sample; they are not a guarantee of future performance.</div>',
            unsafe_allow_html=True,
        )
    with correlation_tab:
        st.image(
            str(ROOT / "images" / "correlation_heatmap.png"),
            caption="Correlation between numeric measurements in the source analysis.",
            width="stretch",
        )


def render_prediction(model, scaler, encoder, metrics):
    """Render and handle the species prediction form."""
    st.markdown('<div class="section-label">Interactive classifier</div>', unsafe_allow_html=True)
    st.title("Predict a penguin species")
    st.markdown(
        '<p class="subtle">Enter the bird\'s measurements and field details. '
        'The form uses the notebook\'s trained encoder, feature order, '
        'scaler and KNN classifier.</p>',
        unsafe_allow_html=True,
    )

    with st.form("penguin_prediction"):
        st.markdown("#### Physical measurements")
        first_row = st.columns(2)
        with first_row[0]:
            bill_length = st.number_input(
                "Bill length (mm)",
                min_value=30.0,
                max_value=65.0,
                value=45.0,
                step=0.1,
                help="Observed dataset range: 32.1–59.6 mm.",
            )
        with first_row[1]:
            bill_depth = st.number_input(
                "Bill depth (mm)",
                min_value=10.0,
                max_value=25.0,
                value=17.0,
                step=0.1,
                help="Observed dataset range: 13.1–21.5 mm.",
            )
        second_row = st.columns(2)
        with second_row[0]:
            flipper_length = st.number_input(
                "Flipper length (mm)",
                min_value=150.0,
                max_value=250.0,
                value=200.0,
                step=1.0,
                help="Observed dataset range: 172–231 mm.",
            )
        with second_row[1]:
            body_mass = st.number_input(
                "Body mass (g)",
                min_value=2000,
                max_value=7000,
                value=4000,
                step=50,
                help="Observed dataset range: 2,700–6,300 g.",
            )

        st.markdown("#### Field details")
        third_row = st.columns(2)
        with third_row[0]:
            island = st.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])
        with third_row[1]:
            sex = st.selectbox("Sex", ["female", "male"])

        st.markdown(
            '<div class="info-box">The trained model also uses observation '
            'year. Since year is not one of the requested form inputs, this '
            'app fixes it to 2008, the middle year in the dataset.</div>',
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button(
            "Classify penguin",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return

    raw_input = pd.DataFrame(
        [
            {
                "bill_length_mm": bill_length,
                "bill_depth_mm": bill_depth,
                "flipper_length_mm": flipper_length,
                "body_mass_g": body_mass,
                "year": 2008,
                "island": island,
                "sex": sex,
            }
        ]
    )
    model_input = build_features(raw_input, encoder)
    scaled_input = scaler.transform(model_input)
    prediction = str(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]
    confidence = float(probabilities[list(model.classes_).index(prediction)])

    st.markdown("---")
    st.markdown('<div class="section-label">Classification result</div>', unsafe_allow_html=True)
    result_columns = st.columns([0.75, 1.25])
    with result_columns[0]:
        st.markdown(
            f'<div class="penguin-illustration">{penguin_illustration(prediction)}</div>',
            unsafe_allow_html=True,
        )
    with result_columns[1]:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="eyebrow">Predicted species</div>
              <h2 style="margin:8px 0 4px">{prediction}</h2>
              <div class="subtle">{SPECIES_INFO[prediction]['description']}</div>
              <div style="margin-top:17px" class="metric-label">
                Top-class model probability · {confidence:.1%}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(confidence)
        st.caption(
            "The displayed probability is the classifier's estimate, not a "
            "calibrated measure of certainty."
        )
    st.caption(
        f"Model: scaled KNN · held-out accuracy: {metrics['accuracy']:.1%} "
        f"on {metrics['test_count']} test records."
    )


def render_about(metrics):
    """Show project background, evaluation, and original analysis assets."""
    st.markdown('<div class="section-label">Project overview</div>', unsafe_allow_html=True)
    st.title("About this project")
    st.markdown(
        """
        <p class="subtle">Penguin Species Classification is an educational
        machine-learning project built around the Palmer Penguins dataset.
        It explores how physical measurements and field observations can be
        used to distinguish Adelie, Chinstrap and Gentoo penguins.</p>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("### Dataset & preparation")
        st.markdown(
            """
            - **Source:** Palmer Penguins dataset
            - **Records:** 344 observations before cleaning; 333 after
            - **Inputs:** bill measurements, flipper length, body mass,
              observation year, island and sex
            - **Preparation:** rows with missing values are removed,
              categorical fields are one-hot encoded, and the same
              stratified 80/20 split from the notebook is retained
            """
        )
    with right:
        st.markdown("### Models & evaluation")
        st.markdown(
            """
            - Decision Tree (maximum depth 5)
            - K-Nearest Neighbors (5 neighbors)
            - Scaled K-Nearest Neighbors (5 neighbors)
            - Accuracy, precision, recall, F1-score and confusion matrix
            """
        )
        st.markdown(
            f'<div class="info-box">The selected scaled KNN scored '
            f'<strong>{metrics["accuracy"]:.1%}</strong> accuracy on the '
            f'{metrics["test_count"]}-record held-out split.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Test-set classification report")
    report = metrics["report"]
    report_frame = pd.DataFrame(
        {
            species: {
                "Precision": report[species]["precision"],
                "Recall": report[species]["recall"],
                "F1-score": report[species]["f1-score"],
                "Support": int(report[species]["support"]),
            }
            for species in SPECIES
        }
    ).T
    st.dataframe(
        report_frame.style.format(
            {
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-score": "{:.3f}",
                "Support": "{:.0f}",
            }
        ),
        width="stretch",
    )

    matrix = pd.DataFrame(
        metrics["confusion"],
        index=[f"Actual {species}" for species in SPECIES],
        columns=[f"Predicted {species}" for species in SPECIES],
    )
    st.markdown("### Confusion matrix")
    st.dataframe(matrix, width="stretch")
    st.image(
        str(ROOT / "images" / "correlation_heatmap.png"),
        caption="Correlation between numeric measurements in the source analysis.",
        width="stretch",
    )
    st.markdown(
        '<div class="info-box">For educational use only. Model predictions '
        'are based on this dataset and should not be treated as professional '
        'biological identification.</div>',
        unsafe_allow_html=True,
    )


def main():
    model, scaler, encoder = load_artifacts()
    metrics = evaluate_models()

    with st.sidebar:
        st.markdown('<div class="brand">🐧 &nbsp; ML-011 FIELD STUDY</div>', unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio(
            "Explore",
            ["Home", "Predict", "About"],
            label_visibility="visible",
        )
        st.markdown("---")
        st.markdown(
            '<div class="subtle">Palmer Penguins<br>Classification study</div>',
            unsafe_allow_html=True,
        )

    if page == "Home":
        render_home(metrics)
    elif page == "Predict":
        render_prediction(model, scaler, encoder, metrics)
    else:
        render_about(metrics)

    st.markdown(
        '<div class="footer">ML-011 · Penguin Species Classification '
        '&nbsp; · &nbsp; Educational machine-learning project</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
