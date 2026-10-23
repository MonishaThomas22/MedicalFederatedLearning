
from pathlib import Path

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
GLOBAL_MODEL_PATH = MODEL_DIR / "federated_model.pth"

DEVICE = torch.device("cpu")
BATCH_SIZE = 16

st.set_page_config(
    page_title="MedFed AI | Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. CLEAN WHITE HEALTHCARE THEME
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F5F9FC;
        color: #193247;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2EBF1;
    }

    [data-testid="stSidebar"] * {
        color: #23445A;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(120deg, #FFFFFF 0%, #E8F7F5 100%);
        border: 1px solid #D9ECEA;
        border-radius: 20px;
        padding: 28px 30px;
        margin-bottom: 24px;
    }

    .hero-label {
        color: #087F8C;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #16364B;
        font-size: 31px;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #607A8B;
        font-size: 14px;
        line-height: 1.8;
    }

    .section-title {
        color: #193B50;
        font-size: 21px;
        font-weight: 750;
        margin-top: 16px;
        margin-bottom: 14px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E1EBF0;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 4px 14px rgba(25, 59, 80, 0.035);
    }

    .metric-label {
        color: #6A8291;
        font-size: 12px;
        font-weight: 650;
        margin-bottom: 11px;
    }

    .metric-value {
        color: #193B50;
        font-size: 29px;
        font-weight: 800;
    }

    .metric-note {
        color: #07877F;
        font-size: 12px;
        margin-top: 7px;
    }

    .hospital-card {
        background: #FFFFFF;
        border: 1px solid #DFEAF0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 10px;
    }

    .hospital-name {
        color: #1B3C51;
        font-size: 17px;
        font-weight: 750;
    }

    .hospital-detail {
        color: #698291;
        font-size: 13px;
        margin-top: 8px;
        line-height: 1.8;
    }

    .status-pill {
        display: inline-block;
        background: #E5F6EF;
        color: #117653;
        padding: 5px 10px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 750;
    }

    .info-panel {
        background: #FFFFFF;
        border: 1px solid #E0EAF0;
        border-radius: 15px;
        padding: 19px;
        margin-bottom: 12px;
    }

    .info-panel h4 {
        color: #193B50;
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 16px;
    }

    .info-panel p {
        color: #627C8B;
        font-size: 13px;
        line-height: 1.8;
        margin-bottom: 0;
    }

    div[data-testid="stButton"] > button {
        background: #087F8C;
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 43px;
        font-weight: 650;
    }

    div[data-testid="stButton"] > button:hover {
        background: #066874;
        color: white;
        border: none;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E1EBF0;
        border-radius: 14px;
        padding: 15px;
    }

    hr {
        border-color: #E0EAF0;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

@st.cache_data
def get_dataset_stats():
    results = {}

    for split in ["train", "val", "test"]:
        split_path = DATA_DIR / split

        if not split_path.exists():
            results[split] = {}
            continue

        split_data = datasets.ImageFolder(split_path)

        results[split] = {
            class_name: len(
                [
                    path
                    for path, label in split_data.samples
                    if split_data.classes[label] == class_name
                ]
            )
            for class_name in split_data.classes
        }

    return results


def load_checkpoint(path):
    if not path.exists():
        return None

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def get_checkpoint_info():
    checkpoint = load_checkpoint(GLOBAL_MODEL_PATH)

    if checkpoint is None:
        return None

    if not isinstance(checkpoint, dict):
        return {
            "class_names": ["Normal", "Tuberculosis"],
            "num_clients": 3,
            "communication_rounds": 0,
            "aggregation": "Unknown",
            "client_sample_counts": [],
        }

    return checkpoint


def create_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )
    return model


@st.cache_data
def evaluate_global_model():
    if not GLOBAL_MODEL_PATH.exists():
        return {"error": "Global model checkpoint was not found."}

    test_path = DATA_DIR / "test"

    if not test_path.exists():
        return {"error": "Test dataset folder was not found."}

    checkpoint = load_checkpoint(GLOBAL_MODEL_PATH)

    if not isinstance(checkpoint, dict):
        return {
            "error": "Checkpoint format is not supported."
        }

    if "model_state_dict" not in checkpoint:
        return {
            "error": "The checkpoint does not contain model_state_dict."
        }

    class_names = checkpoint.get(
        "class_names",
        ["Normal", "Tuberculosis"],
    )

    if isinstance(class_names, dict):
        class_names = [
            class_names[index]
            for index in sorted(class_names)
        ]

    num_classes = checkpoint.get(
        "num_classes",
        len(class_names),
    )

    model = create_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(
        test_path,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1)

            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

    labels = list(range(len(test_dataset.classes)))

    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "f1": f1_score(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "confusion": confusion_matrix(
            actual,
            predicted,
            labels=labels,
        ),
        "report": classification_report(
            actual,
            predicted,
            labels=labels,
            target_names=test_dataset.classes,
            output_dict=True,
            zero_division=0,
        ),
        "classes": test_dataset.classes,
        "test_count": len(test_dataset),
    }


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 4px 20px 4px;">
            <div style="font-size: 29px;">🏥</div>
            <div style="font-size: 21px; font-weight: 800;
                        color: #193B50;">MedFed AI</div>
            <div style="font-size: 12px; color: #718796;
                        margin-top: 4px;">
                Federated Medical Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Hospital Clients",
            "Dataset Statistics",
            "Model Performance",
            "Federated Learning",
            "Privacy & Security",
        ],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("**PROJECT STATUS**")
    st.caption("Framework: PyTorch + Streamlit")
    st.caption("Architecture: ResNet18")
    st.caption("Aggregation: FedAvg")
    st.caption("Environment: Local CPU")


# =========================================================
# 5. LOAD CURRENT PROJECT INFORMATION
# =========================================================

dataset_stats = get_dataset_stats()
checkpoint_info = get_checkpoint_info()

train_stats = dataset_stats.get("train", {})
val_stats = dataset_stats.get("val", {})
test_stats = dataset_stats.get("test", {})

train_total = sum(train_stats.values())
val_total = sum(val_stats.values())
test_total = sum(test_stats.values())

model_exists = GLOBAL_MODEL_PATH.exists()


# =========================================================
# 6. PAGE HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">
            MEDICAL AI • FEDERATED LEARNING
        </div>
        <div class="hero-title">
            Healthcare Intelligence Dashboard
        </div>
        <div class="hero-subtitle">
            Monitor simulated hospital clients, inspect the dataset,
            evaluate the global model, and explore federated learning
            rounds from one workspace.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 7. OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TRAINING IMAGES",
            f"{train_total:,}",
            "Local training dataset",
        )

    with c2:
        metric_card(
            "HOSPITAL CLIENTS",
            "3",
            "Simulated clients",
        )

    with c3:
        metric_card(
            "TEST IMAGES",
            f"{test_total:,}",
            "Held-out test split",
        )

    with c4:
        metric_card(
            "GLOBAL MODEL",
            "Ready" if model_exists else "Missing",
            "federated_model.pth",
        )

    st.markdown(
        '<div class="section-title">System Architecture</div>',
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns(3)

    with left:
        st.markdown(
            """
            <div class="info-panel">
                <h4>🏥 Hospital Clients</h4>
                <p>Each simulated client trains locally using its
                assigned portion of the training dataset.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with middle:
        st.markdown(
            """
            <div class="info-panel">
                <h4>🔄 FedAvg Server</h4>
                <p>The coordinator combines local model parameters
                into an updated global model over communication rounds.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="info-panel">
                <h4>📊 Model Evaluation</h4>
                <p>The saved global model can be evaluated against
                the held-out test dataset.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Dataset Distribution</div>',
        unsafe_allow_html=True,
    )

    if train_stats:
        chart_df = pd.DataFrame({
            "Class": list(train_stats.keys()),
            "Images": list(train_stats.values()),
        }).set_index("Class")

        st.bar_chart(chart_df, color="#159B91")

        st.caption(
            "Counts are read from the current processed training folders."
        )
    else:
        st.warning("Training dataset was not found in data/processed/train.")


# =========================================================
# 8. HOSPITAL CLIENTS
# =========================================================

elif page == "Hospital Clients":

    st.markdown(
        '<div class="section-title">Participating Hospital Clients</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "These are simulated hospital clients created from one public "
        "dataset. They are not connected to real hospitals."
    )

    sample_counts = []

    if checkpoint_info:
        sample_counts = checkpoint_info.get(
            "client_sample_counts", []
        )

    cols = st.columns(3)

    for index in range(3):
        checkpoint_path = (
            MODEL_DIR / f"hospital_{index + 1}_model.pth"
        )

        with cols[index]:
            exists = checkpoint_path.exists()

            if index < len(sample_counts):
                sample_count = sample_counts[index]
            else:
                sample_count = 702

            st.markdown(
                f"""
                <div class="hospital-card">
                    <div style="font-size: 25px;">🏥</div>
                    <div class="hospital-name">
                        Hospital Client {index + 1}
                    </div>
                    <div style="margin-top: 12px;">
                        <span class="status-pill">
                            {"Checkpoint found" if exists else "Checkpoint missing"}
                        </span>
                    </div>
                    <div class="hospital-detail">
                        Training images: {sample_count}<br>
                        Model: ResNet18<br>
                        Client type: Simulated
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Client Model Files</div>',
        unsafe_allow_html=True,
    )

    for index in range(1, 4):
        path = MODEL_DIR / f"hospital_{index}_model.pth"

        if path.exists():
            st.success(f"Hospital {index}: {path.name} found")
        else:
            st.warning(f"Hospital {index}: {path.name} not found")


# =========================================================
# 9. DATASET STATISTICS
# =========================================================

elif page == "Dataset Statistics":

    st.markdown(
        '<div class="section-title">Dataset Statistics</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("TRAINING", f"{train_total:,}", "Images")

    with c2:
        metric_card("VALIDATION", f"{val_total:,}", "Images")

    with c3:
        metric_card("TEST", f"{test_total:,}", "Images")

    split_df = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Images": [train_total, val_total, test_total],
    }).set_index("Split")

    st.markdown("#### Images per dataset split")
    st.bar_chart(split_df, color="#159B91")

    st.markdown("#### Class distribution")

    all_classes = sorted(
        set(train_stats) | set(val_stats) | set(test_stats)
    )

    if all_classes:
        class_df = pd.DataFrame({
            "Class": all_classes,
            "Train": [train_stats.get(c, 0) for c in all_classes],
            "Validation": [val_stats.get(c, 0) for c in all_classes],
            "Test": [test_stats.get(c, 0) for c in all_classes],
        }).set_index("Class")

        st.dataframe(class_df, use_container_width=True)

        st.bar_chart(class_df, color=["#159B91", "#68B7D0", "#E6B566"])

    st.caption(
        "Dataset counts are read from data/processed. "
        "Verify that the split was performed without patient-level leakage."
    )


# =========================================================
# 10. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Global Model Performance</div>',
        unsafe_allow_html=True,
    )

    if not model_exists:
        st.error(
            "The global checkpoint is missing. Train the model first."
        )

    else:
        st.write(
            "Evaluate the saved global model on the held-out test images."
        )

        if st.button("Run Test Evaluation", use_container_width=True):
            with st.spinner(
                "Evaluating the global model on the test dataset..."
            ):
                evaluate_global_model.clear()
                result = evaluate_global_model()

            st.session_state["model_evaluation_result"] = result

        result = st.session_state.get("model_evaluation_result")

        if result is None:
            st.info(
                "Click **Run Test Evaluation** to calculate current metrics. "
                "This can take some time on a CPU."
            )

        elif "error" in result:
            st.error(result["error"])

        else:
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card(
                    "TEST ACCURACY",
                    f'{result["accuracy"] * 100:.2f}%',
                    "Overall correct predictions",
                )

            with c2:
                metric_card(
                    "MACRO PRECISION",
                    f'{result["precision"] * 100:.2f}%',
                    "Across both classes",
                )

            with c3:
                metric_card(
                    "MACRO RECALL",
                    f'{result["recall"] * 100:.2f}%',
                    "Across both classes",
                )

            with c4:
                metric_card(
                    "MACRO F1-SCORE",
                    f'{result["f1"] * 100:.2f}%',
                    "Precision-recall balance",
                )

            st.markdown("#### Confusion Matrix")

            matrix = pd.DataFrame(
                result["confusion"],
                index=[
                    f"Actual {name}" for name in result["classes"]
                ],
                columns=[
                    f"Predicted {name}" for name in result["classes"]
                ],
            )

            st.dataframe(matrix, use_container_width=True)

            st.markdown("#### Per-class report")

            report_rows = []

            for class_name in result["classes"]:
                values = result["report"][class_name]

                report_rows.append({
                    "Class": class_name,
                    "Precision": values["precision"],
                    "Recall": values["recall"],
                    "F1-score": values["f1-score"],
                    "Support": int(values["support"]),
                })

            report_df = pd.DataFrame(report_rows).set_index("Class")

            st.dataframe(
                report_df.style.format({
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1-score": "{:.3f}",
                }),
                use_container_width=True,
            )

            st.warning(
                "Research demonstration only. These metrics do not establish "
                "clinical safety or diagnostic reliability."
            )


# =========================================================
# 11. FEDERATED LEARNING ROUNDS
# =========================================================

elif page == "Federated Learning":

    st.markdown(
        '<div class="section-title">Federated Learning Monitor</div>',
        unsafe_allow_html=True,
    )

    if checkpoint_info:
        rounds = checkpoint_info.get("communication_rounds", 0)
        aggregation = checkpoint_info.get("aggregation", "Unknown")
        clients = checkpoint_info.get("num_clients", 3)

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("CLIENTS", str(clients), "Participating clients")

        with c2:
            metric_card("ROUNDS", str(rounds), "Saved checkpoint metadata")

        with c3:
            metric_card("METHOD", aggregation, "Aggregation method")

        st.markdown("#### Round workflow")

        if rounds > 0:
            for round_id in range(1, rounds + 1):
                st.markdown(
                    f"""
                    <div class="hospital-card">
                        <span class="status-pill">Round {round_id}</span>
                        <div class="hospital-name" style="margin-top: 10px;">
                            Local training and aggregation
                        </div>
                        <div class="hospital-detail">
                            Each simulated client trains from the shared
                            global parameters, then the server aggregates
                            the client model updates.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "This checkpoint does not contain communication-round "
                "metadata. Run src/fedavg_training.py to create a "
                "round-based FedAvg checkpoint."
            )

        st.caption(
            "Round cards show checkpoint metadata and the configured workflow. "
            "Per-round accuracy and loss are not stored in the current checkpoint."
        )

    else:
        st.warning(
            "Global model checkpoint not found. Run federated training first."
        )


# =========================================================
# 12. PRIVACY AND SECURITY
# =========================================================

elif page == "Privacy & Security":

    st.markdown(
        '<div class="section-title">Privacy & Security Overview</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "The current project is a simulated federated learning experiment. "
        "The following status labels describe the code currently implemented."
    )

    security_items = [
        (
            "Local client training",
            "Implemented",
            "Each simulated client trains using its assigned local subset.",
        ),
        (
            "Federated averaging",
            "Implemented",
            "The server averages model parameters from client models.",
        ),
        (
            "Raw image transfer to aggregator",
            "Not part of current workflow",
            "The aggregation script reads model checkpoints, not raw images. "
            "All clients currently run on the same machine.",
        ),
        (
            "Differential privacy",
            "Not implemented",
            "No differential privacy noise or privacy budget is applied.",
        ),
        (
            "Secure aggregation",
            "Not implemented",
            "Client updates are not protected by a secure aggregation protocol.",
        ),
        (
            "Encrypted communication between real hospitals",
            "Not implemented",
            "This local simulation does not establish a secure multi-hospital network.",
        ),
    ]

    for title, status, description in security_items:
        st.markdown(
            f"""
            <div class="hospital-card">
                <div class="hospital-name">{title}</div>
                <div style="margin-top: 9px;">
                    <span class="status-pill"
                          style="background: {
                              '#E5F6EF' if status == 'Implemented'
                              else '#FFF1DA' if status == 'Not implemented'
                              else '#EAF1F7'
                          };
                          color: {
                              '#117653' if status == 'Implemented'
                              else '#9A5B00' if status == 'Not implemented'
                              else '#476577'
                          };">
                        {status}
                    </span>
                </div>
                <div class="hospital-detail">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "Federated learning reduces the need to centralize raw training data, "
        "but model updates can still leak information. Local training alone "
        "does not guarantee privacy."
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding: 10px;
                color:#78909D; font-size:12px;">
        MedFed AI · Federated Medical Image Research Dashboard<br>
        Educational research prototype · Not for clinical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
