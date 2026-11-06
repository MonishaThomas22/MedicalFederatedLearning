
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from datetime import datetime
import shutil
import io
import subprocess
import threading

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_DIR / "data" / "hospital_uploads"
LOG_DIR = PROJECT_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_LOG_PATH = LOG_DIR / "lung_federated_training.log"


# ==========================================
# GLOBAL MODEL CHECKPOINT PATHS
# ==========================================

LUNG_MODEL_PATH = (
    PROJECT_DIR / "models" / "lung_hospital_federated_model.pth"
)

BRAIN_MRI_MODEL_PATH = (
    PROJECT_DIR / "models" / "brain_mri_hospital_federated_model.pth"
)


# Previously measured held-out test accuracies.
# These are not recalculated automatically during training.

MODEL_TEST_ACCURACY = {
    "lung": 92.52,
    "brain_mri": 77.69
}


# ==========================================
# FEDERATED TRAINING STATUS
# ==========================================

training_status = {
    "running": False,
    "completed": False,
    "message": "Federated training has not started.",
    "return_code": None,
    "log_file": None,
    "last_error": None
}

training_lock = threading.Lock()
training_thread = None


# ==========================================
# HOSPITAL CONFIGURATION
# ==========================================

HOSPITALS = {
    1: {
        "name": "Hospital 1",
        "classes": {"Normal", "Pneumonia", "Tuberculosis"}
    },
    2: {
        "name": "Hospital 2",
        "classes": {"glioma", "meningioma", "notumor", "pituitary"}
    },
    3: {
        "name": "Hospital 3",
        "classes": {"0", "1"}
    }
}

MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="MedFed AI Backend",
    description="Federated Learning for Medical Diagnostics",
    version="1.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# BASIC API ENDPOINTS
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to MedFed AI Backend!",
        "status": "Running"
    }


@app.get("/api/health")
def health_check():

    return {
        "status": "healthy",
        "project": "MedFed AI",
        "backend": "FastAPI"
    }


@app.get("/api/hospitals")
def get_hospitals():

    return {
        "hospitals": [
            {
                "id": hospital_id,
                "name": details["name"],
                "status": "Connected"
            }
            for hospital_id, details in HOSPITALS.items()
        ]
    }


@app.get("/api/hospitals/config")
def get_hospital_config():

    return {
        "hospitals": [
            {
                "id": hospital_id,
                "name": details["name"],
                "classes": sorted(details["classes"])
            }
            for hospital_id, details in HOSPITALS.items()
        ]
    }


# ==========================================
# DATASET ZIP VALIDATION
# ==========================================

def validate_dataset_zip(zip_path: Path, hospital_id: int):

    expected_classes = HOSPITALS[hospital_id]["classes"]

    try:

        with ZipFile(zip_path, "r") as archive:

            files = [
                name for name in archive.namelist()
                if not name.endswith("/")
            ]

            if not files:
                raise ValueError("ZIP file is empty.")

            found_classes = set()

            class_counts = {
                class_name: 0
                for class_name in expected_classes
            }

            for name in files:

                normalized = name.replace("\\", "/")
                parts = normalized.split("/")

                if normalized.startswith("/") or ".." in parts:

                    raise ValueError(
                        "Unsafe file path found in ZIP."
                    )

                matching_classes = [
                    part for part in parts
                    if part in expected_classes
                ]

                if matching_classes:

                    class_name = matching_classes[-1]

                    found_classes.add(class_name)

                    if Path(normalized).suffix.lower() in IMAGE_EXTENSIONS:
                        class_counts[class_name] += 1

            missing_classes = expected_classes - found_classes

            if missing_classes:

                raise ValueError(
                    "Missing class folders: "
                    + ", ".join(sorted(missing_classes))
                )

            empty_classes = [
                name for name, count in class_counts.items()
                if count == 0
            ]

            if empty_classes:

                raise ValueError(
                    "No images found for: "
                    + ", ".join(empty_classes)
                )

            return class_counts

    except BadZipFile:

        raise ValueError(
            "Uploaded file is not a valid ZIP archive."
        )


# ==========================================
# HOSPITAL DATASET UPLOAD
# ==========================================

@app.post("/api/hospitals/{hospital_id}/upload")
async def upload_hospital_dataset(
    hospital_id: int,
    file: UploadFile = File(...)
):

    if hospital_id not in HOSPITALS:

        raise HTTPException(
            status_code=404,
            detail="Hospital not found."
        )

    if not file.filename or not file.filename.lower().endswith(".zip"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a ZIP file."
        )

    hospital_dir = UPLOAD_DIR / f"hospital_{hospital_id}"

    hospital_dir.mkdir(parents=True, exist_ok=True)

    zip_path = hospital_dir / "dataset.zip"
    temporary_path = hospital_dir / "dataset_upload.tmp"

    total_size = 0

    try:

        with temporary_path.open("wb") as destination:

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail="ZIP file exceeds the 2 GB upload limit."
                    )

                destination.write(chunk)

        class_counts = validate_dataset_zip(
            temporary_path,
            hospital_id
        )

        temporary_path.replace(zip_path)

        return {
            "message": "Hospital dataset uploaded and validated.",
            "hospital_id": hospital_id,
            "hospital_name": HOSPITALS[hospital_id]["name"],
            "filename": file.filename,
            "size_bytes": total_size,
            "class_counts": class_counts,
            "status": "Uploaded"
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    finally:

        await file.close()

        if temporary_path.exists():
            temporary_path.unlink()


# ==========================================
# UPLOAD STATUS
# ==========================================

@app.get("/api/uploads/status")
def get_upload_status():

    results = []

    for hospital_id, details in HOSPITALS.items():

        zip_path = (
            UPLOAD_DIR
            / f"hospital_{hospital_id}"
            / "dataset.zip"
        )

        results.append({
            "hospital_id": hospital_id,
            "hospital_name": details["name"],
            "uploaded": zip_path.exists(),
            "status": "Uploaded" if zip_path.exists() else "Waiting"
        })

    return {
        "uploads": results
    }


# ==========================================
# GLOBAL FEDERATED MODEL STATUS
# ==========================================

def get_checkpoint_information(
    model_path,
    model_display_name,
    test_accuracy
):

    if not model_path.exists():

        return {
            "available": False,
            "model_name": model_display_name,
            "checkpoint": model_path.name,
            "message": "Model checkpoint not found."
        }

    try:

        file_info = model_path.stat()

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
        )

        client_counts = checkpoint.get(
            "client_sample_counts",
            []
        )

        return {
            "available": True,
            "model_name": model_display_name,
            "checkpoint": model_path.name,
            "file_size_bytes": file_info.st_size,

            "last_modified": datetime.fromtimestamp(
                file_info.st_mtime
            ).strftime("%d-%m-%Y %H:%M:%S"),

            "architecture": checkpoint.get(
                "architecture",
                "Unknown"
            ),

            "classes": checkpoint.get(
                "class_names",
                []
            ),

            "num_classes": checkpoint.get(
                "num_classes"
            ),

            "num_clients": checkpoint.get(
                "num_clients"
            ),

            "client_sample_counts": client_counts,

            "training_images": sum(client_counts),

            "communication_rounds": checkpoint.get(
                "communication_rounds"
            ),

            "local_epochs": checkpoint.get(
                "local_epochs"
            ),

            "aggregation": checkpoint.get(
                "aggregation"
            ),

            "initialization": checkpoint.get(
                "initialization"
            ),

            "training_mode": checkpoint.get(
                "training_mode"
            ),

            "data_distribution": checkpoint.get(
                "data_distribution"
            ),

            "differential_privacy": checkpoint.get(
                "differential_privacy",
                False
            ),

            "secure_aggregation": checkpoint.get(
                "secure_aggregation",
                False
            ),

            "test_accuracy_percent": test_accuracy,

            "accuracy_source": (
                "Previously measured held-out test evaluation"
            ),

            "disclaimer": (
                "Educational research prototype only. "
                "Not a clinical diagnostic tool."
            )
        }

    except Exception as error:

        return {
            "available": False,
            "model_name": model_display_name,
            "checkpoint": model_path.name,
            "message": f"Could not read checkpoint: {str(error)}"
        }


@app.get("/api/models/status")
def get_global_model_status():

    lung_information = get_checkpoint_information(
        LUNG_MODEL_PATH,
        "Lung X-ray Global Federated Model",
        MODEL_TEST_ACCURACY["lung"]
    )

    brain_information = get_checkpoint_information(
        BRAIN_MRI_MODEL_PATH,
        "Brain MRI Global Federated Model",
        MODEL_TEST_ACCURACY["brain_mri"]
    )

    return {
        "success": True,
        "models": [
            lung_information,
            brain_information
        ]
    }


# ==========================================
# LUNG MODEL
# ==========================================

lung_model = None
lung_class_names = None

lung_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_lung_model(force_reload=False):

    global lung_model, lung_class_names

    if force_reload:

        lung_model = None
        lung_class_names = None

    if lung_model is None:

        print("Loading federated Lung X-ray model...")

        if not LUNG_MODEL_PATH.exists():

            raise RuntimeError(
                "Lung model checkpoint was not found."
            )

        checkpoint = torch.load(
            LUNG_MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

        model = models.resnet18(weights=None)

        model.fc = nn.Linear(
            model.fc.in_features,
            checkpoint["num_classes"]
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        lung_model = model
        lung_class_names = checkpoint["class_names"]

        print("Federated Lung model loaded successfully!")

    return lung_model, lung_class_names


# ==========================================
# FEDERATED TRAINING BACKGROUND FUNCTION
# ==========================================

def run_federated_training():

    global lung_model, lung_class_names

    try:

        training_status["running"] = True
        training_status["completed"] = False
        training_status["message"] = (
            "Federated training is running..."
        )
        training_status["return_code"] = None
        training_status["log_file"] = str(TRAINING_LOG_PATH)
        training_status["last_error"] = None

        python_exe = (
            PROJECT_DIR
            / "backend"
            / ".venv"
            / "Scripts"
            / "python.exe"
        )

        training_script = (
            PROJECT_DIR
            / "src"
            / "fedavg_lung_hospital.py"
        )

        if not python_exe.exists():

            raise FileNotFoundError(
                f"Python environment not found: {python_exe}"
            )

        if not training_script.exists():

            raise FileNotFoundError(
                f"Training script not found: {training_script}"
            )

        checkpoint_before = (
            LUNG_MODEL_PATH.stat().st_mtime_ns
            if LUNG_MODEL_PATH.exists()
            else None
        )

        print("Starting federated training...")
        print("Training log:", TRAINING_LOG_PATH)

        with TRAINING_LOG_PATH.open(
            "w",
            encoding="utf-8"
        ) as log_file:

            log_file.write(
                "Starting three-hospital FedAvg training...\n"
            )

            log_file.flush()

            process = subprocess.Popen(
                [
                    str(python_exe),
                    str(training_script)
                ],
                cwd=str(PROJECT_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )

            return_code = process.wait()

        training_status["return_code"] = return_code

        if return_code != 0:

            log_content = TRAINING_LOG_PATH.read_text(
                encoding="utf-8",
                errors="replace"
            )

            training_status["message"] = (
                "Federated training failed. Check the training log."
            )

            training_status["last_error"] = log_content[-4000:]

            print("Federated training failed.")
            return

        if not LUNG_MODEL_PATH.exists():

            raise RuntimeError(
                "Training finished, but the global model checkpoint "
                "was not found."
            )

        checkpoint_after = LUNG_MODEL_PATH.stat().st_mtime_ns

        if checkpoint_before == checkpoint_after:

            raise RuntimeError(
                "Training finished, but the global model checkpoint "
                "was not updated."
            )

        # Load the newly trained global model.

        load_lung_model(force_reload=True)

        training_status["completed"] = True

        training_status["message"] = (
            "Federated training completed successfully. "
            "Updated global model loaded."
        )

        training_status["return_code"] = 0
        training_status["last_error"] = None

        print("Updated global Lung model loaded successfully!")

    except Exception as error:

        training_status["completed"] = False

        training_status["message"] = (
            f"Training failed: {str(error)}"
        )

        training_status["last_error"] = str(error)

        if training_status["return_code"] is None:
            training_status["return_code"] = -1

        print("Federated training exception:", error)

    finally:

        training_status["running"] = False


# ==========================================
# START FEDERATED TRAINING
# ==========================================

@app.post("/api/training/start")
def start_federated_training():

    global training_thread

    with training_lock:

        if training_status["running"]:

            raise HTTPException(
                status_code=409,
                detail="Federated training is already running."
            )

        training_status["running"] = True
        training_status["completed"] = False
        training_status["message"] = (
            "Starting federated training..."
        )
        training_status["return_code"] = None
        training_status["log_file"] = str(TRAINING_LOG_PATH)
        training_status["last_error"] = None

        try:

            training_thread = threading.Thread(
                target=run_federated_training,
                daemon=False
            )

            training_thread.start()

        except Exception as error:

            training_status["running"] = False
            training_status["message"] = (
                f"Could not start training: {str(error)}"
            )
            training_status["last_error"] = str(error)

            raise HTTPException(
                status_code=500,
                detail=str(error)
            )

    return {
        "success": True,
        "message": "Federated training started successfully."
    }


# ==========================================
# TRAINING STATUS
# ==========================================

@app.get("/api/training/status")
def get_training_status():

    return {
        "running": training_status["running"],
        "completed": training_status["completed"],
        "message": training_status["message"],
        "return_code": training_status["return_code"],
        "log_file": training_status["log_file"],
        "last_error": training_status["last_error"]
    }


# ==========================================
# TRAINING LOG
# ==========================================

@app.get("/api/training/log")
def get_training_log():

    if not TRAINING_LOG_PATH.exists():

        return {
            "available": False,
            "log": "No training log found yet."
        }

    try:

        log_content = TRAINING_LOG_PATH.read_text(
            encoding="utf-8",
            errors="replace"
        )

        return {
            "available": True,
            "log_file": str(TRAINING_LOG_PATH),
            "log": log_content[-12000:]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Could not read training log: {str(error)}"
        )


# ==========================================
# LUNG PREDICTION API
# ==========================================

@app.post("/api/predict/lung")
async def predict_lung_xray(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp"
    }

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG, PNG or BMP image."
        )

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail="Image size must be less than 10 MB."
        )

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid X-ray image."
        )

    model, class_names = load_lung_model()

    input_tensor = lung_transform(
        image
    ).unsqueeze(0)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

    predicted_class = class_names[predicted_index]

    confidence = float(
        probabilities[predicted_index].item() * 100
    )

    return {
        "success": True,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),

        "probabilities": {
            class_names[i]: round(
                float(probabilities[i].item() * 100),
                2
            )
            for i in range(len(class_names))
        },

        "model": "ResNet18 Federated Global Model",

        "disclaimer": (
            "Research prototype only. Not a clinical diagnosis."
        )
    }


# ==========================================
# BRAIN MRI MODEL
# ==========================================

brain_mri_model = None
brain_mri_class_names = None

brain_mri_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_brain_mri_model():

    global brain_mri_model, brain_mri_class_names

    if brain_mri_model is None:

        print("Loading federated Brain MRI model...")

        if not BRAIN_MRI_MODEL_PATH.exists():

            raise HTTPException(
                status_code=500,
                detail="Brain MRI model checkpoint was not found."
            )

        checkpoint = torch.load(
            BRAIN_MRI_MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

        model = models.resnet18(weights=None)

        model.fc = nn.Linear(
            model.fc.in_features,
            len(checkpoint["class_names"])
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        brain_mri_model = model
        brain_mri_class_names = checkpoint["class_names"]

        print("Federated Brain MRI model loaded successfully!")

    return brain_mri_model, brain_mri_class_names


# ==========================================
# BRAIN MRI PREDICTION API
# ==========================================

@app.post("/api/predict/brain-mri")
async def predict_brain_mri(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp"
    }

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG, PNG or BMP MRI image."
        )

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail="Image size must be less than 10 MB."
        )

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid MRI image."
        )

    model, class_names = load_brain_mri_model()

    input_tensor = brain_mri_transform(
        image
    ).unsqueeze(0)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

    predicted_class = class_names[predicted_index]

    confidence = float(
        probabilities[predicted_index].item() * 100
    )

    display_names = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "No Tumor",
        "pituitary": "Pituitary"
    }

    return {
        "success": True,

        "predicted_class": predicted_class,

        "display_name": display_names.get(
            predicted_class,
            predicted_class
        ),

        "confidence": round(confidence, 2),

        "probabilities": {
            class_names[i]: round(
                float(probabilities[i].item() * 100),
                2
            )
            for i in range(len(class_names))
        },

        "model": "ResNet18 Federated Global Model",

        "aggregation": "FedAvg",

        "clients": 3,

        "communication_rounds": 3,

        "disclaimer": (
            "Educational research prototype only. "
            "Not a clinical diagnosis."
        )
    }
