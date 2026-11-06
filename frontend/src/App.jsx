import { useEffect, useState } from "react";
import { Routes, Route, Link, useNavigate } from "react-router-dom";

import {
  getHospitals,
  getUploadStatus,
  uploadHospitalDataset,
  predictLungXray,
  predictBrainMRI,
  startFederatedTraining,
  getTrainingStatus,
} from "./api";

import {
  Activity,
  Brain,
  Eye,
  Scan,
  ShieldCheck,
  Network,
  ArrowRight,
  HeartPulse,
  Hospital,
  LockKeyhole,
  Database,
  Microscope,
  LayoutDashboard,
  Home,
  Users,
  Dna,
  Server,
  Upload,
  ArrowLeft,
  Cpu,
  FileImage,
  Workflow,
  ImagePlus,
  Sparkles,
  RotateCcw,
} from "lucide-react";

// =====================================================
// DISEASE MODULES
// =====================================================

const modules = [
  {
    name: "Lung X-ray",
    description: "Normal, Tuberculosis and Pneumonia classification",
    icon: Activity,
    color: "#45dfbd",
  },
  {
    name: "Brain MRI",
    description: "Brain tumor classification",
    icon: Brain,
    color: "#a78bfa",
  },
  {
    name: "Retina Analysis",
    description: "Diabetic retinopathy classification",
    icon: Eye,
    color: "#fbbf24",
  },
  {
    name: "Brain CT",
    description: "Intracranial hemorrhage research",
    icon: Scan,
    color: "#60a5fa",
  },
  {
    name: "Lung CT",
    description: "Lung nodule detection research",
    icon: Microscope,
    color: "#fb923c",
  },
];

// =====================================================
// GLOBAL STYLES
// =====================================================

function GlobalStyles() {
  return (
    <style>{`
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Arial, sans-serif;
        background: #091321;
        color: #f1f5f9;
      }
      button, input { font: inherit; }
      button { cursor: pointer; }
      a { color: inherit; text-decoration: none; }
      .navbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 18px 6%; background: #101d2d;
        border-bottom: 1px solid #243447; gap: 15px; flex-wrap: wrap;
      }
      .brand {
        display: flex; align-items: center; gap: 10px;
        font-size: 21px; font-weight: bold; color: #45dfbd;
      }
      .nav-links {
        display: flex; align-items: center; gap: 25px; flex-wrap: wrap;
      }
      .nav-links a { color: #cbd5e1; font-size: 14px; }
      .nav-links a:hover { color: #45dfbd; }
      .btn, .outline-btn {
        display: inline-flex; align-items: center; justify-content: center;
        gap: 9px; padding: 12px 18px; border-radius: 8px;
        border: none; background: #45dfbd; color: #07131c;
        font-weight: bold; cursor: pointer; transition: 0.2s;
      }
      .btn:hover { background: #65f0d0; }
      .btn:disabled { opacity: 0.6; cursor: not-allowed; }
      .outline-btn {
        background: transparent; border: 1px solid #45dfbd; color: #45dfbd;
      }
      .page { min-height: 100vh; }
      .hero {
        padding: 95px 8%;
        background: radial-gradient(circle at 80% 30%, #123c43, transparent 35%), #091321;
      }
      .hero h1 {
        font-size: clamp(36px, 6vw, 65px); max-width: 800px; line-height: 1.15;
      }
      .hero p {
        color: #9aabba; max-width: 650px; line-height: 1.8; font-size: 17px;
      }
      .eyebrow {
        color: #45dfbd; font-size: 12px; letter-spacing: 2px; font-weight: bold;
      }
      .section { padding: 65px 8%; }
      .section h2 { font-size: 30px; }
      .muted { color: #9aabba; line-height: 1.8; }
      .card-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px; margin-top: 30px;
      }
      .card {
        background: #101d2d; border: 1px solid #243447;
        padding: 25px; border-radius: 13px;
      }
      .card h3 { margin-top: 14px; }
      .card p { color: #9aabba; line-height: 1.7; }
      .tag {
        display: inline-block; padding: 7px 11px; border-radius: 20px;
        background: #173b3b; color: #45dfbd; font-size: 12px;
      }
      .footer {
        padding: 30px; text-align: center; color: #9aabba;
        border-top: 1px solid #243447;
      }
      .dashboard { min-height: 100vh; }
      .dash-layout { display: flex; min-height: calc(100vh - 65px); }
      .sidebar {
        width: 235px; flex-shrink: 0; background: #101d2d;
        padding: 25px 15px; border-right: 1px solid #243447;
      }
      .side-link {
        display: flex; align-items: center; gap: 12px; width: 100%;
        padding: 14px; border: none; border-radius: 8px;
        color: #cbd5e1; background: transparent; text-align: left; margin-bottom: 8px;
      }
      .side-link:hover, .side-link.active { background: #173b3b; color: #45dfbd; }
      .dash-content { flex: 1; min-width: 0; padding: 35px; overflow: hidden; }
      .dash-title {
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 15px;
      }
      .dash-title h1 { margin-bottom: 8px; }
      .dash-title p { color: #9aabba; }
      .dash-stats {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 15px; margin: 30px 0 40px;
      }
      .dash-stat {
        display: flex; flex-direction: column; gap: 10px; padding: 20px;
        background: #101d2d; border: 1px solid #243447; border-radius: 12px;
      }
      .dash-stat svg { color: #45dfbd; }
      .dash-stat strong { font-size: 27px; }
      .dash-stat span { color: #9aabba; font-size: 13px; }
      .dash-section-title {
        margin: 35px 0 20px; font-size: 22px; scroll-margin-top: 20px;
      }
      .hospital-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 20px;
      }
      .hospital-card {
        background: #101d2d; border: 1px solid #243447;
        border-radius: 13px; padding: 23px; min-width: 0;
      }
      .hospital-card h3 { margin-bottom: 5px; }
      .hospital-card p { color: #9aabba; font-size: 14px; line-height: 1.6; }
      .status {
        margin: 15px 0; padding: 9px; border-radius: 7px;
        background: #173b3b; color: #45dfbd; font-size: 13px;
      }
      .upload-label { width: 100%; margin-top: 10px; }
      .upload-message { overflow-wrap: anywhere; font-size: 13px; line-height: 1.6; }
      .dashboard-modules {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 15px;
      }
      .dashboard-module {
        display: flex; gap: 15px; padding: 20px; border-radius: 12px;
        background: #101d2d; border: 1px solid #243447;
      }
      .dashboard-module h3 { margin: 0 0 8px; }
      .dashboard-module p { color: #9aabba; font-size: 13px; line-height: 1.6; }
      .module-icon { flex-shrink: 0; }
      .disclaimer {
        margin-top: 35px; padding: 18px; border: 1px solid #805b2b;
        border-radius: 9px; color: #fbbf24; background: #302619; line-height: 1.6;
      }
      .prediction-section {
        margin-top: 40px; padding: 25px; background: #101d2d;
        border: 1px solid #243447; border-radius: 14px;
      }
      .prediction-heading {
        display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
      }
      .prediction-heading svg { color: #45dfbd; }
      .prediction-description { color: #9aabba; line-height: 1.7; font-size: 14px; }
      .prediction-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 25px; margin-top: 25px;
      }
      .prediction-upload {
        min-height: 230px; padding: 25px; border: 2px dashed #345064;
        border-radius: 12px; display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center; gap: 14px;
      }
      .prediction-upload svg { color: #45dfbd; }
      .prediction-upload p { color: #9aabba; font-size: 13px; line-height: 1.6; }
      .xray-preview {
        width: 100%; max-width: 350px; max-height: 300px;
        object-fit: contain; border-radius: 10px; background: #091321;
      }
      .prediction-result {
        margin-top: 25px; padding: 24px; border: 1px solid #286b5c;
        border-radius: 12px; background: #102a2b;
      }
      .prediction-result h3 { color: #45dfbd; margin-top: 8px; }
      .prediction-confidence {
        font-size: 35px; font-weight: bold; color: #45dfbd; margin: 12px 0;
      }
      .probability-row { margin-top: 16px; }
      .probability-label {
        display: flex; justify-content: space-between; gap: 10px;
        margin-bottom: 7px; font-size: 13px;
      }
      .probability-track {
        width: 100%; height: 9px; background: #243447;
        border-radius: 20px; overflow: hidden;
      }
      .probability-fill {
        height: 100%; background: #45dfbd; border-radius: 20px; transition: width 0.3s;
      }
      .prediction-error {
        margin-top: 15px; padding: 12px; background: #3b2025;
        border: 1px solid #8b4149; border-radius: 8px;
        color: #fca5a5; font-size: 13px; line-height: 1.6;
      }
      .prediction-note {
        margin-top: 20px; padding: 13px; border-radius: 8px;
        background: #302619; color: #fbbf24; font-size: 12px; line-height: 1.6;
      }
      .training-card {
        background: #101d2d; border: 1px solid #243447;
        padding: 25px; border-radius: 13px;
      }
      .training-heading { display: flex; align-items: center; gap: 12px; margin-bottom: 15px; }
      .training-heading svg { color: #45dfbd; flex-shrink: 0; }
      .training-detail {
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 12px; padding: 15px; margin: 18px 0;
        border: 1px solid #243447; border-radius: 9px; background: #091321;
      }
      .training-detail span { color: #9aabba; font-size: 14px; }
      .training-detail strong { color: #45dfbd; font-size: 14px; }
      .training-button { margin-top: 15px; }
      .content-page { max-width: 1100px; margin: auto; padding: 65px 25px; }
      .content-page h1 { font-size: 38px; }
      .content-page p { color: #9aabba; line-height: 1.9; }
      .step {
        display: flex; gap: 18px; padding: 20px; margin: 15px 0;
        background: #101d2d; border: 1px solid #243447; border-radius: 12px;
      }
      .step-number {
        width: 35px; height: 35px; flex-shrink: 0; display: grid;
        place-items: center; border-radius: 50%; background: #45dfbd;
        color: #07131c; font-weight: bold;
      }
      @media (max-width: 750px) {
        .nav-links { gap: 12px; }
        .dash-layout { flex-direction: column; }
        .sidebar { width: 100%; display: flex; overflow-x: auto; gap: 5px; }
        .side-link { width: auto; white-space: nowrap; }
        .dash-content { padding: 20px; }
        .hero { padding: 65px 7%; }
      }
    `}</style>
  );
}

// =====================================================
// SHARED COMPONENTS
// =====================================================

function Brand() {
  return (
    <Link to="/" className="brand">
      <HeartPulse size={28} />
      <span>MedFed AI</span>
    </Link>
  );
}

function Navbar() {
  return (
    <nav className="navbar">
      <Brand />
      <div className="nav-links">
        <Link to="/">Home</Link>
        <Link to="/disease-models">Disease Models</Link>
        <Link to="/how-it-works">How It Works</Link>
        <Link to="/privacy">Privacy</Link>
        <Link to="/dashboard" className="btn">
          Dashboard <ArrowRight size={15} />
        </Link>
      </div>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="footer">
      MedFed AI · Educational Research Prototype · Not for Clinical Use
    </footer>
  );
}

// =====================================================
// HOME PAGE
// =====================================================

function HomePage() {
  return (
    <div className="page">
      <Navbar />
      <section className="hero">
        <div className="eyebrow">PRIVACY-AWARE MEDICAL AI RESEARCH</div>
        <h1>Collaborative Medical AI Without Sharing Raw Patient Images</h1>
        <p>
          MedFed AI explores how multiple hospital clients can collaboratively
          train medical AI models using a federated learning architecture.
        </p>
        <div style={{ display: "flex", gap: 15, flexWrap: "wrap", marginTop: 30 }}>
          <Link to="/dashboard" className="btn">
            Open Dashboard <ArrowRight size={17} />
          </Link>
          <Link to="/how-it-works" className="outline-btn">
            Explore Workflow
          </Link>
        </div>
      </section>

      <section className="section">
        <div className="eyebrow">PLATFORM OVERVIEW</div>
        <h2>What is MedFed AI?</h2>
        <p className="muted">
          A research prototype designed to demonstrate a multi-hospital medical
          AI workflow. The long-term goal is to train models across separate
          hospital clients while keeping raw images at their source.
        </p>
        <div className="card-grid">
          <div className="card">
            <Network size={30} color="#45dfbd" />
            <h3>Federated Learning</h3>
            <p>A central server can aggregate model updates from participating clients.</p>
          </div>
          <div className="card">
            <LockKeyhole size={30} color="#45dfbd" />
            <h3>Privacy Research</h3>
            <p>Explore privacy techniques such as differential privacy and secure aggregation.</p>
          </div>
          <div className="card">
            <Hospital size={30} color="#45dfbd" />
            <h3>Hospital Clients</h3>
            <p>Simulate multiple hospital participants in a medical AI research environment.</p>
          </div>
        </div>
      </section>
      <Footer />
    </div>
  );
}

// =====================================================
// DISEASE MODELS PAGE
// =====================================================

function DiseaseModelsPage() {
  return (
    <div className="page">
      <Navbar />
      <section className="content-page">
        <div className="eyebrow">RESEARCH MODULES</div>
        <h1>Medical AI Disease Models</h1>
        <p>
          The platform interface contains five proposed medical AI research
          modules. Each module requires its own appropriate dataset and training pipeline.
        </p>
        <div className="card-grid">
          {modules.map((module) => {
            const Icon = module.icon;
            return (
              <div className="card" key={module.name}>
                <Icon size={30} color={module.color} />
                <h3>{module.name}</h3>
                <p>{module.description}</p>
                <span className="tag">Research module</span>
              </div>
            );
          })}
        </div>
      </section>
      <Footer />
    </div>
  );
}

// =====================================================
// HOW IT WORKS PAGE
// =====================================================

function HowItWorksPage() {
  const steps = [
    { title: "Hospital Data", description: "Each hospital has its own medical image dataset.", icon: Database },
    { title: "Local Model Training", description: "The hospital client trains a model using its local data.", icon: Cpu },
    { title: "Model Updates", description: "The client sends model parameters or updates instead of raw images.", icon: FileImage },
    { title: "Central Aggregation", description: "The central server combines the received model updates.", icon: Server },
    { title: "Global Model Distribution", description: "The updated global model is distributed to participating clients.", icon: Workflow },
  ];

  return (
    <div className="page">
      <Navbar />
      <section className="content-page">
        <div className="eyebrow">FEDERATED WORKFLOW</div>
        <h1>How MedFed AI Works</h1>
        <p>
          This is the intended federated learning architecture. The current
          prototype demonstrates simulated clients and FedAvg training on one computer.
        </p>
        {steps.map((step, index) => {
          const Icon = step.icon;
          return (
            <div className="step" key={step.title}>
              <div className="step-number">{index + 1}</div>
              <div>
                <Icon color="#45dfbd" size={24} />
                <h3>{step.title}</h3>
                <p>{step.description}</p>
              </div>
            </div>
          );
        })}
      </section>
      <Footer />
    </div>
  );
}

// =====================================================
// PRIVACY PAGE
// =====================================================

function PrivacyPage() {
  return (
    <div className="page">
      <Navbar />
      <section className="content-page">
        <div className="eyebrow">PRIVACY & SECURITY</div>
        <h1>Privacy in MedFed AI</h1>
        <p>
          Federated learning is intended to allow multiple organizations to
          train a shared model without transferring their raw training images
          to a central training server.
        </p>
        <div className="card-grid">
          <div className="card">
            <LockKeyhole color="#45dfbd" size={30} />
            <h3>Differential Privacy</h3>
            <p>A proposed technique for reducing the risk of revealing information about individual training samples.</p>
          </div>
          <div className="card">
            <ShieldCheck color="#45dfbd" size={30} />
            <h3>Secure Aggregation</h3>
            <p>A proposed technique to protect individual client updates during aggregation.</p>
          </div>
        </div>
        <div className="disclaimer">
          <strong>Current prototype limitation:</strong>
          <br />
          The present ZIP upload feature sends datasets to the central backend
          for storage and validation. The simulated FedAvg experiment uses
          local data partitions. Differential privacy and secure aggregation
          are not currently enabled. Do not upload real patient-identifiable
          medical data.
        </div>
      </section>
      <Footer />
    </div>
  );
}

// =====================================================
// REUSABLE PREDICTION RESULT
// =====================================================

function PredictionResult({ result, title }) {
  if (!result) return null;

  return (
    <div className="prediction-result">
      <div className="eyebrow">
        <Sparkles size={14} style={{ display: "inline", verticalAlign: "middle" }} />{" "}
        {title}
      </div>
      <h3>Predicted Class</h3>
      <div className="prediction-confidence">
        {result.display_name || result.predicted_class}
      </div>
      <p className="prediction-description">
        Confidence: {result.confidence}%
      </p>
      <h3>Class Probabilities</h3>
      {Object.entries(result.probabilities || {}).map(([className, probability]) => (
        <div className="probability-row" key={className}>
          <div className="probability-label">
            <span>{className}</span>
            <strong>{probability}%</strong>
          </div>
          <div className="probability-track">
            <div
              className="probability-fill"
              style={{
                width: `${Math.min(Math.max(Number(probability), 0), 100)}%`,
              }}
            />
          </div>
        </div>
      ))}
      <p className="prediction-description">Model: {result.model}</p>
      {result.aggregation && (
        <p className="prediction-description">
          Aggregation: {result.aggregation} · Clients: {result.clients} ·
          Communication rounds: {result.communication_rounds}
        </p>
      )}
    </div>
  );
}

// =====================================================
// DASHBOARD PAGE
// =====================================================

function DashboardPage() {
  const navigate = useNavigate();

  const [hospitals, setHospitals] = useState([]);
  const [uploadStatuses, setUploadStatuses] = useState([]);
  const [uploading, setUploading] = useState({});
  const [uploadMessages, setUploadMessages] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Lung X-ray prediction states
  const [selectedXray, setSelectedXray] = useState(null);
  const [xrayPreview, setXrayPreview] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState("");

  // Brain MRI prediction states
  const [selectedBrainMRI, setSelectedBrainMRI] = useState(null);
  const [brainMRIPreview, setBrainMRIPreview] = useState("");
  const [brainMRIPrediction, setBrainMRIPrediction] = useState(null);
  const [brainMRIPredicting, setBrainMRIPredicting] = useState(false);
  const [brainMRIPredictionError, setBrainMRIPredictionError] = useState("");

  // Existing lung federated training states
  const [trainingStatus, setTrainingStatus] = useState({
    running: false,
    completed: false,
    message: "Federated training has not started.",
    return_code: null,
  });
  const [trainingError, setTrainingError] = useState("");

  // Load hospital information
  useEffect(() => {
    const loadHospitals = async () => {
      try {
        const data = await getHospitals();
        setHospitals(data.hospitals || []);
      } catch (err) {
        console.error("Hospital API error:", err);
        setError("Unable to connect to the backend.");
      } finally {
        setLoading(false);
      }
    };
    loadHospitals();
  }, []);

  const refreshUploadStatus = async () => {
    try {
      const data = await getUploadStatus();
      setUploadStatuses(data.uploads || []);
    } catch (err) {
      console.error("Upload status error:", err);
    }
  };

  useEffect(() => {
    refreshUploadStatus();
  }, []);

  // Poll the existing lung training status while it is running
  useEffect(() => {
    let active = true;
    let interval;

    const refreshTrainingStatus = async () => {
      try {
        const data = await getTrainingStatus();
        if (active) setTrainingStatus(data);
      } catch (err) {
        console.error("Training status error:", err);
      }
    };

    refreshTrainingStatus();

    if (trainingStatus.running) {
      interval = setInterval(refreshTrainingStatus, 2000);
    }

    return () => {
      active = false;
      if (interval) clearInterval(interval);
    };
  }, [trainingStatus.running]);

  const handleStartTraining = async () => {
    try {
      setTrainingError("");
      const result = await startFederatedTraining();
      setTrainingStatus((previous) => ({
        ...previous,
        running: true,
        completed: false,
        message: result.message || "Federated training started.",
        return_code: null,
      }));
    } catch (err) {
      console.error("Federated training error:", err);
      setTrainingError(
        err.response?.data?.detail ||
        "Unable to start federated training. Check the backend."
      );
    }
  };

  const handleDatasetUpload = async (hospitalId, file) => {
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".zip")) {
      setUploadMessages((previous) => ({
        ...previous,
        [hospitalId]: "Please select a ZIP file.",
      }));
      return;
    }

    setUploading((previous) => ({ ...previous, [hospitalId]: true }));
    setUploadMessages((previous) => ({
      ...previous,
      [hospitalId]: "Uploading and validating dataset...",
    }));

    try {
      const result = await uploadHospitalDataset(hospitalId, file);
      setUploadMessages((previous) => ({
        ...previous,
        [hospitalId]: result.message || "Dataset uploaded successfully.",
      }));
      await refreshUploadStatus();
    } catch (err) {
      console.error("Dataset upload error:", err);
      setUploadMessages((previous) => ({
        ...previous,
        [hospitalId]: err.response?.data?.detail || "Dataset upload failed.",
      }));
    } finally {
      setUploading((previous) => ({ ...previous, [hospitalId]: false }));
    }
  };

  // Common image validation
  const validateImage = (file) => {
    const allowedTypes = ["image/jpeg", "image/png", "image/bmp"];
    if (!allowedTypes.includes(file.type)) {
      return "Please select a JPG, JPEG, PNG or BMP image.";
    }
    if (file.size > 10 * 1024 * 1024) {
      return "Image size must be less than 10 MB.";
    }
    return "";
  };

  // Lung X-ray selection and prediction
  const handleXraySelection = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const validationError = validateImage(file);
    if (validationError) {
      setPredictionError(validationError);
      event.target.value = "";
      return;
    }

    if (xrayPreview) URL.revokeObjectURL(xrayPreview);
    setSelectedXray(file);
    setXrayPreview(URL.createObjectURL(file));
    setPrediction(null);
    setPredictionError("");
  };

  const handleLungPrediction = async () => {
    if (!selectedXray) {
      setPredictionError("Please select an X-ray image first.");
      return;
    }

    try {
      setPredicting(true);
      setPredictionError("");
      setPrediction(null);
      const result = await predictLungXray(selectedXray);
      setPrediction(result);
    } catch (err) {
      console.error("Lung prediction error:", err);
      const message = err.response?.data?.detail;
      setPredictionError(
        typeof message === "string"
          ? message
          : "Prediction failed. Please check whether the backend is running."
      );
    } finally {
      setPredicting(false);
    }
  };

  const clearXray = () => {
    if (xrayPreview) URL.revokeObjectURL(xrayPreview);
    setSelectedXray(null);
    setXrayPreview("");
    setPrediction(null);
    setPredictionError("");
    const input = document.getElementById("lung-xray-input");
    if (input) input.value = "";
  };

  // Brain MRI selection and prediction
  const handleBrainMRISelection = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const validationError = validateImage(file);
    if (validationError) {
      setBrainMRIPredictionError(validationError);
      event.target.value = "";
      return;
    }

    if (brainMRIPreview) URL.revokeObjectURL(brainMRIPreview);
    setSelectedBrainMRI(file);
    setBrainMRIPreview(URL.createObjectURL(file));
    setBrainMRIPrediction(null);
    setBrainMRIPredictionError("");
  };

  const handleBrainMRIPrediction = async () => {
    if (!selectedBrainMRI) {
      setBrainMRIPredictionError("Please select a Brain MRI image first.");
      return;
    }

    try {
      setBrainMRIPredicting(true);
      setBrainMRIPredictionError("");
      setBrainMRIPrediction(null);
      const result = await predictBrainMRI(selectedBrainMRI);
      setBrainMRIPrediction(result);
    } catch (err) {
      console.error("Brain MRI prediction error:", err);
      const message = err.response?.data?.detail;
      setBrainMRIPredictionError(
        typeof message === "string"
          ? message
          : "Brain MRI prediction failed. Please check whether the backend is running."
      );
    } finally {
      setBrainMRIPredicting(false);
    }
  };

  const clearBrainMRI = () => {
    if (brainMRIPreview) URL.revokeObjectURL(brainMRIPreview);
    setSelectedBrainMRI(null);
    setBrainMRIPreview("");
    setBrainMRIPrediction(null);
    setBrainMRIPredictionError("");
    const input = document.getElementById("brain-mri-input");
    if (input) input.value = "";
  };

  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="page dashboard">
      <nav className="navbar">
        <Brand />
        <button type="button" className="outline-btn" onClick={() => navigate("/")}>
          <ArrowLeft size={16} /> Back to Home
        </button>
      </nav>

      <div className="dash-layout">
        <aside className="sidebar">
          <button type="button" className="side-link active">
            <LayoutDashboard size={18} /> Overview
          </button>
          <button type="button" className="side-link" onClick={() => scrollToSection("hospital-section")}>
            <Hospital size={18} /> Hospitals
          </button>
          <button type="button" className="side-link" onClick={() => scrollToSection("prediction-section")}>
            <Activity size={18} /> X-ray Prediction
          </button>
          <button type="button" className="side-link" onClick={() => scrollToSection("brain-mri-section")}>
            <Brain size={18} /> Brain MRI Prediction
          </button>
          <button type="button" className="side-link" onClick={() => scrollToSection("training-section")}>
            <Cpu size={18} /> Federated Training
          </button>
          <button type="button" className="side-link" onClick={() => scrollToSection("module-section")}>
            <Dna size={18} /> Disease Modules
          </button>
          <button type="button" className="side-link" onClick={() => navigate("/")}>
            <Home size={18} /> Home
          </button>
        </aside>

        <main className="dash-content">
          <div className="dash-title">
            <div>
              <div className="eyebrow">MEDFED AI / OVERVIEW</div>
              <h1>Research Dashboard</h1>
              <p>Monitor your medical AI research platform.</p>
            </div>
            <span className="tag">Prototype interface</span>
          </div>

          <div className="dash-stats">
            <div className="dash-stat">
              <Hospital size={23} /><strong>03</strong><span>Simulated hospitals</span>
            </div>
            <div className="dash-stat">
              <Dna size={23} /><strong>05</strong><span>Disease modules</span>
            </div>
            <div className="dash-stat">
              <Users size={23} /><strong>03</strong><span>Research clients</span>
            </div>
            <div className="dash-stat">
              <ShieldCheck size={23} /><strong>FL</strong><span>Learning approach</span>
            </div>
          </div>

          {/* HOSPITAL NETWORK AND DATASET UPLOADS */}
          <h2 className="dash-section-title" id="hospital-section">
            Hospital Network & Dataset Uploads
          </h2>

          {loading && <p>Loading hospital information...</p>}
          {error && <p style={{ color: "#fb923c" }}>{error}</p>}
          {!loading && !error && hospitals.length === 0 && (
            <p>No hospitals were returned by the backend.</p>
          )}

          {!loading && !error && (
            <div className="hospital-grid">
              {hospitals.map((hospital) => {
                const status = uploadStatuses.find(
                  (item) => Number(item.hospital_id) === Number(hospital.id)
                );
                const inputId = `hospital-upload-${hospital.id}`;

                return (
                  <div className="hospital-card" key={hospital.id}>
                    <Hospital size={27} color="#45dfbd" />
                    <h3>{hospital.name}</h3>
                    <p>Hospital Client {hospital.id}</p>
                    <div className="status">
                      {status?.uploaded ? "✓ Dataset Uploaded" : "Waiting for Dataset"}
                    </div>
                    <p>Upload status: {status?.status || "Checking..."}</p>

                    <input
                      id={inputId}
                      type="file"
                      accept=".zip,application/zip"
                      disabled={!!uploading[hospital.id]}
                      style={{ display: "none" }}
                      onChange={(event) => {
                        const selectedFile = event.target.files?.[0];
                        handleDatasetUpload(hospital.id, selectedFile);
                        event.target.value = "";
                      }}
                    />

                    <button
                      type="button"
                      className="btn upload-label"
                      disabled={!!uploading[hospital.id]}
                      onClick={() => {
                        const fileInput = document.getElementById(inputId);
                        if (fileInput && !uploading[hospital.id]) fileInput.click();
                      }}
                    >
                      <Upload size={16} />
                      {uploading[hospital.id] ? "Uploading..." : "Upload ZIP Dataset"}
                    </button>

                    {uploadMessages[hospital.id] && (
                      <p className="upload-message">{uploadMessages[hospital.id]}</p>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* LUNG X-RAY PREDICTION */}
          <h2 className="dash-section-title" id="prediction-section">
            Lung X-ray AI Prediction
          </h2>

          <div className="prediction-section">
            <div className="prediction-heading">
              <Activity size={27} />
              <h3>Federated Lung Disease Classifier</h3>
            </div>
            <p className="prediction-description">
              Upload a chest X-ray image to test the saved ResNet18 federated
              global model. The model predicts one of three classes: Normal,
              Pneumonia or Tuberculosis.
            </p>

            <div className="prediction-grid">
              <div className="prediction-upload">
                {xrayPreview ? (
                  <>
                    <img src={xrayPreview} alt="Selected chest X-ray preview" className="xray-preview" />
                    <p>Selected image: {selectedXray?.name}</p>
                    <button type="button" className="outline-btn" onClick={clearXray}>
                      <RotateCcw size={16} /> Remove Image
                    </button>
                  </>
                ) : (
                  <>
                    <ImagePlus size={45} />
                    <h3>Upload Chest X-ray</h3>
                    <p>Choose a JPG, JPEG, PNG or BMP image. Maximum file size: 10 MB.</p>
                    <label htmlFor="lung-xray-input" className="btn">
                      <Upload size={16} /> Choose X-ray Image
                    </label>
                  </>
                )}
                <input
                  id="lung-xray-input"
                  type="file"
                  accept=".jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp"
                  style={{ display: "none" }}
                  onChange={handleXraySelection}
                />
              </div>

              <div>
                <h3>Run AI Prediction</h3>
                <p className="prediction-description">
                  The selected image will be sent to your FastAPI backend.
                  The saved federated ResNet18 model will process the image
                  and return its prediction probabilities.
                </p>
                <button
                  type="button"
                  className="btn"
                  style={{ width: "100%", marginTop: 20 }}
                  disabled={!selectedXray || predicting}
                  onClick={handleLungPrediction}
                >
                  <Sparkles size={17} />
                  {predicting ? "Analyzing X-ray..." : "Predict Using Federated Model"}
                </button>
                {predicting && (
                  <p className="prediction-description">Please wait while the model processes your image...</p>
                )}
                {predictionError && <div className="prediction-error">{predictionError}</div>}
              </div>
            </div>

            <PredictionResult result={prediction} title="MODEL PREDICTION RESULT" />

            <div className="prediction-note">
              <strong>Research disclaimer:</strong> This prediction is generated
              by an educational research prototype. It is not a medical diagnosis
              and must not be used to make treatment decisions. Model confidence
              is not a measure of clinical certainty.
            </div>
          </div>

          {/* BRAIN MRI PREDICTION */}
          <h2 className="dash-section-title" id="brain-mri-section">
            Brain MRI AI Prediction
          </h2>

          <div className="prediction-section">
            <div className="prediction-heading">
              <Brain size={27} />
              <h3>Federated Brain Tumor Classifier</h3>
            </div>
            <p className="prediction-description">
              Upload a Brain MRI image to test the saved ResNet18 federated
              global model. The model predicts one of four dataset classes:
              Glioma, Meningioma, No Tumor or Pituitary.
            </p>

            <div className="prediction-grid">
              <div className="prediction-upload">
                {brainMRIPreview ? (
                  <>
                    <img src={brainMRIPreview} alt="Selected Brain MRI preview" className="xray-preview" />
                    <p>Selected image: {selectedBrainMRI?.name}</p>
                    <button type="button" className="outline-btn" onClick={clearBrainMRI}>
                      <RotateCcw size={16} /> Remove Image
                    </button>
                  </>
                ) : (
                  <>
                    <ImagePlus size={45} />
                    <h3>Upload Brain MRI</h3>
                    <p>Choose a JPG, JPEG, PNG or BMP image. Maximum file size: 10 MB.</p>
                    <label htmlFor="brain-mri-input" className="btn">
                      <Upload size={16} /> Choose MRI Image
                    </label>
                  </>
                )}
                <input
                  id="brain-mri-input"
                  type="file"
                  accept=".jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp"
                  style={{ display: "none" }}
                  onChange={handleBrainMRISelection}
                />
              </div>

              <div>
                <h3>Run Brain MRI Prediction</h3>
                <p className="prediction-description">
                  The selected image will be sent to your FastAPI backend.
                  The saved federated ResNet18 model will return its predicted
                  class and probabilities.
                </p>
                <button
                  type="button"
                  className="btn"
                  style={{ width: "100%", marginTop: 20 }}
                  disabled={!selectedBrainMRI || brainMRIPredicting}
                  onClick={handleBrainMRIPrediction}
                >
                  <Sparkles size={17} />
                  {brainMRIPredicting ? "Analyzing Brain MRI..." : "Predict Using Federated Model"}
                </button>
                {brainMRIPredicting && (
                  <p className="prediction-description">Please wait while the model processes your image...</p>
                )}
                {brainMRIPredictionError && (
                  <div className="prediction-error">{brainMRIPredictionError}</div>
                )}
              </div>
            </div>

            <PredictionResult result={brainMRIPrediction} title="BRAIN MRI MODEL PREDICTION" />

            <div className="prediction-note">
              <strong>Research disclaimer:</strong> This prediction is generated
              by an educational research prototype. It is not a medical diagnosis
              and must not be used to make treatment decisions. Model confidence
              is not a measure of clinical certainty.
            </div>
          </div>

          {/* EXISTING LUNG FEDERATED TRAINING CONTROL */}
          <h2 className="dash-section-title" id="training-section">
            Federated Learning Experiment
          </h2>

          <div className="training-card">
            <div className="training-heading">
              <Cpu size={27} />
              <h3>Federated Training Control</h3>
            </div>
            <p className="muted">
              Start a federated learning experiment using three simulated
              clients. Each client trains locally on its assigned portion of
              the existing lung dataset. The central server combines the model
              updates using Federated Averaging (FedAvg).
            </p>

            <div className="training-detail">
              <span>Number of simulated clients</span><strong>3</strong>
            </div>
            <div className="training-detail">
              <span>Aggregation algorithm</span><strong>FedAvg</strong>
            </div>
            <div className="training-detail">
              <span>Current training status</span>
              <strong>
                {trainingStatus.running
                  ? "Training in progress..."
                  : trainingStatus.completed
                  ? "Training completed successfully"
                  : trainingStatus.return_code !== null
                  ? "Training failed"
                  : "Not currently training"}
              </strong>
            </div>

            <p className="prediction-description">{trainingStatus.message}</p>
            {trainingError && <div className="prediction-error">{trainingError}</div>}

            <button
              type="button"
              className="btn training-button"
              disabled={trainingStatus.running}
              onClick={handleStartTraining}
            >
              <Cpu size={17} />
              {trainingStatus.running ? "Training in Progress..." : "Start Federated Training"}
            </button>

            <p className="prediction-description">
              This experiment uses the existing local three-client lung dataset.
              Hospital ZIP uploads are not used by this training script.
            </p>
            <div className="prediction-note">
              <strong>Privacy limitation:</strong> This is a simulated federated
              experiment on one computer. Differential privacy and secure
              aggregation are not enabled in the current training script.
            </div>
          </div>

          {/* MEDICAL AI MODULES */}
          <h2 className="dash-section-title" id="module-section">Medical AI Modules</h2>
          <div className="dashboard-modules">
            {modules.map((module) => {
              const Icon = module.icon;
              return (
                <div className="dashboard-module" key={module.name}>
                  <div className="module-icon" style={{ color: module.color }}>
                    <Icon size={25} />
                  </div>
                  <div>
                    <h3>{module.name}</h3>
                    <p>{module.description}</p>
                    <span className="tag">Research module</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* GLOBAL MODEL INFORMATION */}
          <h2 className="dash-section-title">Global Model Information</h2>
          <div className="card">
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 15 }}>
              <Server color="#45dfbd" size={25} />
              <h3>Global Model Architecture</h3>
            </div>
            <p className="muted">
              The lung federated training experiment uses ResNet18 with three
              simulated clients. Clients train locally on separate data
              partitions, and the central server combines their model updates
              using FedAvg. The resulting lung global model is saved and
              reloaded for lung X-ray prediction. The separate Brain MRI model
              was trained and saved using its own three-client FedAvg experiment.
            </p>
            <span className="tag">ResNet18 · 3 simulated clients · FedAvg</span>
          </div>

          <div className="disclaimer">
            Educational research prototype only. Not for clinical use.
          </div>
        </main>
      </div>
    </div>
  );
}

// =====================================================
// MAIN APP WITH FIVE ROUTES
// =====================================================

function App() {
  return (
    <>
      <GlobalStyles />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/disease-models" element={<DiseaseModelsPage />} />
        <Route path="/how-it-works" element={<HowItWorksPage />} />
        <Route path="/privacy" element={<PrivacyPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="*" element={<HomePage />} />
      </Routes>
    </>
  );
}

export default App;
