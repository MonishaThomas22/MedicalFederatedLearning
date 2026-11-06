
import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

// Check backend health
export const getHealth = async () => {
  const response = await API.get("/api/health");
  return response.data;
};

// Get hospital information
export const getHospitals = async () => {
  const response = await API.get("/api/hospitals");
  return response.data;
};

// Get dataset upload status
export const getUploadStatus = async () => {
  const response = await API.get("/api/uploads/status");
  return response.data;
};

// Upload a hospital dataset ZIP
export const uploadHospitalDataset = async (hospitalId, file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post(
    `/api/hospitals/${hospitalId}/upload`,
    formData
  );

  return response.data;
};

// Predict Lung X-ray using federated model
export const predictLungXray = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post("/api/predict/lung", formData);
  return response.data;
};

// Predict Brain MRI using federated model
export const predictBrainMRI = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post("/api/predict/brain-mri", formData);
  return response.data;
};

// Start federated training
export const startFederatedTraining = async () => {
  const response = await API.post("/api/training/start");
  return response.data;
};

// Get federated training status
export const getTrainingStatus = async () => {
  const response = await API.get("/api/training/status");
  return response.data;
};

// Get global federated model status
export const getGlobalModelStatus = async () => {
  const response = await API.get("/api/models/status");
  return response.data;
};

export default API;
