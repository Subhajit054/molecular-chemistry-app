# 🧬 MOLECULE AI — 3D Geometric GNN & LLM Molecular Intelligence Suite

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph_Neural_Networks-3C2179)](https://pyg.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/Gemini_API-2.5_Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

An enterprise-grade cheminformatics and deep learning web platform that couples **Continuous-Filter 3D Graph Neural Networks (GIN + SchNet)** with the **Google Gemini API** to perform real-time conformational modeling, multi-task bioassay screening across 128 targets, and automated pharmacological dossiers.

---

## 🌟 Key Architecture & Features

- **Geometric Deep Learning Engine:** Hybrid Graph Isomorphism Network (`GINConv`) fused with continuous Gaussian Radial Basis Function (RBF) spatial filters derived from SchNet convolutions.
- **Automated 3D Conformer Modeling:** Converts 1D SMILES to 3D energy-minimized conformers using RDKit's Distance Geometry (ETKDG) and MMFF94 force field.
- **128-Task High-Throughput Bioassay Screening:** Masked multi-task prediction trained on the `ogbg-molpcba` benchmark for hepatotoxicity, hERG cardiotoxicity, and target binding.
- **Google-Style Compound Search:** Live autocomplete and fuzzy matching over a local drug database with seamless REST fallback to the PubChem NCBI API.
- **Gemini Automated Intelligence:** Generates pharmacological Mechanism of Action (MOA), Lipinski rule-of-5 solvation adherence, and safety dossiers.
- **Interactive WebGL Visualizer:** Hardware-accelerated 3D molecular inspection powered by `3Dmol.js`.

---

## 🛠️ Quickstart & Local Setup

### 1. Clone & Install
```bash
git clone [https://github.com/your-username/molecule-ai-suite.git](https://github.com/your-username/molecule-ai-suite.git)
cd molecule-ai-suite
pip install -r requirements.txt