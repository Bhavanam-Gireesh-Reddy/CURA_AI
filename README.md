# CURA AI — Medical Prediction and RAG Assistant

CURA AI is a FastAPI web application that combines trained machine-learning and deep-learning models with retrieval-augmented generation (RAG). It provides three guided prediction workflows—liver disease, X-ray fracture, and brain MRI classification—and generates an A4 PDF report containing the model result, retrieved medical guidance, and a medical disclaimer.

> **Medical notice:** CURA AI is an educational decision-support prototype. It is not a diagnostic device and must not replace a qualified clinician, radiologist, or emergency service. Model outputs and generated reports must be clinically reviewed before any medical decision is made.

## Contents

- [Capabilities](#capabilities)
- [Application flow](#application-flow)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation and startup](#installation-and-startup)
- [Configuration](#configuration)
- [Using the application](#using-the-application)
- [API reference](#api-reference)
- [LangChain agent and tools](#langchain-agent-and-tools)
- [RAG and PDF reports](#rag-and-pdf-reports)
- [Models and artifacts](#models-and-artifacts)
- [Development notes](#development-notes)
- [Safety, privacy, and limitations](#safety-privacy-and-limitations)

## Capabilities

| Area | Input | Output |
| --- | --- | --- |
| Liver disease prediction | Structured patient laboratory values and a selected ML model | `Liver Disease Detected` or `No Liver Disease`, confidence, model name, and PDF report |
| X-ray fracture detection | X-ray image upload | Fracture class, confidence, image preview, and PDF report |
| Brain MRI classification | MRI image upload | MRI class, confidence, image preview, and PDF report |
| Medical knowledge chat | Natural-language question | Grounded answer from the local Chroma knowledge base |
| Guided navigation | Requests for prediction or image analysis | A chat button that opens the corresponding liver, X-ray, or MRI workflow |

The interface is intentionally split into two modes:

1. **Prediction intent** — requests such as “predict liver disease,” “analyse my X-ray,” or “classify my MRI” show a workflow button. The user must then provide validated clinical values or upload an image.
2. **Information intent** — requests about precautions, prevention, symptoms, causes, diet, treatment, care, or general medical information go directly to RAG. They do not show a prediction/upload button merely because a condition is named.

## Application flow

```text
Browser chat
    │
    ▼
LangChain agent routing (agent/service.py)
    ├── Explicit prediction or image-analysis request ──► navigation button
    └── Information / precaution request ──────────────► Chroma RAG + Gemini answer

Liver form / X-ray upload / MRI upload
    │
    ▼
Local trained model prediction
    │
    ▼
Condition-specific Chroma retrieval + Gemini generation
    │
    ▼
Downloadable A4 PDF report
```

## Architecture

### Web layer

- **FastAPI** serves the web page and JSON endpoints in `main.py`.
- **Jinja2** renders `templates/index.html`.
- **Vanilla JavaScript and CSS** in `static/` submit forms, display predictions, manage the chat window, and expose report download links.
- Uploaded images are served from `/uploads`; generated reports are served from `/reports`.

### Prediction layer

- `machine_learning/scripts.py` loads the selected serialized scikit-learn/XGBoost model and scaler, then predicts from the liver form data.
- `deep_learning/scripts.py` loads the trained EfficientNet-based X-ray model and maps the highest-probability class to a fracture label.
- `deep_learning2/scripts.py` loads the MRI model and maps the highest-probability class to an MRI label.

### RAG layer

- Medical source documents live in `RAG/data/` as TXT, PDF, and DOCX files.
- Chroma persists the vector database in `chromadb/`.
- `RAG/scripts.py` embeds the question, retrieves the top relevant chunks, and asks Gemini to produce an answer grounded in those chunks.
- The RAG endpoint requires `GEMINI_API_KEY`.

### Agent and report layer

- `agent/tools.py` defines exactly four LangChain tools.
- `agent/service.py` applies deterministic intent routing so local prediction navigation does not depend on a cloud LLM being available. `create_cura_agent()` is available when deploying the same tools with a configured LangChain chat model.
- `agent/reports.py` writes dependency-free PDF files. Reports follow a clean A4 text layout with generous spacing, Markdown-style headings/bullets, prediction details, RAG content, and a disclaimer.

## Project structure

```text
CURA_AI/
├── agent/
│   ├── service.py                 # Intent routing and LangChain agent factory
│   ├── tools.py                   # Four LangChain tools
│   └── reports.py                 # A4 prediction/RAG PDF generator
├── machine_learning/
│   ├── artifacts/                 # Liver models, scaler, and feature names
│   └── scripts.py                 # Liver prediction service
├── deep_learning/
│   ├── artifacts/                 # X-ray model and class names
│   └── scripts.py                 # X-ray prediction service
├── deep_learning2/
│   ├── artifacts/                 # MRI model and class names
│   └── scripts.py                 # MRI prediction service
├── RAG/
│   ├── data/                      # Curated medical reference documents
│   ├── scripts.py                 # Retrieval and Gemini answer generation
│   ├── retrieval.py               # Chroma top-k retrieval helper
│   ├── doc_loader.py              # TXT/PDF/DOCX document loaders
│   ├── chunking.py                # Semantic document chunking
│   └── vector_db.py               # Chroma vector store setup
├── chromadb/                      # Persisted Chroma database
├── static/                        # Browser JavaScript and styles
├── templates/index.html           # Main UI
├── uploads/                       # Runtime image uploads
├── reports/                       # Runtime generated PDF reports (ignored by Git)
├── main.py                        # FastAPI application and endpoints
└── requirements.txt               # Python dependencies
```

## Requirements

- Python 3.10 or later
- A Gemini API key for RAG answers and RAG-backed reports
- Sufficient memory to load TensorFlow models and the sentence-transformer embedding model
- Image files supported by Pillow/TensorFlow (for example PNG, JPG, JPEG)

TensorFlow is configured for CPU use at application startup in `main.py`. A GPU is not required for inference.

## Installation and startup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Start the server:

```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). FastAPI’s interactive endpoint documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### First startup notes

- The application loads TensorFlow models lazily when the relevant image endpoint is first called.
- The RAG pipeline loads the Hugging Face embedding model and persisted Chroma store on its first RAG request.
- The repository includes a persisted `chromadb/` directory. If it is absent, create the vector database from the documents in `RAG/data/` using the RAG ingestion utilities before querying RAG.
- Without `GEMINI_API_KEY`, direct model predictions still work. The report generator will create a fallback report that states that retrieved guidance is temporarily unavailable.

## Configuration

| Variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes for RAG | Authenticates Gemini answer/report generation |
| `CUDA_VISIBLE_DEVICES` | Set by the app | Set to `-1` to use CPU inference |
| `TF_CPP_MIN_LOG_LEVEL` | Set by the app | Reduces TensorFlow log noise |

Do not commit `.env` files or real API keys. `.env` is already ignored by Git.

## Using the application

### Liver disease workflow

1. Open **Liver Disease Prediction**, or ask the chat for a liver prediction and click its button.
2. Select one of `random_forest`, `logistic_regression`, or `xgboost`.
3. Enter the required demographic and laboratory values.
4. Select **Predict Liver Disease**.
5. Review the prediction, confidence, selected model, and choose **Download PDF Report**.

### X-ray workflow

1. Open **X-Ray Fracture Detection**, or ask the chat to analyse an X-ray/fracture image.
2. Upload an X-ray image.
3. Select **Analyze X-Ray**.
4. Review the predicted fracture class and confidence.
5. Download the RAG-backed PDF report.

### MRI workflow

1. Open **MRI Brain Tumor Detection**, or ask the chat to analyse an MRI image.
2. Upload an MRI image.
3. Select **Analyze MRI**.
4. Review the predicted MRI class and confidence.
5. Download the RAG-backed PDF report.

### Medical information and precautions

Ask questions such as:

```text
I have liver disease. What precautions should I take?
What foods should be avoided for fatty liver?
What are the precautions after a wrist fracture?
Tell me about meningioma symptoms.
```

These requests use the knowledge base directly and do not show a form-navigation tool.

## API reference

### `GET /`

Returns the main CURA AI web interface.

### `POST /liver_predict`

Predicts liver disease from structured JSON and generates a report.

Example request:

```json
{
  "model": "random_forest",
  "Age": 65,
  "Gender": 1,
  "Total_Bilirubin": 1.2,
  "Direct_Bilirubin": 0.4,
  "Alkaline_Phosphotase": 200,
  "Alamine_Aminotransferase": 30,
  "Aspartate_Aminotransferase": 40,
  "Total_Protiens": 7.0,
  "Albumin": 3.5,
  "Albumin_and_Globulin_Ratio": 1.0
}
```

Example response shape:

```json
{
  "status": "success",
  "prediction": "Liver Disease Detected",
  "confidence": 64.73,
  "model_used": "random_forest",
  "report": "/reports/<generated-report>.pdf"
}
```

### `POST /predict`

Accepts multipart form data with a required `file` image field. Returns `fracture_type`, `confidence`, `image`, and `report`.

### `POST /predict_mri`

Accepts multipart form data with a required `file` image field. Returns `mri_type`, `confidence`, `image`, and `report`.

### `POST /rag_query`

Accepts a chat question:

```json
{ "question": "What precautions should I take for liver disease?" }
```

Informational requests return an `answer`. Explicit prediction/image-analysis requests return an `answer` and a navigation `tool` payload, for example:

```json
{
  "answer": "Open the liver assessment form to enter patient values and run the ML prediction.",
  "tool": {
    "tool_name": "liver",
    "action": "navigate",
    "target": "#liver"
  }
}
```

## LangChain agent and tools

The four tools in `agent/tools.py` are:

| Tool | Purpose |
| --- | --- |
| `liver_prediction_tool` | Returns navigation metadata for the liver ML form |
| `xray_fracture_tool` | Returns navigation metadata for the X-ray upload form |
| `mri_analysis_tool` | Returns navigation metadata for the MRI upload form |
| `medical_rag_tool` | Searches the medical knowledge base and returns a grounded answer |

`create_cura_agent(model)` creates a LangChain `create_agent` graph using these four tools. The web application uses deterministic routing in `get_agent_response()` for predictable healthcare UX: informational/precaution requests call RAG directly; explicit model/image requests call only the appropriate navigation tool.

## RAG and PDF reports

### RAG sources

The knowledge base includes liver disease, liver health/prevention, fracture, and brain MRI/tumor reference materials in `RAG/data/`. The response prompt instructs Gemini to answer only from retrieved context and include a medical disclaimer.

### Report contents

Each completed prediction creates an A4 PDF under `reports/` and returns a URL to it. The report includes:

- prediction type and confidence;
- condition-specific medical information retrieved through RAG;
- a timestamp; and
- a medical disclaimer.

The report writer has a graceful fallback: if RAG or Gemini is unavailable, the prediction response still succeeds and the report identifies that retrieved guidance could not be produced.

## Models and artifacts

### Liver models

The liver workflow reads `machine_learning/artifacts/feature_names.json`, applies the saved scaler, and loads one of the serialized model artifacts selected in the UI.

Input features are:

```text
Age
Gender
Total_Bilirubin
Direct_Bilirubin
Alkaline_Phosphotase
Alamine_Aminotransferase
Aspartate_Aminotransferase
Total_Protiens
Albumin
Albumin_and_Globulin_Ratio
```

The feature spelling is intentionally retained to match the trained artifacts and API schema.

### X-ray classes

The X-ray model classes are loaded from `deep_learning/artifacts/class_names.json`:

- Elbow Fracture
- Finger Fracture
- Forearm Fracture
- Humerus Fracture
- Shoulder Fracture
- Wrist Fracture

### MRI classes

The MRI model classes are loaded from `deep_learning2/artifacts/class_names.json`:

- Glioma
- Meningioma
- No Tumor
- Pituitary

Classifications are model outputs, not clinical diagnoses.

## Development notes

### Useful checks

```bash
# Check syntax without running model inference
python3 -m compileall main.py agent RAG machine_learning deep_learning deep_learning2

# Run the FastAPI application in development mode
uvicorn main:app --reload
```

### Runtime files

- `reports/` is ignored because it contains generated PDFs.
- `uploads/` stores uploaded images. Treat it as sensitive runtime data and avoid committing it.
- Chroma files may change when the vector store is written. Keep the persisted store aligned with the documents used for the deployment.

### Extending the knowledge base

1. Add reviewed TXT, PDF, or DOCX documents to `RAG/data/`.
2. Rebuild or update the Chroma vector store with the ingestion utilities in `RAG/`.
3. Test representative queries and validate their retrieved sources before deployment.

Only add clinically reviewed, rights-cleared documents. Retrieval quality is limited by the completeness and accuracy of this source material.

## Safety, privacy, and limitations

- Do not use this project to make urgent-care, triage, medication, or treatment decisions.
- A confidence score indicates model certainty under its learned distribution; it is not a clinical probability or proof of disease.
- Image quality, preprocessing, and training-data coverage materially affect model performance.
- RAG answers can only reflect the chunks retrieved from the local corpus. They can be incomplete or unavailable when source coverage is limited.
- Uploaded medical images and entered health data are sensitive. Deploy behind appropriate authentication, encryption, retention controls, audit logging, and access policies before handling real patient data.
- Validate the model, RAG corpus, security controls, and regulatory requirements for the intended region and use case before any production deployment.

## License

No license file is currently included. Add an explicit license before distributing or using this project outside its intended development context.
