import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal

import shutil
import uuid

from agent.reports import DISCLAIMER, create_prediction_report
from agent.service import get_agent_response
from RAG.scripts import answer_from_documents


app = FastAPI(
    title="X-Ray Fracture Detection API"
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# Static folder
app.mount(
    "/static",
    StaticFiles(
        directory=os.path.join(BASE_DIR, "static")
    ),
    name="static"
)

REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_FOLDER, exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORT_FOLDER), name="reports")


def build_report(title: str, prediction: str, confidence: float) -> str:
    """Ground each model result in RAG, while keeping prediction usable offline."""
    try:
        rag_answer = answer_from_documents(
            f"Provide patient-friendly information, key considerations, and next steps related to: {prediction}. "
            "Do not diagnose and include the medical disclaimer."
        ).get("answer", "")
    except Exception as exc:
        print("REPORT RAG ERROR:", exc)
        rag_answer = "Retrieved guidance is temporarily unavailable. " + DISCLAIMER
    return create_prediction_report(REPORT_FOLDER, title, prediction, confidence, rag_answer)


# Templates
templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)


UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads"
)

class LiverData(BaseModel):

    model: Literal[
        "random_forest",
        "logistic_regression",
        "xgboost"
    ] = "random_forest"

    Age: float = Field(
        ...,
        ge=1,
        le=120,
        description="Age must be between 1 and 120"
    )


    Gender: int = Field(
        ...,
        ge=0,
        le=1,
        description="0 Female, 1 Male"
    )


    Total_Bilirubin: float = Field(
        ...,
        ge=0,
        le=100
    )


    Direct_Bilirubin: float = Field(
        ...,
        ge=0,
        le=50
    )


    Alkaline_Phosphotase: float = Field(
        ...,
        ge=0,
        le=5000
    )


    Alamine_Aminotransferase: float = Field(
        ...,
        ge=0,
        le=5000
    )


    Aspartate_Aminotransferase: float = Field(
        ...,
        ge=0,
        le=5000
    )


    Total_Protiens: float = Field(
        ...,
        ge=0,
        le=20
    )


    Albumin: float = Field(
        ...,
        ge=0,
        le=10
    )


    Albumin_and_Globulin_Ratio: float = Field(
        ...,
        ge=0,
        le=5
    )

@app.get(
    "/",
    response_class=HTMLResponse
)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )



@app.post(
    "/predict",
    response_class=HTMLResponse
)
async def predict(
    request: Request,
    file: UploadFile = File(...)
):

    file_name = (
        str(uuid.uuid4())
        +
        "_"
        +
        file.filename
    )


    file_path = os.path.join(
        UPLOAD_FOLDER,
        file_name
    )


    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )
    from deep_learning.scripts import predict_fracture

    result = predict_fracture(
        file_path
    )

    report_name = build_report(
        "CURA AI X-Ray Fracture Analysis Report",
        result["fracture_type"],
        result["confidence"],
    )




    return JSONResponse(
    {
        "fracture_type": result["fracture_type"],
        "confidence": result["confidence"],
        "image": "/uploads/" + file_name,
        "report": "/reports/" + report_name,
    }
)

@app.post(
    "/predict_mri",
    response_class=HTMLResponse
)
async def predict_mri_image(

    request: Request,

    file: UploadFile = File(...)

):

    file_name = (
        str(uuid.uuid4())
        +
        "_"
        +
        file.filename
    )


    file_path = os.path.join(
        UPLOAD_FOLDER,
        file_name
    )


    # Save image

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )
    from deep_learning2.scripts import predict_mri

    result = predict_mri(
        file_path
    )

    report_name = build_report(
        "CURA AI MRI Analysis Report",
        result["mri_type"],
        result["confidence"],
    )


    return JSONResponse(
    {
        "mri_type": result["mri_type"],
        "confidence": result["confidence"],
        "image": "/uploads/" + file_name,
        "report": "/reports/" + report_name,
    }
)

@app.post("/liver_predict")
async def liver_predict(
    data: LiverData
):
    from machine_learning.scripts import predict_liver

    result = predict_liver(
        data.dict()
    )

    report_name = build_report(
        "CURA AI Liver Disease Prediction Report",
        result["result"],
        result["confidence"],
    )


    return {

        "status": "success",

        "prediction": result["result"],

        "confidence": result["confidence"],

        "model_used": result["model_used"],
        "report": "/reports/" + report_name,

    }
class ChatRequest(BaseModel):

    question:str




@app.post("/rag_query")
async def rag_query(
    data: ChatRequest
):

    try:

        result = get_agent_response(
            data.question
        )

        return result


    except Exception as e:

        print("RAG ERROR:", e)

        return {
            "error": str(e)
        }
