import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from services.data_service import prepare_dataframe
from services.ml_service import fit_model
from state import STATE

router = APIRouter(prefix="/api", tags=["Dataset"])

MAX_UPLOAD_BYTES = 30 * 1024 * 1024


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    content = await file.read()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="CSV file exceeds the 30 MB limit.",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV is empty.",
        )

    try:
        raw_df = pd.read_csv(io.BytesIO(content))

        # Validate before replacing the active dataset.
        prepare_dataframe(raw_df)

        # fit_model updates shared state only on success.
        result = fit_model(raw_df, source=file.filename)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process CSV: {exc}",
        ) from exc

    return {
        "message": "Dataset processed successfully.",
        "filename": file.filename,
        **result,
    }
