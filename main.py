from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse 
import pandas as pd
import json
import os 
from app.piepline import AutoPipeline
from app.trainer import train_best_model

app = FastAPI(title="AutoStreamline API", version ="1.0.0")

@app.post("/api/v1/train")
async def run_automl_pipeline(
    file: UploadFile = File(...),
    target: str = Form(...),
    text_cols: str = Form("[]")  # Expected as serialized JSON array
):
    try:
        text_columns_parsed = json.loads(text_cols)
        
        # Read incoming data stream completely into RAM safely
        df = pd.read_csv(file.file)
        
        if target not in df.columns:
            return {"error": f"Target column '{target}' not present in provided file artifact."}
            
        # Trigger the automatic preprocessing step
        preprocessor, X_train, X_test, y_train, y_test, task_type = AutoPipeline(
            df, target_column=target, text_columns=text_columns_parsed
        )
        
        # Run training loop execution 
        metrics, model_path, tech = train_best_model(
            preprocessor, X_train, X_test, y_train, y_test, task_type, output_dir="/tmp"
        )
        
        return {
            "status": "Success",
            "task_type": task_type,
            "chosen_framework": tech,
            "evaluation_metrics": metrics,
            "download_route": f"/api/v1/download?path={model_path}"
        }
    except Exception as e:
        return {"status": "Error", "detail": str(e)}

@app.get("/api/v1/download")
def download_model(path: str):
    if os.path.exists(path):
        return FileResponse(path=path, filename=os.path.basename(path), media_type='application/octet-stream')
    return {"error": "File artifact could not be resolved on local disk path."}