from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET", "buiduchieu-ai20k-lab21")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """
    Tai file model.joblib tu cloud storage ve may khi server khoi dong.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.download_file(ARTIFACT_BUCKET, MODEL_KEY, MODEL_PATH)
        print(f"Model downloaded successfully from S3 {ARTIFACT_BUCKET}/{MODEL_KEY} to {MODEL_PATH}")
    except Exception as e:
        print(f"AWS S3 download notice: {e}")
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(ARTIFACT_BUCKET)
            blob = bucket.blob(MODEL_KEY)
            blob.download_to_filename(MODEL_PATH)
            print("Model downloaded from GCP storage.")
        except Exception as gcp_err:
            print(f"GCP download notice: {gcp_err}")


# Thu tai va load model khi khoi dong
download_model()
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f10]}
    Dau ra  : JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}

    Thu tu 10 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        age, workclass, education_num, marital_status, occupation,
        relationship, sex, capital_gain, capital_loss, hours_per_week
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            download_model()
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
            else:
                raise HTTPException(status_code=500, detail="Model file not found")

    # Kiem tra so luong dac trung
    if len(req.features) != 10:
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")

    # Goi model.predict de lay ket qua du doan
    pred = model.predict([req.features])
    prediction_int = int(pred[0])
    label = "thu_nhap_cao" if prediction_int == 1 else "thu_nhap_thap"

    return {"prediction": prediction_int, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
