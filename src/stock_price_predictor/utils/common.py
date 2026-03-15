import os
import pickle
import io
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    if data is None:
        return {}
    return data


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_object(path: str, obj: Any) -> None:
    if path.startswith("gs://"):
        data = pickle.dumps(obj)
        bucket_name, blob_name = parse_gcs_uri(path)
        upload_bytes_to_gcs(data=data, bucket_name=bucket_name, destination_blob=blob_name, content_type="application/octet-stream")
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_object(path: str) -> Any:
    if path.startswith("gs://"):
        bucket_name, blob_name = parse_gcs_uri(path)
        data = download_bytes_from_gcs(bucket_name=bucket_name, blob_name=blob_name)
        return pickle.loads(data)
    with open(path, "rb") as file:
        return pickle.load(file)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a valid GCS URI: {uri}")
    no_scheme = uri[len("gs://") :]
    parts = no_scheme.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"GCS URI must be in format gs://bucket/blob, got: {uri}")
    return parts[0], parts[1]


def _get_storage_client():
    try:
        from google.cloud import storage
    except Exception as error:
        raise ImportError("google-cloud-storage is required for GCS operations.") from error
    load_dotenv()
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip().strip('"').strip("'")
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(
                "GOOGLE_APPLICATION_CREDENTIALS path from environment/.env does not exist: "
                f"{credentials_path}"
            )
    return storage.Client()


def read_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()
    return pd.read_csv(io.StringIO(data))


def upload_to_gcs(df: pd.DataFrame, bucket_name: str, destination_blob: str):
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")


def gcs_blob_exists(bucket_name: str, blob_name: str) -> bool:
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()


def upload_bytes_to_gcs(data: bytes, bucket_name: str, destination_blob: str, content_type: str = "application/octet-stream"):
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_string(data, content_type=content_type)


def download_bytes_from_gcs(bucket_name: str, blob_name: str) -> bytes:
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()
