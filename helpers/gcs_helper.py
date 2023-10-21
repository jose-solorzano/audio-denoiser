import logging
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any, List

from google.cloud import storage
from tqdm import tqdm

from config import LOCAL_MAIN_BUCKET_DIR, GOOGLE_KEY_FILE, MAIN_BUCKET_NAME, GCS_PROJECT

_google_key_file = os.path.join(os.path.dirname(__file__), '..', 'cred', GOOGLE_KEY_FILE)
_google_project = GCS_PROJECT
_warned_gcs_key = False


def get_storage_client():
    global _warned_gcs_key
    if not os.path.exists(_google_key_file):
        if not _warned_gcs_key:
            logging.warning('GCS key file not found. Assuming access exists.')
            _warned_gcs_key = True
        return storage.Client(project=_google_project)
    else:
        return storage.Client.from_service_account_json(_google_key_file)


def upload_byte_array_to_gcs_bucket(bucket_name, file_path, byte_array):
    # Authenticate with Google Cloud using a service account key file
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)

    # Create a blob (file) in the bucket with the specified name and upload the byte array
    blob = bucket.blob(file_path)
    blob.upload_from_string(byte_array, content_type='application/octet-stream')

    # Return the public URL of the uploaded file
    return blob.public_url


def pickle_and_upload_to_gcs_bucket(obj: Any, file_path: str, bucket_name: str):
    # Pickle the object
    byte_array = pickle.dumps(obj)

    # Upload the pickled data to the specified file in the Google Cloud Storage bucket
    return upload_byte_array_to_gcs_bucket(bucket_name, file_path, byte_array)


def download_and_unzip_from_gcs(gcs_path: str, path_local_to_folder: str, bucket_name: str):
    # Initialize the GCS client
    storage_client = get_storage_client()

    # Get the GCS bucket and file name
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    # Create a temporary file to download the ZIP file to
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_name = temp_file.name

        # Download the ZIP file from GCS to the temporary file
        blob.download_to_filename(temp_file_name)

        # Extract the ZIP file to the specified local folder
        with zipfile.ZipFile(temp_file_name, 'r') as zip_ref:
            zip_ref.extractall(path_local_to_folder)

    # Remove the temporary file
    os.remove(temp_file_name)


def download_file_from_gcs(gcs_path: str, local_file_path: str, bucket_name: str):
    # Initialize the GCS client
    storage_client = get_storage_client()

    # Get the GCS bucket and file name
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    if blob.size == 0:
        raise ValueError(f'Blob {gcs_path} is empty file')

    parent_path = str(Path(local_file_path).parent)
    os.makedirs(parent_path, exist_ok=True)
    with open(local_file_path, 'wb') as f:
        with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
            # blob.download_to_file is deprecated
            storage_client.download_blob_to_file(blob, file_obj)


def download_gcs_folder(gcs_path: str, path_local_to_folder: str, bucket_name: str, skip_if_exists: bool = False):
    """Download all files from a GCS folder to a local folder"""

    # Initialize the GCS client
    storage_client = get_storage_client()

    # Get the GCS bucket and list all the files in the folder
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_path)
    # Download each file from GCS to the specified local folder
    for blob in blobs:
        file_path = os.path.join(path_local_to_folder, blob.name)
        if skip_if_exists and os.path.exists(file_path):
            print(f'Not downloading {blob.name} as it already exists.')
        else:
            parent_folder = Path(file_path).parent
            os.makedirs(parent_folder, exist_ok=True)
            blob.download_to_filename(file_path)
            print(f'Downloaded {file_path}')


def ensure_main_bucket_dirs_exist(dir_names: List[str]):
    for req_dir_name in dir_names:
        download_gcs_folder(req_dir_name, LOCAL_MAIN_BUCKET_DIR, bucket_name=MAIN_BUCKET_NAME, skip_if_exists=True)
