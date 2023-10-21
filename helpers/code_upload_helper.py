import os
import zipfile
from pathlib import Path

from gitignore_parser import parse_gitignore
from tqdm import tqdm
from helpers.gcs_helper import get_storage_client


def zip_and_upload_to_gcs(path_to_folder: str, bucket_name: str, no_cache: bool = False, tmp_dir='/opt/tmp'):
    path_to_folder = Path(path_to_folder).resolve().as_posix()

    # Creating a ZipFile object
    base_dir_name = os.path.basename(os.path.abspath(path_to_folder))
    zip_filename = base_dir_name + ".zip"
    os.makedirs(tmp_dir, exist_ok=True)
    zip_path = os.path.join(tmp_dir, zip_filename)
    zipf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

    # Parsing the .gitignore file, if it exists
    gitignore_filename = (Path(path_to_folder) / ".gitignore").as_posix()
    if os.path.exists(gitignore_filename):
        fn_ignore_match = parse_gitignore(gitignore_filename)
    else:
        fn_ignore_match = None

    # Recursively adding files to the zip file
    file_paths = []
    for root, dirs, files in os.walk(path_to_folder):
        for file in files:
            file_path = (Path(root) / file).as_posix()
            if fn_ignore_match and fn_ignore_match(file_path):
                # Skipping files that match gitignore rules
                continue
            file_paths.append(file_path)
    for file_path in tqdm(file_paths, desc='Zipping files', unit='File'):
        zipf.write(file_path, os.path.relpath(file_path, path_to_folder))

    # Closing the ZipFile
    zipf.close()

    file_size = os.path.getsize(zip_path)
    print(f'Uploading ZIP file of size {file_size} bytes...')

    # Uploading the zip file to GCS bucket
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(zip_filename)
    if no_cache:
        blob.cache_control = 'private, no-store, no-cache, max-age=1'
    blob.upload_from_filename(zip_path)

    # Removing the zip file from local disk
    os.remove(zip_path)
