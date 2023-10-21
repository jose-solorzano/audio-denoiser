import datetime
import os

from config import MAIN_BUCKET_NAME
from helpers.code_upload_helper import zip_and_upload_to_gcs

if __name__ == '__main__':
    code_dir = os.path.join(os.path.dirname(__file__), '..')
    ts_file_path = os.path.join(code_dir, '.timestamp.txt')
    with open(ts_file_path, 'w') as fd:
        timestamp = str(datetime.datetime.now())
        print(f'Timestamp: {timestamp}')
        fd.write(timestamp)
    zip_and_upload_to_gcs(code_dir, bucket_name=MAIN_BUCKET_NAME, no_cache=True)
    print('Done.')
