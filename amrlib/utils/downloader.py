import os
import logging
import tarfile
import requests
from   tqdm import tqdm

logger = logging.getLogger(__name__)


# Download the .tar.gz file from the url and place it in the data_dir
# This returns the name of the created directory, which is expected to be the same as the download
# NOTE: by convention, the directory inside of the tar file needs to be the same as the filename
def download_model(url, data_dir, rm_tar=True):
    # Setup the download filename and data directory
    fn  = os.path.join(data_dir, os.path.basename(url))
    os.makedirs(data_dir, exist_ok=True)
    # Download the file and extract it
    download_file(url, fn)
    # Extrac the tarfile
    print('Extracting ', fn)
    tf = tarfile.open(fn)
    tf.extractall(data_dir)
    tf.close()
    # Remove the tar file if requested
    if rm_tar:
        os.remove(fn)
    print('Extraction complete')
    model_dir = fn[:-len('.tar.gz')]     # should be the name of created directory
    if not os.path.isdir(model_dir):
        logger.error('Extraction error, missing model_dir %s' % model_dir)
        return None
    return model_dir


# Create a symbolic link pointing to src named dst.
def set_symlink(src, dst):
    if os.path.lexists(dst):    # remove old link it it exists
        os.remove(dst)
    os.symlink(src, dst, target_is_directory=True)
    print('Symlink set from %s to %s' % (dst, src))


# Download a URL into a file as specified by `path`.
def download_file(url, path):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        file_size = int(r.headers.get('content-length'))
        default_chunk_size = 131072
        desc = 'Downloading ' + url
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=default_chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update(len(chunk))
