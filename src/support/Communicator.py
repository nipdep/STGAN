#! usr/bin/python3

import logging
import os
from pathlib import Path

# set logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from anonfile import AnonFile
from zipfile import ZipFile


class Communicator:

    def __init__(self, log_dir :str, root_dir : str):
        """
        Input:
        log_dir [str] : directory of the url record file

        """
        self.anon = AnonFile()
        self.log_dir = log_dir
        self.root_dir = root_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def uploadFile(self, key : str, dir : str):
        if os.path.exists(dir):
            upload = self.anon.upload(dir, progressbar=True)
            with open(self.log_dir, 'a') as pf:
                pf.write(f"{key}>{upload.url.geturl()} \n")
        else:
            logger.error(f" {dir} : is not found.")

    def uploadFolder(self, key: str, dir : str):

        if os.path.isdir(dir):
            with ZipFile(f'{key}.zip', 'w') as zipObj:
                file_paths = self._get_all_file_paths(dir)
                for file in file_paths:
                    zipObj.write(file)
            self.uploadFile(key, f'{key}.zip')
        else:
            logger.error(f" {dir} : is not found.")

    def downloadFile(self, key : str, dir : str):
        with open(self.log_dir, 'r') as pf:
            for line in pf.readlines():
                key_u, url = line.split('>')
                if key == key_u:
                    filename = self.anon.download(url, os.path.abspath(dir))
                    break
            else:
                logger.error(f" {key} : is not found.")
            

    def _get_all_file_paths(self, directory: str):
        file_paths = []
        for root, directories, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
        return file_paths 

if __name__ == "__main__":
    pass
        