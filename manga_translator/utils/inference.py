import os
import stat
import sys
import tempfile
import re
import torch
import shutil
import filecmp
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

from .general import (
    BASE_PATH,
    download_url_with_progressbar,
    prompt_yes_no,
    replace_prefix,
    get_digest,
    get_filename_from_url,
)
from .log import get_logger


class InfererModule(ABC):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        super().__init__()

# class InfererModuleManager(ABC):
#     _KEY = ''
#     _VARIANTS = []

#     def __init__(self):
#         self.onstart: Callable = None
#         self.onfinish: Callable = None

#     def validate(self):
#         """
#         Throws exception if a 
#         """
#         ...

#     async def prepare(self):
#         ...

#     async def dispatch(self):
#         ...


class ModelVerificationException(Exception):
    pass

class InvalidModelMappingException(ValueError):
    def __init__(self, cls: str, map_key: str, error_msg: str):
        error = f'[{cls}->{map_key}] Invalid _MODEL_MAPPING - {error_msg}'
        super().__init__(error)

class ModelWrapper(ABC):
    r"""
    A class that provides a unified interface for downloading models and making forward passes.
    All model inferer classes should extend it.

    Download specifications can be made through overwriting the `_MODEL_MAPPING` property.

    ```python
    _MODEL_MAPPTING = {
        'model_id': {
            **PARAMETERS
        },
        ...
    }
    ```

    Parameters:

    model_id            - Used for temporary caches and debug messages

        url                 - A direct download url

        hash                - Hash of downloaded file, Can be obtained upon ModelVerificationException

        file                - File download destination, If set to '.' the filename will be infered
                              from the url (fallback is `model_id` value)

        archive             - Dict that contains all files/folders that are to be extracted from
                              the downloaded archive and their destinations, Mutually exclusive with `file`

        executables         - List of files that need to have the executable flag set
    """
    _MODEL_DIR = os.path.join(BASE_PATH, 'models')
    _MODEL_MAPPING = {}
    _KEY = ''

    def __init__(self):
        os.makedirs(self._MODEL_DIR, exist_ok=True)
        self._key = self._KEY or self.__class__.__name__
        self._loaded = False
        self._check_for_malformed_model_mapping()
        self._downloaded = self._check_downloaded()

    def is_loaded(self) -> bool:
        return self._loaded

    def is_downloaded(self) -> bool:
        return self._downloaded

    def _get_file_path(self, relative_path: str) -> str:
        return os.path.join(self._MODEL_DIR, relative_path)

    def _get_used_gpu_memory(self) -> bool:
        '''
        Gets the total amount of GPU memory used by model (Can be used in the future
        to determine whether a model should be loaded into vram or ram or automatically choose a model size).
        TODO: Use together with `--use-cuda-limited` flag to enforce stricter memory checks
        '''
        return torch.cuda.mem_get_info()

    def _check_for_malformed_model_mapping(self):
        for map_key, mapping in self._MODEL_MAPPING.items():
            if 'url' not in mapping:
                raise InvalidModelMappingException(self._key, map_key, 'Missing url property')
            elif not re.search(r'^https?://', mapping['url']):
                raise InvalidModelMappingException(self._key, map_key, 'Malformed url property: "%s"' % mapping['url'])
            if 'file' not in mapping and 'archive' not in mapping:
                mapping['file'] = '.'
            elif 'file' in mapping and 'archive' in mapping:
                raise InvalidModelMappingException(self._key, map_key, 'Properties file and archive are mutually exclusive')

    async def _download_file(self, url: str, path: str):
        print(f' -- Downloading: "{url}"')
        download_url_with_progressbar(url, path)

    async def _verify_file(self, sha256_pre_calculated: str, path: str):
        print(f' -- Verifying: "{path}"')
        sha256_calculated = get_digest(path).lower()
        sha256_pre_calculated = sha256_pre_calculated.lower()

        if sha256_calculated != sha256_pre_calculated:
            self._on_verify_failure(sha256_calculated, sha256_pre_calculated)
        else:
            print(' -- Verifying: OK!')

    def _on_verify_failure(self, sha256_calculated: str, sha256_pre_calculated: str):
        print(f' -- Mismatch between downloaded and created hash: "{sha256_calculated}" <-> "{sha256_pre_calculated}"')
        raise ModelVerificationException()

    @cached_property
    def _temp_working_directory(self):
        p = os.path.join(tempfile.gettempdir(), 'manga-image-translator', self._key.lower())
        os.makedirs(p, exist_ok=True)
        return p

    async def download(self, force=False):
        '''
        Downloads required models.
        '''
        if force or not self.is_downloaded():
            while True:
                try:
                    await self._download()
                    self._downloaded = True
                    break
                except ModelVerificationException:
                    if not prompt_yes_no('Failed to verify signature. Do you want to restart the download?', default=True):
                        print('Aborting.', end='')
                        raise KeyboardInterrupt()

    async def _download(self):
        '''
        Downloads models as defined in `_MODEL_MAPPING`. Can be overwritten (together
        with `_check_downloaded`) to implement unconventional download logic.
        '''
        print('\nDownloading models into ./models\n')
        for map_key, mapping in self._MODEL_MAPPING.items():
            if self._check_downloaded_map(map_key):
                print(f' -- Skipping {map_key} as it\'s already downloaded')
                continue

            is_archive = 'archive' in mapping
            if is_archive:
                download_path = os.path.join(self._temp_working_directory, map_key, '')
            else:
                download_path = self._get_file_path(mapping['file'])
            if not os.path.basename(download_path):
                os.makedirs(download_path, exist_ok=True)
            if os.path.basename(download_path) in ('', '.'):
                download_path = os.path.join(download_path, get_filename_from_url(mapping['url'], map_key))
            if not is_archive:
                download_path += '.part'

            if 'hash' in mapping:
                downloaded = False
                if os.path.isfile(download_path):
                    try:
                        print(' -- Found existing file')
                        await self._verify_file(mapping['hash'], download_path)
                        downloaded = True
                    except ModelVerificationException:
                        print(' -- Resuming interrupted download')
                if not downloaded:
                    await self._download_file(mapping['url'], download_path)
                    await self._verify_file(mapping['hash'], download_path)
            else:
                await self._download_file(mapping['url'], download_path)

            if download_path.endswith('.part'):
                p = download_path[:len(download_path)-5]
                shutil.move(download_path, p)
                download_path = p

            if is_archive:
                extracted_path = os.path.join(os.path.dirname(download_path), 'extracted')
                print(f' -- Extracting files')
                shutil.unpack_archive(download_path, extracted_path)

                def get_real_archive_files():
                    archive_files = []
                    for root, dirs, files in os.walk(extracted_path):
                        for name in files:
                            file_path = replace_prefix(os.path.join(root, name), extracted_path, '')
                            archive_files.append(file_path)
                    return archive_files

                extracted_path = Path(extracted_path)
                # Move every specified file from archive to destination
                for orig, dest in mapping['archive'].items():
                    for p1 in extracted_path.glob(orig): # Handle patterns such as *
                        p1 = str(p1)
                        if os.path.exists(p1):
                            p2 = self._get_file_path(dest)
                            if os.path.basename(p2) in ('', '.'):
                                p2 = os.path.join(p2, os.path.basename(p1))
                            if os.path.isfile(p2):
                                if filecmp.cmp(p1, p2):
                                    continue
                                raise InvalidModelMappingException(self._key, map_key, 'File "{orig}" already exists at "{dest}"')
                            os.makedirs(os.path.dirname(p2), exist_ok=True)
                            shutil.move(p1, p2)
                        else:
                            raise InvalidModelMappingException(self._key, map_key, f'File "{orig}" does not exist within archive' +
                                        '\nAvailable files:\n%s' % '\n'.join(get_real_archive_files()))
                if len(mapping['archive']) == 0:
                    raise InvalidModelMappingException(self._key, map_key, 'No archive files specified' +
                                        '\nAvailable files:\n%s' % '\n'.join(get_real_archive_files()))

                self._grant_execute_permissions(map_key)

            print()
            self._on_download_finished(map_key)

    def _on_download_finished(self, map_key):
        '''
        Can be overwritten to further process the downloaded files
        '''
        pass

    def _check_downloaded(self) -> bool:
        '''
        Scans filesystem for required files as defined in `_MODEL_MAPPING`.
        Returns `False` if files should be redownloaded.
        '''
        for map_key in self._MODEL_MAPPING:
            if not self._check_downloaded_map(map_key):
                return False
        return True

    def _check_downloaded_map(self, map_key: str) -> str:
        mapping = self._MODEL_MAPPING[map_key]

        if 'file' in mapping:
            path = mapping['file']
            if os.path.basename(path) in ('.', ''):
                path = os.path.join(path, get_filename_from_url(mapping['url'], map_key))
            if not os.path.exists(self._get_file_path(path)):
                return False

        elif 'archive' in mapping:
            for orig, dest in mapping['archive'].items():
                if not os.path.exists(self._get_file_path(dest)):
                    return False

        self._grant_execute_permissions(map_key)

        return True

    def _grant_execute_permissions(self, map_key: str):
        mapping = self._MODEL_MAPPING[map_key]

        if sys.platform == 'linux':
            # Grant permission to executables
            for file in mapping.get('executables', []):
                p = self._get_file_path(file)
                if os.path.basename(p) in ('', '.'):
                    p = os.path.join(p, file)
                if not os.path.isfile(p):
                    raise InvalidModelMappingException(self._key, map_key, f'File "{file}" does not exist')
                if not os.access(p, os.X_OK):
                    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    async def reload(self, device: str, *args, **kwargs):
        await self.unload()
        await self.load(*args, **kwargs, device=device)

    async def load(self, device: str, *args, **kwargs):
        '''
        Loads models into memory. Has to be called before `forward`.
        '''
        if not self.is_downloaded():
            await self.download()
        if not self.is_loaded():
            await self._load(*args, **kwargs, device=device)
            self._loaded = True

    async def unload(self):
        if self.is_loaded():
            await self._unload()
            self._loaded = False

    async def forward(self, *args, **kwargs):
        '''
        Makes a forward pass through the network.
        '''
        if not self.is_loaded():
            raise Exception(f'{self._key}: Tried to forward pass without having loaded the model.')
        return await self._forward(*args, **kwargs)

    @abstractmethod
    async def _load(self, device: str, *args, **kwargs):
        pass

    @abstractmethod
    async def _unload(self):
        pass

    @abstractmethod
    async def _forward(self, *args, **kwargs):
        pass
