"""
OSS File Client for handling file operations with Alibaba Cloud OSS
Supports downloading files/folders, listing directories, and uploading files
"""

import hashlib
import os
import shutil
import signal
import tempfile
import threading
import logging
import random
import string
from typing import List, Optional, Union
from pathlib import Path
try:
    from aoss_client.client import Client as PetrelClient
except ImportError:
    from petrel_client.client import Client as PetrelClient


class FileClient:
    """
    A file client for handling OSS operations with local temporary file management.
    
    This class provides methods to:
    - Download files from OSS to local temporary storage
    - Download entire folders from OSS
    - List contents of OSS directories
    - Upload local files to OSS
    - Manage temporary files and cleanup
    """
    
    def __init__(self, temp_dir: Optional[str] = None, auto_cleanup: bool = True):
        """
        Initialize the FileClient.
        
        Args:
            temp_dir: Directory for temporary files. If None, uses system temp directory.
            auto_cleanup: Whether to automatically cleanup temporary files on exit.
        """
        self.client = PetrelClient()
        self.auto_cleanup = auto_cleanup
        
        if temp_dir is None:
            #temp_dir = os.path.join(os.path.expanduser('~'), '.temp_swap')
            temp_dir = os.path.join('/data/kliao/data/', '.temp_swap')
        
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Track created temporary files for cleanup
        self._temp_files = set()
        self._temp_dirs = set()

        # Set process-specific random seed for unique random strings across different dataloader processes
        process_id = os.getpid()
        random.seed(process_id)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)
        
    def _get_temp_path(self, oss_path: str, suffix: str = "") -> Path:
        """
        Generate a temporary file path based on OSS path.
        
        Args:
            oss_path: The OSS file path
            suffix: Optional suffix to add to the filename
            
        Returns:
            Path to temporary file
        """
        # Create hash from OSS path to avoid conflicts
        hash_str = hashlib.md5(oss_path.encode('utf-8')).hexdigest()
        filename = os.path.basename(oss_path)

        # Add suffix if provided
        if suffix:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{suffix}{ext}"

        # Prefix with the PID so each dataloader worker gets its OWN temp file
        # for the same oss_path. Without this, all workers share one
        # deterministic path and one worker's cleanup_temp_files() can delete
        # a file another worker is still reading -> "No such file or directory".
        return self.temp_dir / f"{os.getpid()}_{hash_str}_{filename}"

    def _get_with_timeout(self, oss_path: str):
        """self.client.get() with a wall-clock timeout, so a HUNG AOSS read raises
        (TimeoutError) instead of blocking the dataloader worker forever -- a hung
        worker stalls its rank and shows up as an NCCL collective timeout (or the
        resume fast-forward hanging for hours). The caller's dataset __getitem__
        catches the exception and retries the next sample. Uses SIGALRM, which
        only works in a process main thread (DataLoader workers / num_workers=0);
        otherwise falls back to a plain get(). Configure via env
        AOSS_DOWNLOAD_TIMEOUT seconds (default 120; 0 disables)."""
        timeout_sec = int(os.environ.get('AOSS_DOWNLOAD_TIMEOUT', '120'))
        if timeout_sec <= 0 or threading.current_thread() is not threading.main_thread():
            return self.client.get(oss_path)

        def _on_timeout(signum, frame):
            raise TimeoutError(f"AOSS get() exceeded {timeout_sec}s: {oss_path}")

        old = signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(timeout_sec)
        try:
            return self.client.get(oss_path)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def download_file(self, oss_path: str, force_redownload: bool = False) -> str:
        """
        Download a single file from OSS to local temporary storage.
        
        Args:
            oss_path: Path to file in OSS (e.g., "aoss:s3://bucket/path/file.jpg")
            force_redownload: If True, download even if file already exists locally
            
        Returns:
            Path to the downloaded temporary file
            
        Raises:
            Exception: If download fails
        """
        temp_path = self._get_temp_path(oss_path)
        
        # Check if file already exists and we don't want to redownload
        if temp_path.exists() and not force_redownload:
            self.logger.info(f"File already exists locally: {temp_path}")
            return str(temp_path)
        
        try:
            self.logger.info(f"Downloading file from OSS: {oss_path}")
            file_bytes = self._get_with_timeout(oss_path)
            if file_bytes is None:
                raise FileNotFoundError(f"OSS object not found: {oss_path}")

            # Defensive: ensure the temp dir still exists (it can be wiped
            # mid-run by external cleanup / a vanished mount), else open('wb')
            # below raises ENOENT on the parent directory.
            temp_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)
            
            # Track for cleanup
            self._temp_files.add(str(temp_path))
            
            self.logger.info(f"Successfully downloaded to: {temp_path}")
            return str(temp_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download file {oss_path}: {str(e)}")
            raise
    
    def download_folder(self, oss_folder_path: str, force_redownload: bool = False) -> str:
        """
        Download an entire folder from OSS to local temporary storage.
        
        Args:
            oss_folder_path: Path to folder in OSS (e.g., "aoss:s3://bucket/path/folder/")
            force_redownload: If True, download even if folder already exists locally
            
        Returns:
            Path to the downloaded temporary folder
            
        Raises:
            Exception: If download fails
        """
        # Ensure folder path ends with '/'
        if not oss_folder_path.endswith('/'):
            oss_folder_path += '/'
        
        # Create temporary folder
        hash_str = hashlib.md5(oss_folder_path.encode('utf-8')).hexdigest()
        folder_name = os.path.basename(oss_folder_path.rstrip('/'))
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        temp_folder = self.temp_dir / f"{hash_str}_{random_str}_{folder_name}"
        
        # Check if folder already exists and we don't want to redownload
        if temp_folder.exists() and not force_redownload:
            self.logger.info(f"Folder already exists locally: {temp_folder}")
            return str(temp_folder)
        
        try:
            # Create temporary folder
            temp_folder.mkdir(parents=True, exist_ok=True)
            self._temp_dirs.add(str(temp_folder))
            
            # List all files in the folder
            file_list = self.list_dir(oss_folder_path, recursive=True)
            
            self.logger.info(f"Downloading folder from OSS: {oss_folder_path}")
            self.logger.info(f"Found {len(file_list)} files to download")
            
            # Download each file
            for file_path in file_list:
                # Get relative path within the folder
                relative_path = file_path[len(oss_folder_path):]
                local_file_path = temp_folder / relative_path
                
                # Create subdirectories if needed
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                file_bytes = self.client.get(file_path)
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                
                self._temp_files.add(str(local_file_path))
            
            self.logger.info(f"Successfully downloaded folder to: {temp_folder}")
            return str(temp_folder)
            
        except Exception as e:
            self.logger.error(f"Failed to download folder {oss_folder_path}: {str(e)}")
            # Cleanup partial download
            if temp_folder.exists():
                shutil.rmtree(temp_folder)
            raise
    
    def list_dir(self, oss_path: str, recursive: bool = False, return_only_dir: bool = False) -> List[str]:
        """
        List contents of an OSS directory, similar to os.listdir.
        
        Args:
            oss_path: Path to directory in OSS (e.g., "aoss:s3://bucket/path/")
            recursive: If True, list files recursively in subdirectories
            return_only_dir: If True, return only directories; if False, return all items
            
        Returns:
            List of file/folder names in the directory (not full paths)
            
        Raises:
            Exception: If listing fails
        """
        try:
            self.logger.info(f"Listing directory: {oss_path}")
            
            # Ensure path ends with '/' for directory listing
            if not oss_path.endswith('/'):
                oss_path += '/'
            
            # Use petrel client to list directory
            if recursive:
                # For recursive listing, we need to get all files
                file_list = []
                items = self.client.list(oss_path)
                for item in items:
                    if item.endswith('/'):
                        # It's a directory, list recursively
                        sub_files = self.list_dir(item, recursive=True, return_only_dir=return_only_dir)
                        file_list.extend(sub_files)
                    else:
                        # It's a file, get just the filename
                        if not return_only_dir:
                            filename = os.path.basename(item)
                            file_list.append(filename)
                return file_list
            else:
                # Non-recursive listing - return just filenames/foldernames
                items = self.client.list(oss_path)
                result = []
                for item in items:
                    if item.endswith('/'):
                        # It's a directory, get just the folder name
                        folder_name = os.path.basename(item.rstrip('/'))
                        result.append(folder_name)
                    else:
                        # It's a file, get just the filename
                        if not return_only_dir:
                            filename = os.path.basename(item)
                            result.append(filename)
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to list directory {oss_path}: {str(e)}")
            raise
        
    def is_dir(self, oss_path: str) -> bool:
        """
        Check if a path in OSS is a 'directory'-like prefix.

        Args:
            oss_path: Path in OSS (e.g. "aoss:s3://bucket/path/" or without trailing '/')

        Returns:
            True if path behaves like a directory (has children), False otherwise
        """
        try:
            if not oss_path.endswith('/'):
                oss_path += '/'

            items = self.client.list(oss_path)
            print(items)
            return len(items) > 0
        except Exception:
            return False
        

    
    def exists(self, oss_path: str) -> bool:
        """
        Check if a file exists in OSS.
        
        Args:
            oss_path: Path to file in OSS
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            # Try to get file info to check if it exists
            self.client.get(oss_path)
            return self.client.get(oss_path) != None
        except:
            return False
    
    def upload_file(self, local_path: str, oss_path: str) -> bool:
        """
        Upload a local file to OSS.
        
        Args:
            local_path: Path to local file to upload
            oss_path: Destination path in OSS
            
        Returns:
            True if upload successful, False otherwise
            
        Raises:
            Exception: If upload fails
        """
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            self.logger.info(f"Uploading file to OSS: {local_path} -> {oss_path}")
            
            # Read local file
            with open(local_path, 'rb') as f:
                file_bytes = f.read()
            
            # Upload to OSS
            self.client.put(oss_path, file_bytes)
            
            self.logger.info(f"Successfully uploaded to: {oss_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload file {local_path} to {oss_path}: {str(e)}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up all temporary files and directories created by this client."""
        try:
            # Remove temporary files
            for temp_file in self._temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"Removed temporary file: {temp_file}")
            
            # Remove temporary directories
            for temp_dir in self._temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"Removed temporary directory: {temp_dir}")
            
            # Clear tracking sets
            self._temp_files.clear()
            self._temp_dirs.clear()
            
            self.logger.info("Cleaned up all temporary files and directories")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def get_temp_file_info(self) -> dict:
        """
        Get information about currently tracked temporary files.
        
        Returns:
            Dictionary with information about temp files and directories
        """
        return {
            'temp_files': list(self._temp_files),
            'temp_dirs': list(self._temp_dirs),
            'temp_dir': str(self.temp_dir),
            'total_files': len(self._temp_files),
            'total_dirs': len(self._temp_dirs)
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.auto_cleanup:
            self.cleanup_temp_files()
    
    def __del__(self):
        """Destructor with cleanup."""
        if self.auto_cleanup:
            try:
                self.cleanup_temp_files()
            except:
                pass  # Ignore errors during cleanup in destructor


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    with FileClient() as client:
        # Download a single file
        try:
            temp_file = client.download_file("aoss:s3://yhluo_sgacer/data/tracking/processed_mvs_synth/0000/rgb/0000.jpg")
            print(f"Downloaded file to: {temp_file}")
        except Exception as e:
            print(f"Error downloading file: {e}")
        
        # List directory contents
        try:
            print(client.is_dir("aoss:s3://yhluo_sgacer/data/tracking/processed_dl3dv_ours_parts/processed_dl3dv_ours/"))
            files = client.list_dir("aoss:s3://yhluo_sgacer/data/tracking/processed_mvs_synth/0000/rgb/")
            print(f"Found {len(files)} files in directory")
            for file in files[:5]:  # Show first 5 files
                print(f"  - {file}")
        except Exception as e:
            print(f"Error listing directory: {e}")
        
        # Get temp file info
        info = client.get_temp_file_info()
        print(f"Temp file info: {info}")
        
        # Cleanup happens automatically when exiting the context