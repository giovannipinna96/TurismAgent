"""
Data management utilities for the tourist guide agent.
Handles file operations, directory management, and data persistence.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data operations including file I/O, directory operations,
    and data persistence for the tourist guide system.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the DataManager with a base directory.
        
        Args:
            base_dir: Base directory for all data operations
        """
        self.base_dir = Path(base_dir)
        self.ensure_directory(self.base_dir)
        
        # Create standard subdirectories
        self.images_dir = self.base_dir / "images"
        self.segmented_dir = self.base_dir / "img_segm"
        self.metadata_dir = self.base_dir / "metadata"
        self.database_dir = self.base_dir / "database"
        
        for dir_path in [self.images_dir, self.segmented_dir, 
                        self.metadata_dir, self.database_dir]:
            self.ensure_directory(dir_path)

    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, create if it doesn't.
        
        Args:
            path: Directory path to ensure
            
        Returns:
            Path object of the directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {path}")
        return path

    def save_file(self, data: Any, filepath: Union[str, Path], 
                  format_type: str = "json") -> bool:
        """
        Save data to file in specified format.
        
        Args:
            data: Data to save
            filepath: Path where to save the file
            format_type: Format type ('json', 'pickle', 'text')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format_type == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            elif format_type == "text":
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
            logger.info(f"File saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving file {filepath}: {e}")
            return False

    def load_file(self, filepath: Union[str, Path], 
                  format_type: str = "json") -> Optional[Any]:
        """
        Load data from file.
        
        Args:
            filepath: Path to the file to load
            format_type: Format type ('json', 'pickle', 'text')
            
        Returns:
            Loaded data or None if error
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.warning(f"File does not exist: {filepath}")
                return None
                
            if format_type == "json":
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif format_type == "pickle":
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            elif format_type == "text":
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = f.read()
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
            logger.info(f"File loaded: {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            return None

    def save_image(self, image: Image.Image, filepath: Union[str, Path], 
                   format_type: str = "PNG") -> bool:
        """
        Save PIL Image to file.
        
        Args:
            image: PIL Image object
            filepath: Path where to save the image
            format_type: Image format (PNG, JPEG, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            image.save(filepath, format=format_type)
            logger.info(f"Image saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving image {filepath}: {e}")
            return False

    def load_image(self, filepath: Union[str, Path]) -> Optional[Image.Image]:
        """
        Load image from file.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            PIL Image object or None if error
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.warning(f"Image file does not exist: {filepath}")
                return None
                
            image = Image.open(filepath)
            logger.info(f"Image loaded: {filepath}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {filepath}: {e}")
            return None

    def delete_file(self, filepath: Union[str, Path]) -> bool:
        """
        Delete a file.
        
        Args:
            filepath: Path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            if filepath.exists():
                filepath.unlink()
                logger.info(f"File deleted: {filepath}")
                return True
            else:
                logger.warning(f"File does not exist: {filepath}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file {filepath}: {e}")
            return False

    def delete_directory(self, dirpath: Union[str, Path], 
                        force: bool = False) -> bool:
        """
        Delete a directory and its contents.
        
        Args:
            dirpath: Path to the directory to delete
            force: If True, delete even if not empty
            
        Returns:
            True if successful, False otherwise
        """
        try:
            dirpath = Path(dirpath)
            if dirpath.exists():
                if force:
                    shutil.rmtree(dirpath)
                else:
                    dirpath.rmdir()  # Only works if empty
                logger.info(f"Directory deleted: {dirpath}")
                return True
            else:
                logger.warning(f"Directory does not exist: {dirpath}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting directory {dirpath}: {e}")
            return False

    def list_files_in_directory(self, dirpath: Union[str, Path], 
                               pattern: str = "*", 
                               recursive: bool = False) -> List[Path]:
        """
        List files in a directory.
        
        Args:
            dirpath: Directory path to search
            pattern: File pattern to match (e.g., "*.jpg")
            recursive: If True, search subdirectories
            
        Returns:
            List of file paths
        """
        try:
            dirpath = Path(dirpath)
            if not dirpath.exists():
                logger.warning(f"Directory does not exist: {dirpath}")
                return []
                
            if recursive:
                files = list(dirpath.rglob(pattern))
            else:
                files = list(dirpath.glob(pattern))
                
            # Filter out directories, keep only files
            files = [f for f in files if f.is_file()]
            
            logger.info(f"Found {len(files)} files in {dirpath}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {dirpath}: {e}")
            return []

    def list_directories(self, dirpath: Union[str, Path], 
                        recursive: bool = False) -> List[Path]:
        """
        List directories in a given path.
        
        Args:
            dirpath: Directory path to search
            recursive: If True, search subdirectories
            
        Returns:
            List of directory paths
        """
        try:
            dirpath = Path(dirpath)
            if not dirpath.exists():
                logger.warning(f"Directory does not exist: {dirpath}")
                return []
                
            if recursive:
                dirs = [p for p in dirpath.rglob("*") if p.is_dir()]
            else:
                dirs = [p for p in dirpath.iterdir() if p.is_dir()]
                
            logger.info(f"Found {len(dirs)} directories in {dirpath}")
            return dirs
            
        except Exception as e:
            logger.error(f"Error listing directories in {dirpath}: {e}")
            return []

    def get_file_info(self, filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get file information including size, modification time, etc.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary with file information or None if error
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return None
                
            stat = filepath.stat()
            info = {
                "name": filepath.name,
                "path": str(filepath),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "is_file": filepath.is_file(),
                "is_directory": filepath.is_dir(),
                "suffix": filepath.suffix
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting file info for {filepath}: {e}")
            return None

    def copy_file(self, src: Union[str, Path], 
                  dst: Union[str, Path]) -> bool:
        """
        Copy a file from source to destination.
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            src = Path(src)
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src, dst)
            logger.info(f"File copied from {src} to {dst}")
            return True
            
        except Exception as e:
            logger.error(f"Error copying file from {src} to {dst}: {e}")
            return False

    def get_directory_size(self, dirpath: Union[str, Path]) -> int:
        """
        Get total size of a directory and its contents.
        
        Args:
            dirpath: Directory path
            
        Returns:
            Total size in bytes
        """
        try:
            dirpath = Path(dirpath)
            total_size = 0
            
            for file_path in dirpath.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    
            return total_size
            
        except Exception as e:
            logger.error(f"Error calculating directory size for {dirpath}: {e}")
            return 0