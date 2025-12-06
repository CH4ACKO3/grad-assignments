import os
import json
import uuid
import shutil
import time
import fcntl
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MetadataStorage:
    """
    JSON-based indexed file storage system with directory-level locking
    
    Features:
    1. Create UUID-named files when storing files and record metadata in index
    2. Query matching files based on metadata
    3. Support partial metadata matching queries
    4. Directory-level locking to prevent concurrent access
    """
    
    def __init__(self, storage_dir: str, fix_uuid: Optional[str] = None):
        """
        Initialize storage system
        
        Args:
            storage_dir: File storage directory
            index_file: Index file name
        """
        self.storage_dir = Path(storage_dir)
        self.index_file = self.storage_dir / "index.json"
        self.lock_file = self.storage_dir / ".lock"
        self.index_data = {}
        self.fix_uuid = fix_uuid

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _acquire_lock(self):
        """Acquire directory lock by creating a lock file"""
        max_retries = 10
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                # Create lock file with exclusive access
                # Don't use 'with' statement as we need to keep the file handle open
                f = open(self.lock_file, 'w')
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Keep file handle open to maintain lock
                self._lock_handle = f
                logger.debug("Directory lock acquired")
                self._load_index()
                return
            except BlockingIOError:
                # Lock is held by another process
                if attempt < max_retries - 1:
                    logger.debug(f"Directory is locked, retrying in {retry_delay}s (attempt {attempt + 1})")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 2.0)
                else:
                    logger.error(f"Failed to acquire directory lock after {max_retries} attempts")
                    raise RuntimeError("Failed to acquire directory lock")
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to acquire lock (attempt {attempt + 1}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to acquire lock after {max_retries} attempts: {e}")
                    raise
    
    def _release_lock(self):
        """Release directory lock"""
        try:
            if hasattr(self, '_lock_handle') and self._lock_handle:
                # Clean up orphaned files before releasing lock
                self._cleanup_orphaned_files()
                
                # First release the lock, then close the file
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                self._lock_handle.close()
                self._lock_handle = None
                logger.debug("Directory lock released")
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            # Try to clean up the handle even if lock release failed
            try:
                if hasattr(self, '_lock_handle') and self._lock_handle:
                    self._lock_handle.close()
                    self._lock_handle = None
            except:
                pass
    
    def _load_index(self):
        """Load index file (assumes lock is already held)"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index_data = json.load(f)
                    logger.debug(f"Index file loaded, containing {len(self.index_data)} records")
            except Exception as e:
                logger.error(f"Failed to load index file: {e}")
                self.index_data = {}
        else:
            logger.debug("Index file does not exist, creating new index")
            self.index_data = {}
    
    def _save_index(self):
        """Save index file (assumes lock is already held)"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index_data, f, ensure_ascii=False, indent=2)
                logger.debug("Index file saved successfully")
        except Exception as e:
            logger.error(f"Failed to save index file: {e}")
            raise
    
    def _cleanup_orphaned_files(self):
        """Internal method to cleanup orphaned files and index entries (assumes lock is already held)"""
        try:
            # Get all files in storage directory (excluding index.json and .lock)
            storage_files = set()
            for file_path in self.storage_dir.iterdir():
                if file_path.is_file() and file_path.name not in [self.index_file.name, self.lock_file.name]:
                    storage_files.add(file_path.name)
            
            # Get all files referenced in index
            indexed_files = set()
            orphaned_index_entries = []
            
            for uuid_val, file_info in list(self.index_data.items()):
                stored_name = file_info.get("stored_name")
                if stored_name:
                    indexed_files.add(stored_name)
                    # Check if physical file exists
                    file_path = Path(file_info.get("file_path", ""))
                    if not file_path.exists():
                        orphaned_index_entries.append(uuid_val)
                        logger.warning(f"Index entry has no physical file: {uuid_val}")
            
            # Find orphaned files (files without index entries)
            orphaned_files = storage_files - indexed_files
            
            # Remove orphaned files
            for orphaned_file in orphaned_files:
                file_path = self.storage_dir / orphaned_file
                try:
                    file_path.unlink()
                    logger.debug(f"Removed orphaned file: {orphaned_file}")
                except Exception as e:
                    logger.error(f"Failed to remove orphaned file {orphaned_file}: {e}")
            
            # Remove orphaned index entries
            for uuid_val in orphaned_index_entries:
                del self.index_data[uuid_val]
                logger.debug(f"Removed orphaned index entry: {uuid_val}")
            
            # Save index if any changes were made
            if orphaned_files or orphaned_index_entries:
                self._save_index()
                logger.info(f"Cleanup completed: removed {len(orphaned_files)} orphaned files and {len(orphaned_index_entries)} orphaned index entries")
            else:
                logger.debug("No cleanup needed - all files and index entries are consistent")
                
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned files: {e}")
    
    def store_file(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> str:
        """
        Store file and record metadata with directory lock
        
        Args:
            file_path: Path of file to store
            metadata: File metadata information
            
        Returns:
            str: UUID of stored file
        """
        try:
            self._acquire_lock()
            
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")
            
            # Check if there's an existing file with exactly matching metadata
            existing_files, _ = self._search_files_internal(metadata, exact_match=True)
            
            if existing_files:
                # Overwrite existing file
                existing_file_info = existing_files[0]  # Take the first match
                existing_uuid = None
                
                # Find the UUID for the existing file
                for uuid_val, info in self.index_data.items():
                    if info["file_path"] == existing_file_info["file_path"]:
                        existing_uuid = uuid_val
                        break
                
                if existing_uuid is None:
                    logger.warning("Found matching metadata but couldn't find UUID, creating new file")
                else:
                    # Overwrite the existing file
                    existing_path = Path(existing_file_info["file_path"])
                    
                    try:
                        # Copy new file to replace existing file
                        shutil.copy2(file_path, existing_path)
                        file_size = existing_path.stat().st_size
                        file_size_mb = round(file_size / (1024 * 1024), 2)
                        logger.info(f"File overwritten: {existing_file_info['stored_name']} ({file_size_mb} MB)")
                        
                        # Delete original file
                        file_path.unlink()
                        logger.debug(f"Original file deleted: {file_path}")
                        
                        # Update index with new file info
                        self.index_data[existing_uuid].update({
                            "original_name": file_path.name,
                            "file_size": existing_path.stat().st_size,
                            "created_time": existing_path.stat().st_ctime
                        })
                        
                        # Save index
                        self._save_index()
                        
                        return existing_uuid
                        
                    except Exception as e:
                        logger.error(f"Failed to overwrite file: {e}")
                        # Fall back to creating new file
            
            # No matching metadata found, create new file
            if self.fix_uuid is None:
                file_uuid = str(uuid.uuid4())
            else:
                file_uuid = self.fix_uuid
            file_extension = file_path.suffix
            stored_filename = f"{file_uuid}{file_extension}"
            stored_path = self.storage_dir / stored_filename
            
            try:
                # Copy file to storage directory with UUID naming
                shutil.copy2(file_path, stored_path)
                file_size = stored_path.stat().st_size
                file_size_mb = round(file_size / (1024 * 1024), 2)
                logger.info(f"File stored: {stored_filename} ({file_size_mb} MB)")
                
                # Delete original file
                file_path.unlink()
                logger.debug(f"Original file deleted: {file_path}")
                
                # Record to index
                self.index_data[file_uuid] = {
                    "original_name": file_path.name,
                    "stored_name": stored_filename,
                    "file_path": str(stored_path),
                    "metadata": metadata,
                    "file_size": stored_path.stat().st_size,
                    "created_time": stored_path.stat().st_ctime
                }
                
                # Save index
                self._save_index()
                
                return file_uuid
                
            except Exception as e:
                logger.error(f"Failed to store file: {e}")
                # If storage fails, delete the copied file
                if stored_path.exists():
                    stored_path.unlink()
                raise
        finally:
            self._release_lock()
    
    def _search_files_internal(self, metadata_query: Dict[str, Any], exact_match: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Internal search method that doesn't acquire locks (assumes lock is already held)"""
        if not metadata_query:
            return list(self.index_data.values()), list(self.index_data.keys())
        
        matched_files = []
        matched_uuids = []
        
        for file_uuid, file_info in self.index_data.items():
            file_metadata = file_info.get("metadata", {})
            
            if exact_match:
                # Exact match: all queried key-value pairs must be completely consistent
                if self._exact_metadata_match(file_metadata, metadata_query):
                    matched_files.append(file_info)
                    matched_uuids.append(file_uuid)
            else:
                # Partial match: queried key-value pairs exist and are consistent in file metadata
                if self._partial_metadata_match(file_metadata, metadata_query):
                    matched_files.append(file_info)
                    matched_uuids.append(file_uuid)
        
        return matched_files, matched_uuids

    def search_files(self, metadata_query: Dict[str, Any], exact_match: bool = False) -> List[Dict[str, Any]]:
        """
        Query files based on metadata with directory lock
        
        Args:
            metadata_query: Metadata query conditions
            exact_match: Whether to require exact match, False for partial match
            
        Returns:
            List[Dict]: List of matching file information
        """
        try:
            self._acquire_lock()
            
            matched_files, _ = self._search_files_internal(metadata_query, exact_match)
            logger.info(f"Found {len(matched_files)} matching files")
            return matched_files
        finally:
            self._release_lock()
    
    def search_uuids(self, metadata_query: Dict[str, Any], exact_match: bool = False) -> List[str]:
        """
        Query file UUIDs based on metadata with directory lock
        """
        try:
            self._acquire_lock()
            _, matched_uuids = self._search_files_internal(metadata_query, exact_match)
            return matched_uuids
        finally:
            self._release_lock()
    
    def _exact_metadata_match(self, file_metadata: Dict[str, Any], query_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches exactly"""
        return file_metadata == query_metadata
    
    def _partial_metadata_match(self, file_metadata: Dict[str, Any], query_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches partially"""
        for key, value in query_metadata.items():
            file_value = file_metadata.get(key, None)
            
            # Handle list matching: if query value is a list, check for intersection
            if value is None:
                if file_value is not None:
                    return False
                else:
                    continue
            elif file_value is None:
                return False
            elif isinstance(value, list) and isinstance(file_value, list):
                if file_value == value:
                    continue
                else:
                    return False
            # Handle list inclusion: if query value is a single value, check if it's in file value list
            elif isinstance(file_value, list) and value in file_value:
                continue
            # Regular value comparison
            elif file_value != value:
                return False
                
        return True
    
    def get_file_paths(self, metadata_query: Dict[str, Any], exact_match: bool = False) -> List[str]:
        """
        Query file path list based on metadata with directory lock
        
        Args:
            metadata_query: Metadata query conditions
            exact_match: Whether to require exact match
            
        Returns:
            List[str]: List of matching file paths
        """
        try:
            self._acquire_lock()
            
            matched_files, _ = self._search_files_internal(metadata_query, exact_match)
            return [file_info["file_path"] for file_info in matched_files]
        finally:
            self._release_lock()
    
    def get_file_info(self, file_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Get file information by UUID with directory lock
        
        Args:
            file_uuid: File UUID
            
        Returns:
            Dict: File information, returns None if not exists
        """
        try:
            self._acquire_lock()
            
            return self.index_data.get(file_uuid)
        finally:
            self._release_lock()
    
    def __del__(self):
        """Destructor to ensure lock is released"""
        try:
            if hasattr(self, '_lock_handle') and self._lock_handle:
                self._release_lock()
        except:
            pass  # Ignore errors during cleanup
    
    def delete_files_by_metadata(self, metadata_query: Dict[str, Any], exact_match: bool = False) -> int:
        """
        Delete files based on metadata query with directory lock
        
        Args:
            metadata_query: Metadata query conditions
            exact_match: Whether to require exact match, False for partial match
            
        Returns:
            int: Number of files deleted
        """
        try:
            self._acquire_lock()
            
            # Find matching files
            matching_files, _ = self._search_files_internal(metadata_query, exact_match)
            
            if not matching_files:
                logger.warning(f"No files found matching metadata: {metadata_query}")
                return 0
            
            deleted_count = 0
            
            for file_info in matching_files:
                file_path = Path(file_info["file_path"])
                file_uuid = None
                
                # Find the UUID for this file
                for uuid_val, info in self.index_data.items():
                    if info["file_path"] == file_info["file_path"]:
                        file_uuid = uuid_val
                        break
                
                if file_uuid is None:
                    logger.warning(f"UUID not found for file: {file_info['file_path']}")
                    continue
                
                try:
                    # Delete physical file
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"Physical file deleted: {file_path}")
                    
                    # Delete index record
                    del self.index_data[file_uuid]
                    deleted_count += 1
                    
                    logger.debug(f"File record deleted: {file_uuid}")
                    
                except Exception as e:
                    logger.error(f"Failed to delete file {file_uuid}: {e}")
            
            # Save index if any files were deleted
            if deleted_count > 0:
                self._save_index()
                logger.info(f"Deleted {deleted_count} files matching metadata: {metadata_query}")
            
            return deleted_count
        finally:
            self._release_lock()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics with directory lock
        
        Returns:
            Dict: Storage statistics information
        """
        try:
            self._acquire_lock()
            
            total_files = len(self.index_data)
            total_size = sum(file_info.get("file_size", 0) for file_info in self.index_data.values())
            
            # Count metadata key usage frequency
            metadata_keys = {}
            for file_info in self.index_data.values():
                for key in file_info.get("metadata", {}).keys():
                    metadata_keys[key] = metadata_keys.get(key, 0) + 1
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "metadata_keys": metadata_keys,
                "storage_directory": str(self.storage_dir)
            }
        finally:
            self._release_lock()

if __name__ == "__main__":
    storage = MetadataStorage("data")
    print(storage.get_storage_stats())
    storage = MetadataStorage("model")
    print(storage.get_storage_stats())