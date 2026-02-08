from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import httpx
import zipfile
import shutil
import yaml
import logging
import subprocess
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

def git_commit_and_push(framework_key: str, framework_name: str):
    """Commit and push extracted documentation to GitHub."""
    try:
        # Check if git is available
        if not os.path.exists(".git"):
            logger.warning("Not a git repository, skipping auto-commit")
            return False
        
        # Configure git if needed (for CI/CD environments)
        subprocess.run(
            ["git", "config", "user.name", "Bato-AI Bot"],
            check=False,
            capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.email", "bot@bato-ai.com"],
            check=False,
            capture_output=True
        )
        
        # Add the docs folder
        result = subprocess.run(
            ["git", "add", f"docs/{framework_key}"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Git add failed: {result.stderr}")
            return False
        
        # Add frameworks.yaml
        subprocess.run(
            ["git", "add", "frameworks.yaml"],
            capture_output=True,
            check=False
        )
        
        # Commit
        commit_message = f"Add {framework_name} documentation via admin dashboard"
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            # Check if it's because there's nothing to commit
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                logger.info("No changes to commit")
                return True
            logger.error(f"Git commit failed: {result.stderr}")
            return False
        
        # Push to remote
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            # Try 'master' branch if 'main' fails
            result = subprocess.run(
                ["git", "push", "origin", "master"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Git push failed: {result.stderr}")
                return False
        
        logger.info(f"Successfully committed and pushed {framework_key} documentation")
        return True
        
    except Exception as e:
        logger.error(f"Git auto-commit error: {e}")
        return False

class DownloadRequest(BaseModel):
    documentId: str
    frameworkKey: str
    downloadUrl: str
    subdirectory: str | None = None

class ExtractLocalRequest(BaseModel):
    documentId: str
    frameworkKey: str
    localPath: str
    subdirectory: str | None = None

class RegisterRequest(BaseModel):
    frameworkKey: str
    frameworkName: str
    baseUrl: str
    version: str
    docsPath: str
    collectionName: str
    extensions: list[str]
    subdirectory: str | None
    preprocessing: dict
    url_config: dict

@router.post("/extract-local")
async def extract_local_file(request: ExtractLocalRequest):
    """Extract ZIP from local file path (no cloud storage needed)."""
    
    docs_path = Path("docs") / request.frameworkKey
    
    try:
        logger.info(f"Extracting local file for {request.frameworkKey}")
        
        # Check if local file exists
        local_zip = Path(request.localPath)
        if not local_zip.exists():
            raise HTTPException(status_code=404, detail=f"Local file not found: {request.localPath}")
        
        logger.info(f"Extracting to {docs_path}")
        
        # Extract to docs folder
        docs_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(docs_path)
        
        # Handle subdirectory if specified
        if request.subdirectory:
            subdir_path = docs_path / request.subdirectory
            if subdir_path.exists() and subdir_path.is_dir():
                logger.info(f"Moving contents from subdirectory: {request.subdirectory}")
                # Move contents up one level
                for item in subdir_path.iterdir():
                    dest = docs_path / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                # Remove empty subdirectory
                subdir_path.rmdir()
        
        # Count files and calculate size
        file_count = sum(1 for _ in docs_path.rglob('*') if _.is_file())
        total_size = sum(f.stat().st_size for f in docs_path.rglob('*') if f.is_file())
        
        logger.info(f"Extraction complete: {file_count} files, {total_size} bytes")
        
        # Auto-commit to GitHub (Note: frameworks.yaml will be committed later by register endpoint)
        git_committed = git_commit_and_push(request.frameworkKey, request.frameworkKey)
        
        return {
            "success": True,
            "frameworkKey": request.frameworkKey,
            "docsPath": str(docs_path),
            "fileCount": file_count,
            "totalSize": total_size,
            "gitCommitted": git_committed
        }
        
    except Exception as e:
        logger.error(f"Extract error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download")
async def download_and_extract(request: DownloadRequest):
    """Download ZIP from cloud storage and extract to docs folder."""
    
    docs_path = Path("docs") / request.frameworkKey
    zip_path = Path("temp") / f"{request.frameworkKey}.zip"
    
    try:
        logger.info(f"Downloading documentation for {request.frameworkKey}")
        
        # Download ZIP
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(request.downloadUrl)
            response.raise_for_status()
            
            # Save temporarily
            zip_path.parent.mkdir(exist_ok=True)
            zip_path.write_bytes(response.content)
        
        logger.info(f"Extracting to {docs_path}")
        
        # Extract to docs folder
        docs_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(docs_path)
        
        # Handle subdirectory if specified
        if request.subdirectory:
            subdir_path = docs_path / request.subdirectory
            if subdir_path.exists() and subdir_path.is_dir():
                logger.info(f"Moving contents from subdirectory: {request.subdirectory}")
                # Move contents up one level
                for item in subdir_path.iterdir():
                    dest = docs_path / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                # Remove empty subdirectory
                subdir_path.rmdir()
        
        # Count files and calculate size
        file_count = sum(1 for _ in docs_path.rglob('*') if _.is_file())
        total_size = sum(f.stat().st_size for f in docs_path.rglob('*') if f.is_file())
        
        # Cleanup temp ZIP
        zip_path.unlink()
        
        logger.info(f"Extraction complete: {file_count} files, {total_size} bytes")
        
        return {
            "success": True,
            "frameworkKey": request.frameworkKey,
            "docsPath": str(docs_path),
            "fileCount": file_count,
            "totalSize": total_size
        }
        
    except Exception as e:
        logger.error(f"Download/extract error: {e}")
        # Cleanup on error
        if zip_path.exists():
            zip_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register")
async def register_framework(request: RegisterRequest):
    """Register new framework in frameworks.yaml."""
    
    config_file = Path("frameworks.yaml")
    
    try:
        logger.info(f"Registering framework: {request.frameworkKey}")
        
        # Read existing config
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {"frameworks": {}}
        else:
            config = {"frameworks": {}}
        
        # Check if framework already exists
        if request.frameworkKey in config.get("frameworks", {}):
            logger.warning(f"Framework {request.frameworkKey} already exists, updating...")
        
        # Add/update framework configuration
        config["frameworks"][request.frameworkKey] = {
            "name": request.frameworkName,
            "key": request.frameworkKey,
            "base_url": request.baseUrl,
            "docs_path": request.docsPath,
            "collection": request.collectionName,
            "version": request.version,
            "extensions": request.extensions,
            "subdirectory": request.subdirectory,
            "preprocessing": request.preprocessing,
            "url_config": request.url_config,
        }
        
        # Write back to file
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Framework {request.frameworkKey} registered successfully")
        
        # Auto-commit frameworks.yaml to GitHub
        git_committed = git_commit_and_push(request.frameworkKey, request.frameworkName)
        
        return {
            "success": True,
            "message": f"Framework '{request.frameworkKey}' registered",
            "gitCommitted": git_committed
        }
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{framework_key}")
async def delete_framework(framework_key: str):
    """Delete framework files and remove from config."""
    
    try:
        logger.info(f"Deleting framework: {framework_key}")
        
        # Delete docs folder
        docs_path = Path("docs") / framework_key
        if docs_path.exists():
            shutil.rmtree(docs_path)
            logger.info(f"Deleted docs folder: {docs_path}")
        
        # Remove from frameworks.yaml
        config_file = Path("frameworks.yaml")
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if framework_key in config.get("frameworks", {}):
                del config["frameworks"][framework_key]
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                logger.info(f"Removed {framework_key} from frameworks.yaml")
        
        return {"success": True, "message": f"Framework '{framework_key}' deleted"}
        
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
