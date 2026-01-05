import os
import subprocess
import argparse
import sys
from pathlib import Path
import yaml

def load_framework_configs():
    """
    Load framework configurations from frameworks.yaml.
    
    Returns dict with framework download configurations.
    """
    config_file = Path(__file__).parent.parent / "frameworks.yaml"
    
    if not config_file.exists():
        print(f"❌ frameworks.yaml not found at {config_file}")
        print("Please create frameworks.yaml in the project root.")
        sys.exit(1)
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        frameworks = config.get("frameworks", {})
        
        # Convert to download format
        doc_sources = {}
        for key, fw_config in frameworks.items():
            # Only include frameworks that have git repo info
            if "git_repo" in fw_config:
                doc_sources[key] = {
                    "repo": fw_config["git_repo"],
                    "branch": fw_config.get("git_branch", "main"),
                    "sparse_paths": fw_config.get("git_sparse_paths", []),
                    "target_dir": fw_config.get("docs_path", f"docs/{key}")
                }
        
        return doc_sources
    
    except Exception as e:
        print(f"❌ Error loading frameworks.yaml: {e}")
        sys.exit(1)

def run_command(cmd, cwd=None):
    """Run a shell command and print output."""
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        return False
    return True

def download_source(key, source, force=False):
    """Download documentation for a framework."""
    target_dir = Path(source["target_dir"])
    
    if target_dir.exists():
        if not force:
            print(f"Directory {target_dir} already exists. Use --force to overwrite or update.")
            return
        print(f"Updating {key} docs...")
    else:
        print(f"Downloading {key} docs...")
        target_dir.parent.mkdir(parents=True, exist_ok=True)

    repo_url = source["repo"]
    branch = source["branch"]
    sparse_paths = source.get("sparse_paths", [])
    
    # 1. Clone with no checkout (lightweight)
    if not target_dir.exists():
        cmd = f"git clone --filter=blob:none --no-checkout {repo_url} {target_dir}"
        if not run_command(cmd):
            return
    
    # 2. Configure sparse checkout (if paths specified)
    cwd = str(target_dir)
    
    if sparse_paths:
        sparse_paths_str = " ".join(sparse_paths)
        run_command("git sparse-checkout init --cone", cwd=cwd)
        run_command(f"git sparse-checkout set {sparse_paths_str}", cwd=cwd)
    
    # 3. Checkout the branch
    run_command(f"git checkout {branch}", cwd=cwd)
    
    print(f"✅ Successfully downloaded {key} documentation to {target_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Download framework documentation from frameworks.yaml"
    )
    parser.add_argument(
        "frameworks", 
        nargs="*", 
        help="Frameworks to download. Leave empty for all."
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force update/overwrite existing directories"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available frameworks"
    )
    
    args = parser.parse_args()
    
    # Load configurations from YAML
    doc_sources = load_framework_configs()
    
    if args.list:
        print("Available frameworks (with git repo configured):")
        for key in doc_sources:
            print(f" - {key}")
        
        print("\nTo add more frameworks:")
        print("1. Edit frameworks.yaml")
        print("2. Add git_repo, git_branch, and git_sparse_paths fields")
        print("3. Run this script again")
        return

    targets = args.frameworks if args.frameworks else doc_sources.keys()
    
    for target in targets:
        if target not in doc_sources:
            print(f"❌ Framework '{target}' not found or missing git configuration")
            print(f"Available: {', '.join(doc_sources.keys())}")
            continue
        
        download_source(target, doc_sources[target], force=args.force)

if __name__ == "__main__":
    main()

