import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from app.ingestion.ingest_qdrant import main

if __name__ == "__main__":
    print("🧪 Running Ingestion Test...")
    try:
        main()
        print("✅ Ingestion Test Passed")
    except Exception as e:
        print(f"❌ Ingestion Test Failed: {e}")
        import traceback
        traceback.print_exc()
