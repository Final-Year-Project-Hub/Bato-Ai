import sys
import os
# Add project root to path
sys.path.append(os.getcwd())

from app.schemas.document import ChatRequest
from pydantic import ValidationError

payload = {
  "message": "learn nextjs",
  "conversation_history": [
    {
      "role": "user",
      "content": "learn nextjs"
    }
  ],
  "user_context": {
    "user_id": "5d21712a-695d-4bb0-98da-eb9d3c717829",
    "user_name": "Kuber Pathak",
    "known_technologies": []
  }
}

try:
    print("Validating payload...")
    validated = ChatRequest(**payload)
    print("✅ Validation Successful!")
    print(validated.json(indent=2))
except ValidationError as e:
    print("❌ Validation Failed!")
    print(e)
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
