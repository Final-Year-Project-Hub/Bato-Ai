import requests
import json
import time

URL = "http://localhost:8000/api/v1/roadmap/generate"

def test_generate_roadmap():
    test_cases = [
        {
            "goal": "Build a personal portfolio website",
            "intent": "build",
            "proficiency": "beginner"
        },
        {
            "goal": "Learn about Next.js Routing and Data Fetching",
            "intent": "learn",
            "proficiency": "intermediate"
        }
    ]

    for payload in test_cases:
        print(f"\n🚀 Testing Intent: {payload['intent'].upper()}...")
        print(f"📝 Goal: {payload['goal']}")
        
        start_time = time.time()
        try:
            response = requests.post(URL, json=payload)
            response.raise_for_status()
            
            data = response.json()
            elapsed = time.time() - start_time
            
            print(f"✅ Success! (Took {elapsed:.2f}s)")
            # Print a summary of phases to keep output clean but verifiable
            for phase in data.get("phases", []):
                print(f"  - Phase: {phase['title']} ({phase['estimated_hours']}h)")
                for topic in phase.get("topics", [])[:2]: # Show first 2 topics per phase
                    print(f"    * {topic['title']}: {topic['description'][:60]}...")
            
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to server. Is uvicorn running?")
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            print(response.text)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_generate_roadmap()
