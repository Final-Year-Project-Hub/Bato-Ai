import requests
import json

URL = "http://localhost:8000/api/v1/chat"

def test_chat():
    # Scenario 1: Partial Query
    print("\n🔹 Testing Partial Query...")
    payload1 = {"message": "I want to learn Next.js"}
    print(f"User: {payload1['message']}")
    
    try:
        response = requests.post(URL, json=payload1)
        response.raise_for_status()
        data = response.json()
        
        if "missing_fields" in data:
            print(f"🤖 Bot: {data['message']}")
            print(f"   (Missing: {data['missing_fields']})")
        else:
            print("❌ Unexpected: Bot generated roadmap without proficiency!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

    # Scenario 2: Complete Query
    print("\n🔹 Testing Complete Query...")
    payload2 = {"message": "I want to build a portfolio website using Next.js. I am a beginner."}
    print(f"User: {payload2['message']}")
    
    try:
        response = requests.post(URL, json=payload2)
        response.raise_for_status()
        data = response.json()
        
        if "phases" in data:
            print(f"✅ Bot: Generated Roadmap with {len(data['phases'])} phases.")
             # Print a summary of phases to keep output clean but verifiable
            for phase in data.get("phases", []):
                print(f"\n  [Phase]: {phase['title']} ({phase['estimated_hours']}h)")
                for topic in phase.get("topics", []):
                    print(f"    * {topic['title']}")
                    print(f"      - Why: {topic.get('why_it_matters', 'N/A')[:60]}...")
                    print(f"      - Concepts: {topic.get('key_concepts', [])}")
        else:
             print(f"🤖 Bot (Unexpected): {data.get('message')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_chat()
