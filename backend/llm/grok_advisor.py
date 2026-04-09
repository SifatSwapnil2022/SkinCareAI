from groq import Groq
from dotenv import load_dotenv
import os
import json
import re
import time

load_dotenv()
os.environ.pop("GROQ_BASE_URL", None)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"


def build_prompt(disease: str, confidence: float) -> str:
    confidence_pct = round(confidence * 100, 1)
    return f"""You are a professional medical AI assistant specializing in dermatology.

A patient uploaded a skin image. The AI model detected:
- Disease: {disease}
- Confidence: {confidence_pct}%

Search your medical knowledge for detailed information about "{disease}" skin condition.
Respond ONLY with a valid JSON object. No extra text, no markdown, no code fences.

{{
    "recommendations": "2-3 sentences about what {disease} is and general care advice",
    "next_steps": "2-3 clear action steps the patient should take for {disease}",
    "tips": "2-3 practical daily tips specific to managing {disease}",
    "severity": "Low or Medium or High",
    "see_doctor_urgently": true or false
}}

Rules:
- Be empathetic and specific to {disease}
- Do NOT definitively diagnose — remind user this is AI-based
- Melanoma and BCC must have severity=High and see_doctor_urgently=true
- Return ONLY the JSON, nothing else
"""


def get_recommendations(disease: str, confidence: float) -> dict:
    prompt = build_prompt(disease, confidence)

    for attempt in range(3):
        try:
            print(f"[LLM] Calling Groq for: {disease} ...")

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a dermatology medical AI. Always respond with valid JSON only."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.4,
                max_tokens=500,
            )

            text = response.choices[0].message.content.strip()
            print(f"[LLM RAW] {text[:300]}")

            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$',     '', text)

            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"[LLM] ✅ Success for {disease}")
                return result

            raise ValueError(f"No JSON found: {text[:100]}")

        except Exception as e:
            print(f"[LLM ERROR] Attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(3)

    # All 3 attempts failed — return empty strings, no hardcoded text
    print(f"[LLM] ❌ All attempts failed for {disease}")
    return {
        "recommendations":     "",
        "next_steps":          "",
        "tips":                "",
        "severity":            "",
        "see_doctor_urgently": False
    }