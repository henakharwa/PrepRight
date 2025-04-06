# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict
import google.generativeai as genai
import PyPDF2
import io
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
genai.configure(api_key=os.getenv("API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

@app.post("/api/process-initial")
async def process_initial(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        # Read and process the PDF
        resume_content = await resume.read()
        resume_text = extract_text_from_pdf(resume_content)

        # Create prompt for Gemini
        prompt = f"""
        As an AI interviewer, analyze the following resume and job description:

        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Based on the job requirements and candidate's experience:
        1. Identify the key skills required for the position
        2. Assess the candidate's skill level in each area
        3. Generate 5 technical assessment questions that match the required skill level
        4. For each question, specify the expected skill level (beginner/intermediate/advanced)

        Format the response as a JSON object with the following structure:
        {{
            "skills": [
                {{
                    "name": "skill name",
                    "required_level": "level",
                    "candidate_level": "level"
                }}
            ],
            "questions": [
                {{
                    "id": "number",
                    "question": "question text",
                    "skill_level": "level",
                    "related_skill": "skill name"
                }}
            ]
        }}
        """

        # Generate response using Gemini
        response = model.generate_content(prompt)
        processed_response = json.loads(response.text)

        return JSONResponse(content=processed_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-plan")
async def generate_plan(request_data: Dict):
    try:
        answers = request_data.get("answers")
        skills = request_data.get("skills")
        job_description = request_data.get("jobDescription")

        prompt = f"""
        As an AI interview coach, analyze the following assessment responses and create a personalized interview preparation plan:

        Job Description:
        {job_description}

        Required Skills:
        {json.dumps(skills)}

        Assessment Answers:
        {json.dumps(answers)}

        Create a 7-day interview preparation plan that:
        1. Identifies weak areas based on the assessment responses
        2. Provides daily focus areas and specific tasks
        3. Includes study resources and practice exercises
        4. Prioritizes areas that need the most improvement

        Format the response as a JSON object with the following structure:
        {{
            "weak_areas": [
                {{
                    "skill": "skill name",
                    "current_level": "level",
                    "target_level": "level"
                }}
            ],
            "daily_plan": [
                {{
                    "day": "number",
                    "focus_area": "area",
                    "tasks": ["task1", "task2"],
                    "resources": ["resource1", "resource2"]
                }}
            ]
        }}
        """

        response = model.generate_content(prompt)
        preparation_plan = json.loads(response.text)

        return JSONResponse(content=preparation_plan)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
