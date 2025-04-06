import os
import re
import logging
import json
from typing import Dict, Any, List, Optional

# FastAPI and supporting libraries
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# PDF processing
import PyPDF2
import io

# AI Libraries
import google.generativeai as genai
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResumeParser:
    """Advanced resume parsing with flexible extraction methods."""
    
    def __init__(self):
        # Comprehensive section extraction patterns
        self.section_patterns = {
            'education': [
                r'Education\s*(.*?)(?=\n\w+:|$)',
                r'EDUCATION\s*(.*?)(?=\n\w+:|$)'
            ],
            'experience': [
                r'Experience\s*(.*?)(?=\n\w+:|$)',
                r'PROFESSIONAL EXPERIENCE\s*(.*?)(?=\n\w+:|$)'
            ],
            'skills': [
                r'(?:Technical\s*)?Skills\s*(.*?)(?=\n\w+:|$)',
                r'SKILLS\s*(.*?)(?=\n\w+:|$)'
            ],
            'projects': [
                r'Projects\s*(.*?)(?=\n\w+:|$)',
                r'PROJECTS\s*(.*?)(?=\n\w+:|$)'
            ]
        }
    
    def extract_section(self, text: str, patterns: List[str]) -> str:
        """
        Tries multiple regex patterns to extract a section from resume text.
        
        Args:
            text (str): Full resume text
            patterns (List[str]): List of regex patterns to try
        
        Returns:
            str: Extracted section text, or empty string if no match
        """
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Parses resume text into structured sections with fallback mechanisms.
        
        Args:
            resume_text (str): Full resume text
        
        Returns:
            Dict[str, Any]: Parsed resume sections
        """
        parsed_data = {}
        
        # Extract sections using multiple patterns
        for section, patterns in self.section_patterns.items():
            parsed_data[section] = self.extract_section(resume_text, patterns)
        
        # Additional processing
        parsed_data['skills_list'] = [
            skill.strip() for skill in 
            re.findall(r'\b[A-Za-z0-9#+\-/]+\b', parsed_data.get('skills', ''))
        ]
        
        return parsed_data

class JobDescriptionParser:
    """Enhanced job description parsing with advanced extraction."""
    
    def parse_job_description(self, job_text: str) -> Dict[str, Any]:
        """
        Extracts structured information from job description.
        
        Args:
            job_text (str): Full job description text
        
        Returns:
            Dict[str, Any]: Parsed job description details
        """
        # Advanced skill and requirement extraction
        skills_patterns = [
            r'\b(?:Python|Java|SQL|AWS|GCP|ML|AI|NLP|PyTorch|TensorFlow|Kubernetes|Docker|React|Node\.js)\b',
            r'\b(?:Machine Learning|Data Science|Cloud Computing|Backend|Frontend|Full Stack)\b'
        ]
        
        responsibilities_patterns = [
            r'\b(?:Develop|Design|Implement|Maintain|Test|Create|Build|Manage|Optimize)\b',
            r'Responsible for\s+([^.]+)'
        ]
        
        # Combine and deduplicate results
        necessary_qualifications = list(set(
            match.group(0) if match.group(0) else match.group(1)
            for pattern in skills_patterns 
            for match in re.finditer(pattern, job_text, re.IGNORECASE)
        ))
        
        responsibilities = list(set(
            match.group(0) if match.group(0) else match.group(1)
            for pattern in responsibilities_patterns 
            for match in re.finditer(pattern, job_text, re.IGNORECASE)
        ))
        
        return {
            'necessary_qualifications': necessary_qualifications,
            'responsibilities': responsibilities
        }

class SkillsAssessmentGenerator:
    """
    Generates dynamic skills assessment using AI.
    Supports both OpenAI and Google Gemini for flexibility.
    """
    
    def __init__(self, ai_model=None):
        """
        Initialize with optional AI model.
        
        Args:
            ai_model: AI model (OpenAI or Gemini) for question generation
        """
        self.ai_model = ai_model or genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_assessment_questions(
        self, 
        skills: List[str], 
        difficulty: str = 'intermediate'
    ) -> List[Dict[str, Any]]:
        """
        Generate assessment questions for given skills.
        
        Args:
            skills (List[str]): Skills to assess
            difficulty (str, optional): Question difficulty. Defaults to 'intermediate'
        
        Returns:
            List[Dict[str, Any]]: Generated assessment questions
        """
        prompt = f"""
        Generate multiple-choice assessment questions for the following skills: {', '.join(skills)}
        
        Requirements:
        - {len(skills)} questions total
        - Difficulty level: {difficulty}
        - Each question should have 4 options
        - Include a detailed explanation for the correct answer
        - Focus on practical application and problem-solving
        
        Format as JSON with this structure:
        [
            {{
                "skill": "skill name",
                "question": "question text",
                "options": ["option1", "option2", "option3", "option4"],
                "correct_index": 0,
                "explanation": "detailed explanation"
            }}
        ]
        """
        
        try:
            response = self.ai_model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error generating assessment questions: {e}")
            return []

class InterviewPreparationPlanGenerator:
    """
    Generates personalized interview preparation plans
    using AI-driven analysis.
    """
    
    def __init__(self, ai_model=None):
        """
        Initialize with optional AI model.
        
        Args:
            ai_model: AI model for plan generation
        """
        self.ai_model = ai_model or genai.GenerativeModel('gemini-1.5-flash')
    
    def create_preparation_plan(
        self, 
        resume_data: Dict[str, Any], 
        job_description: Dict[str, Any],
        assessment_results: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive interview preparation plan.
        
        Args:
            resume_data (Dict[str, Any]): Parsed resume details
            job_description (Dict[str, Any]): Parsed job description
            assessment_results (Dict[str, Any], optional): Previous assessment results
        
        Returns:
            Dict[str, Any]: Detailed preparation plan
        """
        prompt = f"""
        Create a comprehensive 7-day interview preparation plan based on:
        
        Resume Skills: {resume_data.get('skills', [])}
        Job Requirements: {job_description.get('necessary_qualifications', [])}
        Assessment Results: {assessment_results or 'Not available'}
        
        Plan should include:
        1. Daily learning objectives
        2. Skill gap identification
        3. Specific study resources
        4. Practice exercises
        5. Interview preparation techniques
        6. Time allocation for each activity
        """
        
        try:
            response = self.ai_model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error generating preparation plan: {e}")
            return {}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_bytes (bytes): PDF file content
    
    Returns:
        str: Extracted text from PDF
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

# FastAPI Application Setup
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize AI models
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
openai_model = OpenAI(temperature=0.7, max_tokens=2000)

# Instantiate key components
resume_parser = ResumeParser()
job_parser = JobDescriptionParser()
skills_assessor = SkillsAssessmentGenerator()
plan_generator = InterviewPreparationPlanGenerator()

@app.post("/api/process-resume")
async def process_resume(
    resume: UploadFile = File(...), 
    job_description: str = Form(...)
):
    """
    Process resume and job description to generate initial assessment.
    
    Args:
        resume (UploadFile): Uploaded resume PDF
        job_description (str): Job description text
    
    Returns:
        JSONResponse with skills assessment
    """
    try:
        # Extract resume text
        resume_content = await resume.read()
        resume_text = extract_text_from_pdf(resume_content)
        
        # Parse resume and job description
        parsed_resume = resume_parser.parse_resume(resume_text)
        parsed_job = job_parser.parse_job_description(job_description)
        
        # Generate skills assessment
        skills_to_assess = list(set(parsed_job.get('necessary_qualifications', [])))
        assessment_questions = skills_assessor.generate_assessment_questions(skills_to_assess)
        
        return JSONResponse(content={
            "resume_skills": parsed_resume.get('skills_list', []),
            "job_requirements": parsed_job,
            "assessment_questions": assessment_questions
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-preparation-plan")
async def generate_preparation_plan(request_data: Dict):
    """
    Generate personalized interview preparation plan.
    
    Args:
        request_data (Dict): Request containing assessment data
    
    Returns:
        JSONResponse with preparation plan
    """
    try:
        resume_data = request_data.get("resume_data", {})
        job_description = request_data.get("job_description", {})
        assessment_results = request_data.get("assessment_results")
        
        preparation_plan = plan_generator.create_preparation_plan(
            resume_data, 
            job_description, 
            assessment_results
        )
        
        return JSONResponse(content=preparation_plan)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)