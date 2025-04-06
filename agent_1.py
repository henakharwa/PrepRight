import os
import re
import json
import logging
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
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create AI model
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

class SkillsAssessmentGenerator:
    """
    Generates dynamic skills assessment using AI.
    Supports flexible AI model initialization.
    """
    
    def __init__(self, ai_model=None):
        """
        Initialize the skills assessment generator.
        
        Args:
            ai_model: AI model for question generation (defaults to Gemini)
        """
        # Use Gemini as default if no model provided
        if ai_model is None:
            try:
                import google.generativeai as genai
                # Ensure API key is configured
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.ai_model = genai.GenerativeModel('gemini-1.5-flash')
            except ImportError:
                # Fallback error handling
                logger.error("Gemini AI not available. Unable to generate questions.")
                self.ai_model = None
        else:
            # Use provided model
            self.ai_model = ai_model
  
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
        if self.ai_model is None:
            logger.error("No AI model available for generating questions")
            return self.generate_fallback_questions(skills)
        # More explicit and constrained prompt
        prompt = f"""Please generate a JSON-formatted list of assessment questions with this exact structure:
{{
    "questions": [
        {{
            "id": integer_id,
            "question": "Specific question text addressing {difficulty} {skills[0]} skills",
            "skill": "{skills[0]}"
        }}
    ]
}}

Guidelines:
- Ensure the response is valid JSON
- Create {len(skills)} questions total
- Focus on practical, problem-solving scenarios
- Use clear, concise language
- Difficulty level: {difficulty}"""
        
        try:
            # Attempt to generate response
            response = self.ai_model.generate_content(prompt)
            
            # Additional parsing safeguards
            response_text = response.text.strip()
            
            # Remove any markdown code block formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            # Validate JSON parsing
            try:
                parsed_response = json.loads(response_text)
                
                # Validate structure
                if not isinstance(parsed_response, dict) or 'questions' not in parsed_response:
                    raise ValueError("Invalid response structure")
                
                # Ensure each question has required fields
                for i, q in enumerate(parsed_response['questions'], 1):
                    q['id'] = i  # Ensure unique IDs
                    if not all(k in q for k in ['id', 'question', 'skill']):
                        raise ValueError(f"Question {i} missing required fields")
                
                return parsed_response['questions']
            
            except (json.JSONDecodeError, ValueError) as parse_error:
                logger.error(f"JSON Parsing Error: {parse_error}")
                logger.error(f"Problematic Response: {response_text}")
                
                # Fallback: Generate default questions
                return [
                    {
                        "id": i+1, 
                        "question": f"Tell me about your experience with {skill}",
                        "skill": skill
                    } for i, skill in enumerate(skills[:3])
                ]
        
        except Exception as e:
            logger.error(f"Error generating assessment questions: {e}")
            
            # Comprehensive fallback
            return [
                {
                    "id": i+1, 
                    "question": f"Describe your proficiency and experience with {skill}",
                    "skill": skill
                } for i, skill in enumerate(skills[:3])
            ]
    def generate_fallback_questions(self, skills: List[str]) -> List[Dict[str, Any]]:
        """
        Generate default questions when AI generation fails.
        
        Args:
            skills (List[str]): Skills to create fallback questions for
        
        Returns:
            List[Dict[str, Any]]: Manually created assessment questions
        """
        return [
            {
                "id": i + 1,
                "question": f"Describe your experience and proficiency with {skill}",
                "skill": skill
            } for i, skill in enumerate(skills[:3])
        ]    

# Instantiate components with explicit model passing
skills_assessor = SkillsAssessmentGenerator(ai_model=gemini_model)
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
    Supports flexible AI model initialization.
    """
    
    def __init__(self, ai_model=None):
        """
        Initialize the skills assessment generator.
        
        Args:
            ai_model: AI model for question generation (defaults to Gemini)
        """
        # Use Gemini as default if no model provided
        if ai_model is None:
            try:
                import google.generativeai as genai
                # Ensure API key is configured
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.ai_model = genai.GenerativeModel('gemini-1.5-flash')
            except ImportError:
                # Fallback error handling
                logger.error("Gemini AI not available. Unable to generate questions.")
                self.ai_model = None
        else:
            # Use provided model
            self.ai_model = ai_model
  
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
        if self.ai_model is None:
            logger.error("No AI model available for generating questions")
            return self.generate_fallback_questions(skills)
        # More explicit and constrained prompt
        prompt = f"""Please generate a JSON-formatted list of assessment questions with this exact structure:
{{
    "questions": [
        {{
            "id": integer_id,
            "question": "Specific question text addressing {difficulty} {skills[0]} skills",
            "skill": "{skills[0]}"
        }}
    ]
}}

Guidelines:
- Ensure the response is valid JSON
- Create {len(skills)} questions total
- Focus on practical, problem-solving scenarios
- Use clear, concise language
- Difficulty level: {difficulty}"""
        
        try:
            # Attempt to generate response
            response = self.ai_model.generate_content(prompt)
            
            # Additional parsing safeguards
            response_text = response.text.strip()
            
            # Remove any markdown code block formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            # Validate JSON parsing
            try:
                parsed_response = json.loads(response_text)
                
                # Validate structure
                if not isinstance(parsed_response, dict) or 'questions' not in parsed_response:
                    raise ValueError("Invalid response structure")
                
                # Ensure each question has required fields
                for i, q in enumerate(parsed_response['questions'], 1):
                    q['id'] = i  # Ensure unique IDs
                    if not all(k in q for k in ['id', 'question', 'skill']):
                        raise ValueError(f"Question {i} missing required fields")
                
                return parsed_response['questions']
            
            except (json.JSONDecodeError, ValueError) as parse_error:
                logger.error(f"JSON Parsing Error: {parse_error}")
                logger.error(f"Problematic Response: {response_text}")
                
                # Fallback: Generate default questions
                return [
                    {
                        "id": i+1, 
                        "question": f"Tell me about your experience with {skill}",
                        "skill": skill
                    } for i, skill in enumerate(skills[:3])
                ]
        
        except Exception as e:
            logger.error(f"Error generating assessment questions: {e}")
            
            # Comprehensive fallback
            return [
                {
                    "id": i+1, 
                    "question": f"Describe your proficiency and experience with {skill}",
                    "skill": skill
                } for i, skill in enumerate(skills[:3])
            ]
        
    def generate_fallback_questions(self, skills: List[str]) -> List[Dict[str, Any]]:
        """
        Generate default questions when AI generation fails.
        
        Args:
            skills (List[str]): Skills to create fallback questions for
        
        Returns:
            List[Dict[str, Any]]: Manually created assessment questions
        """
        return [
            {
                "id": i + 1,
                "question": f"Describe your experience and proficiency with {skill}",
                "skill": skill
            } for i, skill in enumerate(skills[:3])
        ]    

# Similar modifications for PreparationPlanGenerator
class PreparationPlanGenerator:
    def create_preparation_plan(
        self, 
        answers: Dict[str, str],
        job_description: str
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive interview preparation plan.
        
        Args:
            answers (Dict[str, str]): User's assessment answers
            job_description (str): Job description text
        
        Returns:
            Dict[str, Any]: Detailed preparation plan
        """
        prompt = f"""Generate a JSON-formatted preparation plan with this structure:
{{
    "weak_areas": [
        {{
            "skill": "skill_name",
            "current_level": "beginner/intermediate/advanced",
            "target_level": "beginner/intermediate/advanced"
        }}
    ],
    "daily_plan": [
        {{
            "day": 1,
            "focus_area": "specific_skill_area",
            "tasks": ["task1", "task2"],
            "resources": ["resource1", "resource2"]
        }}
    ]
}}

Context:
- Job Description: {job_description}
- User Answers: {json.dumps(answers)}

Requirements:
- Create a 7-day plan
- Identify skill gaps
- Provide actionable tasks
- Include learning resources"""
        
        try:
            response = self.ai_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            try:
                parsed_response = json.loads(response_text)
                
                # Validate structure
                if not isinstance(parsed_response, dict):
                    raise ValueError("Invalid response structure")
                
                # Ensure required keys exist with default values
                parsed_response.setdefault('weak_areas', [])
                parsed_response.setdefault('daily_plan', [])
                
                return parsed_response
            
            except (json.JSONDecodeError, ValueError) as parse_error:
                logger.error(f"JSON Parsing Error: {parse_error}")
                logger.error(f"Problematic Response: {response_text}")
                
                # Comprehensive fallback
                return {
                    "weak_areas": [
                        {
                            "skill": "General Technical Skills",
                            "current_level": "intermediate",
                            "target_level": "advanced"
                        }
                    ],
                    "daily_plan": [
                        {
                            "day": 1,
                            "focus_area": "Technical Skills Review",
                            "tasks": [
                                "Review core technical concepts",
                                "Practice coding problem solving"
                            ],
                            "resources": [
                                "https://leetcode.com",
                                "https://github.com/trending"
                            ]
                        }
                    ]
                }
        
        except Exception as e:
            logger.error(f"Error generating preparation plan: {e}")
            
            # Safest possible fallback
            return {
                "weak_areas": [],
                "daily_plan": []
            }
        
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
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Instantiate key components
resume_parser = ResumeParser()
job_parser = JobDescriptionParser()
skills_assessor = SkillsAssessmentGenerator()
plan_generator = PreparationPlanGenerator()

@app.post("/api/process-initial")
async def process_initial(
    resume: UploadFile = File(...), 
    jobDescription: str = Form(...)
):
    """
    Process resume and job description to generate initial assessment.
    
    Args:
        resume (UploadFile): Uploaded resume PDF
        jobDescription (str): Job description text
    
    Returns:
        JSONResponse with skills assessment
    """
    try:
        # Extract resume text
        resume_content = await resume.read()
        resume_text = extract_text_from_pdf(resume_content)
        
        # Parse resume and job description
        parsed_resume = resume_parser.parse_resume(resume_text)
        parsed_job = job_parser.parse_job_description(jobDescription)
        
        # Generate skills assessment
        skills_to_assess = list(set(parsed_job.get('necessary_qualifications', [])))
        assessment_response = skills_assessor.generate_assessment_questions(skills_to_assess)
        
        return JSONResponse(content={
            "questions": assessment_response.get("questions", []),
            "resume_skills": parsed_resume.get('skills_list', []),
            "job_requirements": parsed_job
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-plan")
async def generate_plan(request_data: Dict):
    """
    Generate personalized interview preparation plan.
    
    Args:
        request_data (Dict): Request containing assessment data
    
    Returns:
        JSONResponse with preparation plan
    """
    try:
        answers = request_data.get("answers", {})
        job_description = request_data.get("jobDescription", "")
        
        preparation_plan = plan_generator.create_preparation_plan(
            answers, 
            job_description
        )
        
        return JSONResponse(content=preparation_plan)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)