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
                
                # Return the questions array directly
                questions = parsed_response['questions']
                
                # Ensure each question has required fields
                for i, q in enumerate(questions, 1):
                    q['id'] = i  # Ensure unique IDs
                    if not all(k in q for k in ['id', 'question', 'skill']):
                        raise ValueError(f"Question {i} missing required fields")
                
                return questions
            
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
        skills_text = parsed_data.get('skills', '')  # Get skills text or empty string if not found
        parsed_data['skills_list'] = [
            skill.strip() for skill in 
            re.findall(r'\b[A-Za-z0-9#+\-/]+\b', skills_text)
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
            r'\b(?:Python|Java|SQL|AWS|GCP|ML|AI|NLP|PyTorch|TensorFlow|Kubernetes|Docker|React|Node\.js|Microsoft Office|Excel|Word|Outlook)\b',
            r'\b(?:Machine Learning|Data Science|Cloud Computing|Backend|Frontend|Full Stack|Human Resources|HR Software)\b'
        ]
        
        responsibilities_patterns = [
            r'\b(?:Develop|Design|Implement|Maintain|Test|Create|Build|Manage|Optimize|Assist|Collaborate)\b',
            r'Responsible for\s+([^.]+)'
        ]
        
        # Combine and deduplicate results
        necessary_qualifications = list(set(
            match.group(0) if match.group(0) else match.group(1)
            for pattern in skills_patterns 
            for match in re.finditer(pattern, job_text, re.IGNORECASE)
        ))
        
        if not necessary_qualifications:
            # Fallback if no skills found
            necessary_qualifications = ["General Skills"]
        
        responsibilities = list(set(
            match.group(0) if match.group(0) else match.group(1)
            for pattern in responsibilities_patterns 
            for match in re.finditer(pattern, job_text, re.IGNORECASE)
        ))
        
        if not responsibilities:
            # Fallback if no responsibilities found
            responsibilities = ["General Responsibilities"]
        
        return {
            'necessary_qualifications': necessary_qualifications,
            'responsibilities': responsibilities
        }
class PreparationPlanGenerator:
    def __init__(self, ai_model=None):
        """
        Initialize the preparation plan generator with an optional AI model.
        If no model is provided, defaults to using Gemini.
        """
        if ai_model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.ai_model = genai.GenerativeModel('gemini-1.5-flash')
            except ImportError:
                logger.error("Gemini AI not available. Unable to generate preparation plans.")
                self.ai_model = None
        else:
            self.ai_model = ai_model

    def create_preparation_plan(
        self, 
        answers: Dict[str, str],
        job_description: str,
        duration: int = 7
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive interview preparation plan based on assessment answers.
        """
        if duration not in [7, 14, 21]:
            logger.warning(f"Invalid duration {duration}, defaulting to 7 days")
            duration = 7

        # Create a more analytical prompt that focuses on gap analysis
        prompt = f"""Analyze the following assessment answers and generate a detailed preparation plan.

    Assessment Answers:
    {json.dumps(answers, indent=2)}

    Job Requirements:
    {job_description}

    Based on a thorough analysis of the assessment answers and job requirements, generate a JSON-formatted preparation plan that includes:

    1. A detailed analysis of skill gaps, identifying:
    - Areas where the candidate's responses show room for improvement
    - Specific skills needed for the job that weren't demonstrated strongly
    - The current level and target level for each identified gap

    2. A {duration}-day plan that specifically addresses these gaps

    Return the response in this exact JSON structure:
    {{
        "weak_areas": [
            {{
                "skill": "specific_skill_from_assessment",
                "current_level": "beginner/intermediate/advanced",
                "target_level": "intermediate/advanced",
                "gap_analysis": "brief explanation of why this is a gap"
            }}
        ],
        "daily_plan": [
            {{
                "day": 1,
                "focus_area": "specific_skill_area",
                "tasks": ["task1", "task2"],
                "resources": [
                    {{
                        "name": "resource_name",
                        "url": "official_documentation_url",
                        "type": "documentation/tutorial/practice"
                    }}
                ]
            }}
        ]
    }}

    The weak areas should be directly derived from analyzing the assessment answers, not predefined.
    Each day's focus should align with addressing the identified gaps."""

        try:
            # Generate and process the response
            response = self.ai_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response format
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            try:
                parsed_response = json.loads(response_text)
                
                # Validate structure
                if not isinstance(parsed_response, dict):
                    raise ValueError("Invalid response structure")
                
                # Ensure required sections exist
                if 'weak_areas' not in parsed_response or 'daily_plan' not in parsed_response:
                    raise ValueError("Missing required sections in response")
                
                # Validate the daily plan matches the requested duration
                if len(parsed_response.get('daily_plan', [])) != duration:
                    raise ValueError(f"Daily plan length does not match requested duration of {duration} days")
                
                return parsed_response
                
            except (json.JSONDecodeError, ValueError) as parse_error:
                logger.error(f"JSON Parsing Error: {parse_error}")
                return self._generate_fallback_plan(duration)
                
        except Exception as e:
            logger.error(f"Error generating preparation plan: {e}")
            return self._generate_fallback_plan(duration)
    
    def _generate_fallback_plan(self, duration: int) -> Dict[str, Any]:
        """
        Generate a fallback preparation plan with official documentation URLs.
        """
        weak_areas = [{
            "skill": "General Technical Skills",
            "current_level": "intermediate",
            "target_level": "advanced"
        }]
        
        daily_plan = []
        for day in range(1, duration + 1):
            # Create different focuses and resources based on the day
            if day <= duration // 3:  # First third: Fundamentals
                focus = "Core Fundamentals"
                resources = [
                    {
                        "name": "Python Official Documentation",
                        "url": "https://docs.python.org/3/",
                        "type": "documentation"
                    },
                    {
                        "name": "Python Practice Exercises",
                        "url": "https://www.hackerrank.com/domains/python",
                        "type": "practice"
                    }
                ]
                tasks = [
                    "Review core language concepts",
                    "Complete basic coding exercises",
                    "Study documentation"
                ]
            elif day <= 2 * (duration // 3):  # Second third: Intermediate
                focus = "Data Structures and Algorithms"
                resources = [
                    {
                        "name": "LeetCode Problems",
                        "url": "https://leetcode.com/problemset/all/",
                        "type": "practice"
                    },
                    {
                        "name": "Python Data Structures",
                        "url": "https://docs.python.org/3/tutorial/datastructures.html",
                        "type": "documentation"
                    }
                ]
                tasks = [
                    "Practice algorithmic problems",
                    "Study common data structures",
                    "Implement basic algorithms"
                ]
            else:  # Final third: Advanced
                focus = "Advanced Topics"
                resources = [
                    {
                        "name": "System Design Primer",
                        "url": "https://github.com/donnemartin/system-design-primer",
                        "type": "tutorial"
                    },
                    {
                        "name": "Python Advanced Topics",
                        "url": "https://docs.python.org/3/howto/functional.html",
                        "type": "documentation"
                    }
                ]
                tasks = [
                    "Study advanced Python concepts",
                    "Practice system design",
                    "Work on complex problems"
                ]

            daily_plan.append({
                "day": day,
                "focus_area": f"{focus} - Day {day}",
                "tasks": tasks,
                "resources": resources
            })
        
        return {
            "weak_areas": weak_areas,
            "daily_plan": daily_plan
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
        resume (UploadFile): Uploaded resume PDF file
        jobDescription (str): Job description text from form
    
    Returns:
        JSONResponse: Contains assessment questions, resume skills, and job requirements
    """
    try:
        # Extract resume text from the uploaded PDF
        resume_content = await resume.read()
        resume_text = extract_text_from_pdf(resume_content)
        
        # Parse both the resume and job description
        parsed_resume = resume_parser.parse_resume(resume_text)
        parsed_job = job_parser.parse_job_description(jobDescription)
        
        # Generate assessment questions based on required qualifications
        skills_to_assess = list(set(parsed_job.get('necessary_qualifications', ['General Skills'])))
        assessment_questions = skills_assessor.generate_assessment_questions(skills_to_assess)
        # Return structured response with all components
        return JSONResponse(content={
            "questions": assessment_questions,  # Direct use of questions list
            "resume_skills": parsed_resume.get('skills_list', []),
            "job_requirements": parsed_job
        })
    
    except Exception as e:
        logger.error(f"Error processing initial assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-plan")
async def generate_plan(request_data: Dict):
    """
    Generate personalized interview preparation plan based on assessment answers.
    
    Args:
        request_data (Dict): Contains assessment answers, job description, and plan duration
    
    Returns:
        JSONResponse: Contains structured preparation plan with weak areas and daily tasks
    """
    try:
        # Extract required data from request
        answers = request_data.get("answers", {})
        job_description = request_data.get("jobDescription", "")
        duration = request_data.get("duration", 7)
        
        # Validate duration
        if duration not in [7, 14, 21]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid duration. Must be 7, 14, or 21 days."
            )
        
        # Generate preparation plan with specified duration
        preparation_plan = plan_generator.create_preparation_plan(
            answers, 
            job_description,
            duration
        )
        
        return JSONResponse(content=preparation_plan)
    
    except Exception as e:
        logger.error(f"Error generating preparation plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)