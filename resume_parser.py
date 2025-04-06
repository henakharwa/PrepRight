from typing import Dict, Any
import re

def parse_resume(resume_text: str) -> Dict[str, Any]:
    """
    Parses the resume text and extracts relevant information such as skills, experience, and education.
    
    Args:
        resume_text (str): The text of the resume to parse.
        
    Returns:
        Dict[str, Any]: A dictionary containing extracted information.
    """
    # Example regex patterns for extracting information
    skills_pattern = r'Skills:\s*(.*)'
    experience_pattern = r'Experience:\s*(.*?)(?=Education:|$)'
    education_pattern = r'Education:\s*(.*)'

    skills_match = re.search(skills_pattern, resume_text, re.DOTALL)
    experience_match = re.search(experience_pattern, resume_text, re.DOTALL)
    education_match = re.search(education_pattern, resume_text, re.DOTALL)

    skills = skills_match.group(1).split(',') if skills_match else []
    experience = experience_match.group(1).strip() if experience_match else ""
    education = education_match.group(1).strip() if education_match else ""

    return {
        "skills": [skill.strip() for skill in skills],
        "experience": experience,
        "education": education
    }