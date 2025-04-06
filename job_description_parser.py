import re
from typing import Dict, Any,List

class JobDescriptionParser:
    """Handles parsing and extraction of information from job descriptions."""
    
    def parse_job_description(self, job_description_text: str) -> Dict[str, Any]:
        """Parses job description text into structured sections."""
        # This is a placeholder implementation. You can add more detailed parsing logic here.
        return {
            'necessary_qualifications': re.findall(r'\b(?:Python|Java|SQL|AWS|GCP|ML|AI|NLP|PyTorch|TensorFlow)\b', job_description_text),
            'responsibilities': re.findall(r'\b(?:Develop|Design|Implement|Maintain|Test)\b', job_description_text)
        }

def extract_required_skills(parsed_data: Dict) -> List[str]:
    """
    Extracts the required skills from the parsed job description data.
    
    Args:
        parsed_data (Dict): The parsed job description data.
        
    Returns:
        List[str]: A list of required skills.
    """
    return parsed_data.get("required_skills", [])

def extract_qualifications(parsed_data: Dict) -> List[str]:
    """
    Extracts the qualifications from the parsed job description data.
    
    Args:
        parsed_data (Dict): The parsed job description data.
        
    Returns:
        List[str]: A list of qualifications.
    """
    return parsed_data.get("qualifications", [])

def extract_technologies(parsed_data: Dict) -> List[str]:
    """
    Extracts the technologies from the parsed job description data.
    
    Args:
        parsed_data (Dict): The parsed job description data.
        
    Returns:
        List[str]: A list of technologies.
    """
    return parsed_data.get("technologies", [])