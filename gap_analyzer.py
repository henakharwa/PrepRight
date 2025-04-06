from typing import Dict, List,Any

class GapAnalyzer:
    """Analyzes gaps between resume and job description."""
    
    def __init__(self, resume_data: Dict[str, Any], job_data: Dict[str, Any]):
        self.resume_data = resume_data
        self.job_data = job_data

    def analyze_gaps(self) -> Dict[str, List[str]]:
        """Analyzes and identifies gaps between resume and job description."""
        resume_skills = set(self.resume_data.get('skills', '').split(', '))
        job_skills = set(self.job_data.get('necessary_qualifications', []))
        
        missing_necessary = list(job_skills - resume_skills)
        strengths = list(resume_skills & job_skills)
        
        return {
            'missing_necessary': missing_necessary,
            'strengths': strengths
        }