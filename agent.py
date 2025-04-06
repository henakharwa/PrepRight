import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from typing import Dict, Any, List
import logging
import re
# Set the base path for file operations
base_path = "/Users/shubhamjain/Documents/project/interview-prep-agent/src"
# Configure logging with a detailed format for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class ResumeParser:
    """Handles parsing and extraction of information from resumes."""
    
    def __init__(self):
        # Define patterns to extract different sections from the resume
        self.section_patterns = {
            'education': r'Education\s*(.*?)(?=\n\w+:|$)',
            'experience': r'Experience\s*(.*?)(?=\n\w+:|$)',
            'skills': r'Technical Skills\s*(.*?)(?=\n\w+:|$)',
            'projects': r'Projects\s*(.*?)(?=\n\w+:|$)'
        }
    def extract_section(self, text: str, pattern: str) -> str:
        """Extracts a specific section from the resume text using regex."""
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """Parses resume text into structured sections."""
        parsed_data = {}
        # Extract each section using the defined patterns
        for section, pattern in self.section_patterns.items():
            parsed_data[section] = self.extract_section(resume_text, pattern)
        return parsed_data
class JobDescriptionParser:
    """Handles parsing and extraction of information from job descriptions."""
    
    def parse_job_description(self, job_description_text: str) -> Dict[str, Any]:
        """Parses job description text into structured sections."""
        # This is a placeholder implementation. You can add more detailed parsing logic here.
        return {
            'necessary_qualifications': re.findall(r'\b(?:Python|Java|SQL|AWS|GCP|ML|AI|NLP|PyTorch|TensorFlow)\b', job_description_text),
            'responsibilities': re.findall(r'\b(?:Develop|Design|Implement|Maintain|Test)\b', job_description_text)
        }
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
class InterviewPreparationAgent:
    """Main agent class for generating interview preparation advice."""
    
    def __init__(self):
        # Create a more concise prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["resume_analysis", "job_analysis", "gap_analysis"],
            template="""Provide a focused interview preparation analysis. Be direct and specific:
Resume Overview:
{resume_analysis}
Job Requirements:
{job_analysis}
Skills Gap:
{gap_analysis}
Provide:
1. Top 3 strengths to emphasize
2. Top 3 gaps to address
3. Key preparation recommendations
4. 2-3 specific talking points for technical discussions"""
        )
        
        try:
            # Use a lower temperature for more focused responses
            self.llm = OpenAI(
                temperature=0.3,
                max_tokens=1000  # Limit response length
            )
            logger.info("Successfully initialized OpenAI LLM")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
            raise
    def _format_list(self, items: List[str]) -> str:
        """Formats a list of items into a bullet-point string."""
        return '\n'.join(f"- {item}" for item in items)
    def _format_resume_analysis(self, resume_data: Dict) -> str:
        """Creates a concise resume summary."""
        skills = resume_data.get('skills_list', [])
        return f"""Key Skills: {', '.join(skills[:5])}... ({len(skills)} total)
Education: {resume_data.get('education', 'Not specified').split('\n')[0]}
Experience Highlights: {resume_data.get('experience', 'Not specified').split('\n')[0]}"""
    def _format_job_analysis(self, job_data: Dict) -> str:
        """Creates a focused job requirement summary."""
        return f"""Core Requirements:
{self._format_list(job_data.get('necessary_qualifications', [])[:3])}
Key Responsibilities:
{self._format_list(job_data.get('responsibilities', [])[:3])}"""
    def _format_gap_analysis(self, gaps: Dict) -> str:
        """Creates a concise gap analysis."""
        return f"""Critical Gaps:
{self._format_list(gaps.get('missing_necessary', [])[:3])}
Key Strengths:
{self._format_list(gaps.get('strengths', [])[:3])}"""
    def prepare_interview_response(self, input_str: str) -> str:
        """Processes input and generates focused interview advice."""
        try:
            # Check if we have the full formatted input string
            if "Please analyze the following information" not in input_str:
                resume = self._read_file_safely(f"{base_path}/resume.txt")
                job_role = self._read_file_safely(f"{base_path}/job_role.txt")
                min_qualifications = self._read_file_safely(f"{base_path}/min_qualifications.txt")
                
                input_str = f"""Please analyze the following information:
Resume:
{resume}
Job Role:
{job_role}
Minimum Qualifications:
{min_qualifications}"""
            sections = self._parse_input_sections(input_str)
            
            if not all(k in sections for k in ['resume', 'job_role', 'min_qualifications']):
                return "Please provide complete information including resume, job role, and minimum qualifications."
            # Parse and analyze with length limits
            resume_parser = ResumeParser()
            job_parser = JobDescriptionParser()
            
            resume_data = resume_parser.parse_resume(sections['resume'])
            job_data = job_parser.parse_job_description(
                sections['job_role'] + '\n' + sections['min_qualifications']
            )
            
            gap_analyzer = GapAnalyzer(resume_data, job_data)
            gaps = gap_analyzer.analyze_gaps()
            
            # Format the analyses with length limits
            prompt_data = {
                'resume_analysis': self._format_resume_analysis(resume_data),
                'job_analysis': self._format_job_analysis(job_data),
                'gap_analysis': self._format_gap_analysis(gaps)
            }
            
            # Generate response with length limit
            prompt = self.prompt_template.format(**prompt_data)
            response = self.llm.invoke(prompt)
            
            return response
        except Exception as e:
            logger.error(f"Error in prepare_interview_response: {str(e)}")
            return f"Error processing input: {str(e)}"
    def _parse_input_sections(self, input_str: str) -> Dict[str, str]:
        """Parses input string into sections with length limits."""
        sections = {}
        current_section = None
        section_text = []
        
        for line in input_str.split('\n'):
            line = line.strip()
            
            if 'Resume:' in line:
                if current_section:
                    sections[current_section] = '\n'.join(section_text)
                current_section = 'resume'
                section_text = []
            elif 'Job Role:' in line:
                if current_section:
                    sections[current_section] = '\n'.join(section_text)
                current_section = 'job_role'
                section_text = []
            elif 'Minimum Qualifications:' in line:
                if current_section:
                    sections[current_section] = '\n'.join(section_text)
                current_section = 'min_qualifications'
                section_text = []
            elif current_section and line:
                section_text.append(line)
        
        if current_section and section_text:
            sections[current_section] = '\n'.join(section_text)
        
        return sections
    def _read_file_safely(self, filepath: str) -> str:
        """Safely reads a file with error handling."""
        try:
            with open(filepath, 'r') as file:
                content = file.read().strip()
            return content
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            return ""
from typing import Dict, Any, List, Optional
import logging
from langchain_openai import OpenAI
class DynamicSkillsAssessor:
    """Generates and evaluates skill assessment questions using OpenAI."""
    
    def __init__(self, llm: OpenAI):
        self.llm = llm
        self.question_generation_template = """
        Create {num_questions} multiple choice questions to assess knowledge in {skill}. 
        Each question should:
        - Test practical understanding
        - Have 4 options with one correct answer
        - Include a detailed explanation of the correct answer
        - Be at an intermediate to advanced level
        
        Format the response as a Python list of dictionaries with the following structure:
        [
            {
                'question': 'Question text',
                'options': ['option1', 'option2', 'option3', 'option4'],
                'correct': correct_option_index,  # 0-based index
                'explanation': 'Detailed explanation'
            }
        ]
        
        Focus on real-world applications and problem-solving scenarios rather than just theoretical concepts.
        """
    def generate_assessment(self, skills: List[str], num_questions: int = 3) -> List[Dict]:
        """Generates skill assessment questions using OpenAI."""
        all_questions = []
        
        for skill in skills:
            prompt = self.question_generation_template.format(
                num_questions=min(2, num_questions),  # Generate 1-2 questions per skill
                skill=skill
            )
            
            try:
                # Generate questions using OpenAI
                response = self.llm.invoke(prompt)
                # Parse the response into a Python object
                questions = eval(response)  # Note: In production, use safer parsing
                
                # Add skill information to each question
                for question in questions:
                    question['skill'] = skill
                
                all_questions.extend(questions)
                
            except Exception as e:
                logging.error(f"Error generating questions for {skill}: {str(e)}")
                continue
            
            if len(all_questions) >= num_questions:
                break
                
        return all_questions[:num_questions]
    def evaluate_responses(self, questions: List[Dict], responses: List[int]) -> Dict[str, Any]:
        """Processes user responses and generates detailed feedback."""
        results = {
            'total_score': 0,
            'feedback': [],
            'skill_scores': {}
        }
        
        for i, (question, response) in enumerate(zip(questions, responses)):
            is_correct = response == question['correct']
            skill = question['skill']
            
            if skill not in results['skill_scores']:
                results['skill_scores'][skill] = {'correct': 0, 'total': 0}
            
            results['skill_scores'][skill]['total'] += 1
            if is_correct:
                results['total_score'] += 1
                results['skill_scores'][skill]['correct'] += 1
            
            feedback_prompt = f"""
            Based on the user's {'correct' if is_correct else 'incorrect'} answer to:
            Question: {question['question']}
            
            Provide:
            1. A detailed explanation of why the answer is {'correct' if is_correct else 'incorrect'}
            2. Key concepts to review
            3. A practical tip for applying this knowledge in interviews
            """
            
            try:
                feedback_response = self.llm.invoke(feedback_prompt)
                
                results['feedback'].append({
                    'question_num': i + 1,
                    'skill': skill,
                    'correct': is_correct,
                    'explanation': question['explanation'],
                    'detailed_feedback': feedback_response
                })
            except Exception as e:
                logging.error(f"Error generating feedback: {str(e)}")
                results['feedback'].append({
                    'question_num': i + 1,
                    'skill': skill,
                    'correct': is_correct,
                    'explanation': question['explanation']
                })
        
        return results
class DynamicPlanGenerator:
    """Generates personalized learning plans using OpenAI."""
    
    def __init__(self, llm: OpenAI):
        self.llm = llm
        
    def create_weekly_plan(
        self, 
        gap_analysis: Dict, 
        assessment_results: Dict,
        resume_data: Optional[Dict] = None
    ) -> str:
        """Creates a detailed weekly plan using OpenAI."""
        
        plan_prompt = f"""
        Create a detailed weekly interview preparation plan based on the following information:
        Skill Gaps: {gap_analysis['missing_necessary']}
        Skill Assessment Results: {assessment_results['skill_scores']}
        Current Skills: {gap_analysis['strengths']}
        
        The plan should:
        1. Cover 7 days with 2-3 focused activities per day
        2. Include specific resources (courses, tutorials, practice problems)
        3. Prioritize skills based on assessment performance
        4. Balance theoretical learning with practical exercises
        5. Include interview preparation activities
        6. Suggest specific projects or exercises
        7. Include time estimates for each activity
        8. Add checkpoints for self-assessment
        
        Format the response as a detailed schedule with clear daily objectives and specific actionable tasks.
        Include additional recommendations for interview preparation and long-term learning.
        """
        
        try:
            weekly_plan = self.llm.invoke(plan_prompt)
            return weekly_plan
        except Exception as e:
            logging.error(f"Error generating weekly plan: {str(e)}")
            return "Error generating weekly plan. Please try again."
class EnhancedInterviewPreparationAgent:
    """Enhanced agent that uses OpenAI for dynamic content generation."""
    
    def __init__(self):
        self.llm = OpenAI(
            temperature=0.7,  # Slightly higher for more creative responses
            max_tokens=2000   # Increased for detailed plans
        )
        self.skills_assessor = DynamicSkillsAssessor(self.llm)
        self.plan_generator = DynamicPlanGenerator(self.llm)
        
    def prepare_complete_analysis(self, input_str: str, assessment_responses: List[int] = None) -> Dict[str, Any]:
        """Generates complete analysis including dynamic assessment and planning."""
        try:
            # Parse and analyze input (similar to previous version)
            sections = self._parse_input_sections(input_str)
            resume_parser = ResumeParser()
            job_parser = JobDescriptionParser()
            
            resume_data = resume_parser.parse_resume(sections['resume'])
            job_data = job_parser.parse_job_description(
                sections['job_role'] + '\n' + sections['min_qualifications']
            )
            
            gap_analyzer = GapAnalyzer(resume_data, job_data)
            gaps = gap_analyzer.analyze_gaps()
            
            # Generate dynamic assessment questions
            assessment_questions = self.skills_assessor.generate_assessment(
                gaps['missing_necessary']
            )
            
            if assessment_responses:
                # Evaluate responses and generate personalized plan
                assessment_results = self.skills_assessor.evaluate_responses(
                    assessment_questions, assessment_responses
                )
                weekly_plan = self.plan_generator.create_weekly_plan(
                    gaps, assessment_results, resume_data
                )
            else:
                assessment_results = None
                weekly_plan = None
            
            return {
                'gaps': gaps,
                'assessment_questions': assessment_questions,
                'assessment_results': assessment_results,
                'weekly_plan': weekly_plan
            }
            
        except Exception as e:
            logging.error(f"Error in prepare_complete_analysis: {str(e)}")
            return f"Error processing input: {str(e)}"
def main():
    """Main execution function."""
    try:
        # Set up OpenAI API key
        os.environ["OPENAI_API_KEY"] = "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"
        
        # Initialize and run the agent
        interview_agent = InterviewPreparationAgent()
        
        # Execute analysis and print results
        response = interview_agent.prepare_interview_response("")
        print("\nInterview Preparation Analysis:")
        print(response)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
if __name__ == "__main__":
    main()
