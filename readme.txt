# Interview Preparation Agent

This project provides an interview preparation agent that helps users prepare for job interviews by analyzing resumes and job descriptions. It includes a backend implemented in Python and a frontend implemented in React.

## Table of Contents

- Installation
- Usage
- API Endpoints
- Frontend Integration
- Contributing
- License

## Installation

### Backend

1. Clone the repository:
   git clone https://github.com/yourusername/interview-prep-agent.git
   cd interview-prep-agent

2. Create a virtual environment and activate it:
   python3 -m venv venv
   source venv/bin/activate

3. Install the required dependencies:
   pip install -r requirements.txt

4. Run the Flask API:
   python api.py

### Frontend

1. Navigate to the frontend directory:
   cd interview-prep-frontend

2. Install the required dependencies:
   npm install

3. Start the development server:
   npm start

## Usage

1. Upload your resume (PDF format) and enter the job description in the provided text area.
2. Click on "Prepare Interview" to get interview preparation suggestions.
3. Click on "Complete Analysis" to get a detailed analysis based on the job description and your resume.

## API Endpoints

### POST /prepare-interview

- Description: Prepares interview suggestions based on the job description.
- Request Body:
  {
    "input_str": "Job description text"
  }
- Response:
  {
    "response": "Interview preparation suggestions"
  }

### POST /complete-analysis

- Description: Provides a detailed analysis based on the job description and assessment responses.
- Request Body:
  {
    "input_str": "Job description text",
    "assessment_responses": ["response1", "response2"]
  }
- Response:
  {
    "response": "Detailed analysis"
  }

## Frontend Integration

The frontend is implemented in React and interacts with the backend API to provide a seamless user experience. The main component is `InterviewPrepApp.tsx`, which handles file uploads, job description input, and API calls.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.