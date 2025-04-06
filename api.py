from flask import Flask, request, jsonify
from agent import InterviewPreparationAgent, EnhancedInterviewPreparationAgent

app = Flask(__name__)

# Initialize the agents
interview_agent = InterviewPreparationAgent()
enhanced_interview_agent = EnhancedInterviewPreparationAgent()

@app.route('/prepare-interview', methods=['POST'])
def prepare_interview():
    data = request.json
    input_str = data.get('input_str', '')
    response = interview_agent.prepare_interview_response(input_str)
    return jsonify({'response': response})

@app.route('/complete-analysis', methods=['POST'])
def complete_analysis():
    data = request.json
    input_str = data.get('input_str', '')
    assessment_responses = data.get('assessment_responses', [])
    response = enhanced_interview_agent.prepare_complete_analysis(input_str, assessment_responses)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)