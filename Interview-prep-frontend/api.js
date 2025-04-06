import axios from 'axios';

const API_URL = 'http://localhost:5000';

export const prepareInterview = async (inputStr) => {
  const response = await axios.post(`${API_URL}/prepare-interview`, { input_str: inputStr });
  return response.data;
};

export const completeAnalysis = async (inputStr, assessmentResponses) => {
  const response = await axios.post(`${API_URL}/complete-analysis`, {
    input_str: inputStr,
    assessment_responses: assessmentResponses,
  });
  return response.data;
};