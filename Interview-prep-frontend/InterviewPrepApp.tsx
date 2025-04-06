import React, { useState, ErrorInfo } from 'react';
// Type definitions for better type safety
interface Question {
  id: number;
  question: string;
  skill: string;
}

interface WeakArea {
  skill: string;
  current_level: string;
  target_level: string;
}

interface Resource {
  name: string;
  url: string;
  type: string;
}

interface DailyPlanItem {
  day: number;
  focus_area: string;
  tasks: string[];
  resources: Resource[];  // Changed from string[] to Resource[]
}

interface PreparationPlan {
  weak_areas: WeakArea[];
  daily_plan: DailyPlanItem[];
}

type PlanDuration = 7 | 14 | 21;

const InterviewPrepApp: React.FC = () => {
  // State management with explicit types
  const [selectedDuration, setSelectedDuration] = useState<PlanDuration>(7);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [resume, setResume] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState<string>('');
  const [questions, setQuestions] = useState<Question[]>([]);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [preparationPlan, setPreparationPlan] = useState<PreparationPlan | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  // File upload handler
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      setResume(file);
      setError('');
    } else {
      setError('Please upload a PDF file');
    }
  };

  // Job description handler
  const handleJobDescription = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setJobDescription(event.target.value);
  };

  // Generate questions based on resume and job description
  const generateQuestions = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const formData = new FormData();
      
      if (resume) {
        formData.append('resume', resume, resume.name);
      } else {
        throw new Error('Resume file is required');
      }
      
      if (!jobDescription.trim()) {
        throw new Error('Job description cannot be empty');
      }
      
      formData.append('jobDescription', jobDescription);
  
      const response = await fetch('http://localhost:8000/api/process-initial', {
        method: 'POST',
        body: formData,
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process resume and job description');
      }
  
      const data = await response.json();
      const questionsData = data.questions || [];
      
      const processedQuestions = questionsData.map((q: Question, index: number) => ({
        ...q,
        id: q.id || index + 1
      }));
  
      if (processedQuestions.length === 0) {
        throw new Error('No assessment questions could be generated');
      }
  
      setQuestions(processedQuestions);
      setCurrentPage(2);
    } catch (error: any) {
      setError(error.message || 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle answer submission for individual questions
  const handleAnswerSubmit = (questionId: number, answer: string) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }));
  };

  // Move to duration selection page
  const moveToSelectDuration = () => {
    if (Object.keys(answers).length !== questions.length) {
      setError('Please answer all questions before proceeding');
      return;
    }
    setCurrentPage(3);
  };

  // Generate preparation plan based on assessment answers and selected duration
  const generatePreparationPlan = async () => {
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/generate-plan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          answers,
          jobDescription,
          duration: selectedDuration
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate preparation plan');
      }

      const plan = await response.json();
      setPreparationPlan(plan);
      setCurrentPage(4);
    } catch (error: any) {
      setError(`Error generating preparation plan: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Render first page - File upload and job description
  const renderFirstPage = () => (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Interview Preparation Starter</h2>
      <p className="text-gray-600 mb-6">
        Upload your resume and provide the job description to begin your personalized interview preparation journey.
      </p>
      
      <div className="space-y-4">
        {/* Resume Upload Section */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Resume (PDF only)
          </label>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          {resume && (
            <p className="text-sm text-green-600 mt-2">
              {resume.name} - Ready to upload
            </p>
          )}
        </div>
        
        {/* Job Description Section */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Job Description
          </label>
          <textarea
            value={jobDescription}
            onChange={handleJobDescription}
            placeholder="Paste the complete job description here..."
            className="w-full h-40 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {error && (
          <div className="p-4 bg-red-100 border border-red-300 text-red-700 rounded">
            <p className="font-medium">Error:</p>
            <p>{error}</p>
          </div>
        )}

        <button 
          onClick={generateQuestions}
          disabled={!resume || !jobDescription}
          className={`w-full py-2 px-4 rounded-md transition-all ${
            resume && jobDescription 
              ? 'bg-blue-600 text-white hover:bg-blue-700' 
              : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          }`}
        >
          Continue to Skills Assessment
        </button>
      </div>
    </div>
  );

  // Render second page - Skills Assessment Questions
  const renderSecondPage = () => (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Skills Assessment</h2>
      <p className="text-gray-600 mb-6">
        Answer the following questions to help us understand your current skill level.
      </p>
      
      <div className="space-y-6">
        {questions.map((question) => (
          <div key={question.id} className="bg-gray-50 p-4 rounded-md">
            <p className="font-medium text-gray-800 mb-3">
              {question.question}
            </p>
            <textarea
              value={answers[question.id] || ''}
              onChange={(e) => handleAnswerSubmit(question.id, e.target.value)}
              placeholder="Enter your answer here..."
              className="w-full h-24 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        ))}

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 text-red-700 rounded-md">
            {error}
          </div>
        )}

        <button
          onClick={moveToSelectDuration}
          disabled={Object.keys(answers).length !== questions.length}
          className={`w-full py-2 px-4 rounded-md transition-all ${
            Object.keys(answers).length === questions.length
              ? 'bg-blue-600 text-white hover:bg-blue-700' 
              : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          }`}
        >
          Continue to Plan Duration
        </button>
      </div>
    </div>
  );

  // Render duration selection page
  const renderDurationSelection = () => (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Select Your Preparation Duration</h2>
      <p className="text-gray-600 mb-6">
        Choose how long you would like your preparation plan to be. A longer duration allows for more in-depth preparation and practice.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {[7, 14, 21].map((duration) => (
          <button
            key={duration}
            onClick={() => setSelectedDuration(duration as PlanDuration)}
            className={`p-4 rounded-lg border-2 transition-all ${
              selectedDuration === duration
                ? 'border-blue-600 bg-blue-50 text-blue-700'
                : 'border-gray-200 hover:border-blue-400'
            }`}
          >
            <h3 className="text-xl font-semibold mb-2">{duration} Days</h3>
            <p className="text-sm text-gray-600">
              {duration === 7 && 'Quick preparation'}
              {duration === 14 && 'Balanced approach'}
              {duration === 21 && 'Comprehensive prep'}
            </p>
          </button>
        ))}
      </div>

      {error && (
        <div className="p-4 mb-4 bg-red-50 border border-red-200 text-red-700 rounded-md">
          {error}
        </div>
      )}

      <button
        onClick={generatePreparationPlan}
        className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-all"
      >
        Generate {selectedDuration}-Day Plan
      </button>
    </div>
  );

  // Render fourth page - Preparation Plan
  const renderFourthPage = () => {
    if (!preparationPlan) {
      return (
        <div className="max-w-3xl mx-auto p-6 bg-white rounded-lg shadow-lg text-center">
          <h2 className="text-2xl font-bold text-red-600 mb-4">
            Preparation Plan Could Not Be Generated
          </h2>
          <p className="text-gray-700">
            There was an issue creating your personalized plan. Please try again or contact support.
          </p>
          <button 
            onClick={() => setCurrentPage(1)}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Start Over
          </button>
        </div>
      );
    }

    const weakAreas = preparationPlan.weak_areas || [];
    const dailyPlan = preparationPlan.daily_plan || [];

    return (
      <div className="max-w-3xl mx-auto p-6 bg-white rounded-lg shadow-lg">
        <h2 className="text-3xl font-bold mb-6 text-center text-blue-800">
          Your {selectedDuration}-Day Preparation Plan
        </h2>
        
        {/* Weak Areas Section */}
        <section className="mb-8">
          <h3 className="text-2xl font-semibold mb-4 text-gray-800">
            Areas for Improvement
          </h3>
          {weakAreas.length === 0 ? (
            <p className="text-gray-600 italic">No specific weak areas identified.</p>
          ) : (
            <div className="grid md:grid-cols-2 gap-4">
              {weakAreas.map((area, index) => (
                <div 
                  key={index} 
                  className="bg-yellow-50 p-4 rounded-lg border border-yellow-200"
                >
                  <h4 className="font-bold text-lg mb-2 text-yellow-800">
                    {area.skill}
                  </h4>
                  <p className="text-gray-700">
                    Current Level: {area.current_level}
                  </p>
                  <p className="text-gray-700">
                    Target Level: {area.target_level}
                  </p>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Daily Plan Section */}
        <section>
          <h3 className="text-2xl font-semibold mb-6 text-gray-800">
            {selectedDuration}-Day Preparation Roadmap
          </h3>
          <div className="space-y-6">
            {dailyPlan.map((dayPlan) => (
              <div 
                key={dayPlan.day} 
                className="bg-blue-50 p-5 rounded-lg border border-blue-200 hover:shadow-md transition-all"
              >
                <h4 className="text-xl font-bold mb-3 text-blue-800">
                  Day {dayPlan.day}: {dayPlan.focus_area}
                </h4>
                
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="font-semibold text-gray-700 mb-2">
                      Tasks
                    </h5>
                    <ul className="list-disc list-inside space-y-2 text-gray-600">
                      {dayPlan.tasks.map((task, index) => (
                        <li key={index}>{task}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                  <h5 className="font-semibold text-gray-700 mb-2">
                    Resources
                  </h5>
                  {dayPlan.resources && dayPlan.resources.length > 0 ? (
                    <ul className="list-disc list-inside space-y-2">
                      {dayPlan.resources.map((resource, index) => (
                        <li key={index} className="text-gray-700">
                          <a 
                            href={resource.url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="text-blue-600 hover:underline"
                          >
                            {resource.name}
                          </a>
                          {resource.type && (
                            <span className="text-sm text-gray-500 ml-2">
                              ({resource.type})
                            </span>
                          )}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-500 italic">No resources specified</p>
                  )}
                </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Restart Option */}
        <div className="text-center mt-8">
          <button 
            onClick={() => setCurrentPage(1)}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Start Over
          </button>
        </div>
      </div>
    );
  };


  // Progress Indicator
  const renderProgressIndicator = () => (
    <div className="max-w-2xl mx-auto mb-6">
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className="h-full bg-blue-600 transition-all duration-300 ease-in-out"
          style={{ width: `${(currentPage / 4) * 100}%` }}
        />
      </div>
      <div className="flex justify-between mt-2 text-sm text-gray-600">
        <span className={currentPage >= 1 ? "text-blue-600 font-medium" : ""}>
          Upload
        </span>
        <span className={currentPage >= 2 ? "text-blue-600 font-medium" : ""}>
          Assessment
        </span>
        <span className={currentPage >= 3 ? "text-blue-600 font-medium" : ""}>
          Duration
        </span>
        <span className={currentPage >= 4 ? "text-blue-600 font-medium" : ""}>
          Plan
        </span>
      </div>
    </div>
  );

  // Loading Spinner
  const renderLoadingSpinner = () => (
    <div className="flex items-center justify-center h-64">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-blue-500 mx-auto mb-4"></div>
        <p className="text-gray-600 text-lg">
          Processing your information, please wait...
        </p>
      </div>
    </div>
  );

  // Main render
  return (
    <div className="min-h-screen bg-gray-50 py-10 px-4">
      <div className="container mx-auto">
        {renderProgressIndicator()}
        
        {isLoading ? (
          renderLoadingSpinner()
        ) : (
          <>
            {currentPage === 1 && renderFirstPage()}
            {currentPage === 2 && renderSecondPage()}
            {currentPage === 3 && renderDurationSelection()}
            {currentPage === 4 && renderFourthPage()}
          </>
        )}
      </div>
    </div>
  );
};

export default InterviewPrepApp;