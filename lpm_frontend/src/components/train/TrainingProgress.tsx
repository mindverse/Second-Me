import type { TrainProgress } from '@/service/train';
import type { IStepOutputInfo } from '../trainExposureModel';
import TrainExposureModel from '../trainExposureModel';
import { useState } from 'react';
import classNames from 'classnames';

interface TrainingProgressProps {
  trainingProgress: TrainProgress;
  status: string;
}

const descriptionMap = [
  'At this stage, we obtain the foundational model that will serve as the starting point for your Second Me. This base structure is a blank slate, ready to be shaped and enriched with your personal data, acting as the vessel that will eventually carry your unique presence.',
  "This step starts by processing and organizing your memories into a structured digital format that forms the groundwork for your Second Me. We break down your life experiences into smaller, meaningful pieces, encode them systematically, and extract essential insights to create a solid base. It's the first move toward building an entity that reflects your past and present.",
  "Here, we take the fragments of your memories and weave them into a complete, flowing biography that captures your essence. This process connects the dots between your experiences, shaping them into a coherent story that defines who you are. It's like crafting the blueprint of a new being born from your life's journey.",
  "To enable your Second Me to understand you fully, we create specialized training data tailored to your unique profile. This step lays the groundwork for it to grasp your preferences, identity, and knowledge accurately, ensuring the entity we're constructing can think and respond in ways that feel authentic to you.",
  'Finally, we train the core model with your specific memories, traits, and preferences, blending them seamlessly into its framework. This step transforms the model into a living representation of you, merging technology with your individuality to create a Second Me that feels real and true to your essence.'
];

const TrainingProgress = (props: TrainingProgressProps) => {
  const { trainingProgress, status } = props;

  const [stepOutputInfo, setStepOutputInfo] = useState<IStepOutputInfo>({} as IStepOutputInfo);

  const formatUnderscoreToName = (_str: string) => {
    const str = _str || '';

    return str
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const formatToUnderscore = (str: string): string => {
    if (!str) return '';

    return str.toLowerCase().replace(/\s+/g, '_');
  };

  const trainingStages = trainingProgress.stages.map((stage, index) => {
    return { ...stage, description: descriptionMap[index] };
  });

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900">
          Training Progress (may take long with more data and larger model)
        </h3>
        {status === 'trained' && (
          <span className="px-2.5 py-1 bg-green-50 text-green-700 text-sm font-medium rounded-full">
            Training Complete
          </span>
        )}
      </div>
      <div className="space-y-6">
        {/* Overall Progress */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-lg font-semibold text-gray-900">Overall Progress</span>
              <span className="text-2xl font-bold text-blue-600">
                {Math.round(trainingProgress.overall_progress)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${trainingProgress.overall_progress}%` }}
              />
            </div>
          </div>
        </div>

        {/* All Training Stages */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-700">Training Stages</h4>
          <div className="space-y-4">
            {trainingStages.map((stage) => {
              const stageStatus = stage.status;
              const progress = stage.progress;

              // Handle NaN case
              const displayProgress = isNaN(progress) ? 0 : progress;

              const isCurrentStage =
                formatUnderscoreToName(trainingProgress.current_stage) == stage.name;

              return (
                <div
                  key={stage.name}
                  className="bg-white rounded-lg border border-gray-100 p-4 shadow-sm"
                >
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="flex-shrink-0">
                      {stageStatus === 'completed' ? (
                        <div className="w-6 h-6 rounded-full bg-green-100 flex items-center justify-center">
                          <svg
                            className="w-4 h-4 text-green-600"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              d="M5 13l4 4L19 7"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth="2"
                            />
                          </svg>
                        </div>
                      ) : stageStatus === 'in_progress' ? (
                        <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
                          <div className="w-3 h-3 rounded-full bg-blue-600 animate-pulse" />
                        </div>
                      ) : (
                        <div className="w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center">
                          <div className="w-3 h-3 rounded-full bg-gray-300" />
                        </div>
                      )}
                    </div>

                    <div className="flex-grow">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center">
                          <span
                            className={`text-sm font-medium ${
                              isCurrentStage ? 'text-blue-700' : 'text-gray-700'
                            }`}
                          >
                            {stage.name}
                            {isCurrentStage && stage.current_step && (
                              <span className="ml-2 text-xs text-gray-500">
                                {formatUnderscoreToName(stage.current_step)}
                              </span>
                            )}
                          </span>
                          <button
                            className="ml-1 p-1 rounded-full text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors"
                            onClick={() => {
                              const modal = document.createElement('div');

                              modal.className =
                                'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
                              modal.innerHTML = `
                                <div class="bg-white rounded-xl max-w-md p-6 m-4 space-y-4 relative shadow-xl">
                                  <h3 class="text-xl font-semibold">${stage.name}</h3>
                                  <div class="space-y-4 text-gray-600">
                                    <p>${stage.description}</p>
                                  </div>
                                  <button class="absolute top-4 right-4 p-2 text-gray-400 hover:text-gray-600" onclick="this.parentElement.parentElement.remove()">
                                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                  </button>
                                </div>
                              `;
                              document.body.appendChild(modal);
                              modal.onclick = (e) => {
                                if (e.target === modal) modal.remove();
                              };
                            }}
                            title="Learn more about this stage"
                          >
                            <svg
                              className="w-3 h-3"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                              />
                            </svg>
                          </button>
                        </div>
                        <span className="text-xs text-gray-500">
                          {Math.round(displayProgress)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-100 rounded-full h-1.5 mt-1">
                        <div
                          className={`h-1.5 rounded-full transition-all duration-500 ${
                            stageStatus === 'completed'
                              ? 'bg-green-500'
                              : stageStatus === 'in_progress'
                                ? 'bg-blue-500'
                                : displayProgress > 0
                                  ? 'bg-blue-300'
                                  : 'bg-gray-200'
                          }`}
                          style={{ width: `${displayProgress}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Step list */}
                  <div className="mt-3 pl-9">
                    {stage.steps.length > 0 ? (
                      <div className="space-y-2">
                        {stage.steps.map((step, stepIndex) => (
                          <div key={stepIndex} className="flex items-center space-x-2">
                            <div className="flex-shrink-0">
                              {step.completed ? (
                                <div className="w-4 h-4 rounded-full bg-green-100 flex items-center justify-center">
                                  <svg
                                    className="w-3 h-3 text-green-600"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      d="M5 13l4 4L19 7"
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth="2"
                                    />
                                  </svg>
                                </div>
                              ) : stage.current_step &&
                                formatUnderscoreToName(stage.current_step) == step.name ? (
                                <div className="w-4 h-4 rounded-full bg-blue-100 flex items-center justify-center">
                                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                                </div>
                              ) : (
                                <div className="w-4 h-4 rounded-full bg-gray-100 flex items-center justify-center">
                                  <div className="w-2 h-2 rounded-full bg-gray-300" />
                                </div>
                              )}
                            </div>
                            <span
                              className={classNames(
                                'text-xs',
                                stage.current_step &&
                                  formatUnderscoreToName(stage.current_step) == step.name
                                  ? 'text-blue-600 font-medium'
                                  : 'text-gray-600'
                                // step.completed ? 'hover:text-green-600 cursor-pointer' : ''
                              )}
                            >
                              {step.name}
                            </span>

                            {/* Current downloading file name */}
                            {step.name === 'Model Download' &&
                            step.status !== 'completed' &&
                            !!step.current_file ? (
                              <span
                                className={classNames('text-xs ml-2', 'text-blue-600 font-medium')}
                              >
                                <span className="mr-2">{'-'}</span>
                                <span>{step.current_file}</span>
                              </span>
                            ) : null}

                            {step.completed && step.have_output && (
                              <span
                                className="text-xs text-blue-500 underline cursor-pointer hover:text-blue-600"
                                onClick={() => {
                                  setStepOutputInfo({
                                    stepName: formatToUnderscore(step.name),
                                    path: step.path
                                  });
                                }}
                              >
                                View Resources
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : stageStatus !== 'pending' ? (
                      <div className="text-xs text-gray-500">Processing...</div>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <TrainExposureModel
        handleClose={() => setStepOutputInfo({} as IStepOutputInfo)}
        stepOutputInfo={stepOutputInfo}
      />
    </div>
  );
};

export default TrainingProgress;
