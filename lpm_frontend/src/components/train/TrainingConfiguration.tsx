'use client';

import type React from 'react';
import { Fragment, useMemo, useState } from 'react';
import { Listbox, Transition } from '@headlessui/react';
import { PlayIcon, StopIcon } from '@heroicons/react/24/outline';
import { EVENT } from '../../utils/event';
import { Checkbox, InputNumber, message, Radio, Spin, Tooltip } from 'antd';
import type { TrainingConfig } from '@/service/train';
import { QuestionCircleOutlined } from '@ant-design/icons';
import OpenAiModelIcon from '../svgs/OpenAiModelIcon';
import CustomModelIcon from '../svgs/CustomModelIcon';
import ColumnArrowIcon from '../svgs/ColumnArrowIcon';
import DoneIcon from '../svgs/DoneIcon';
import ThinkingModelModal from '../ThinkingModelModal';
import { useModelConfigStore } from '@/store/useModelConfigStore';
import classNames from 'classnames';

interface BaseModelOption {
  value: string;
  label: string;
}

interface ModelConfig {
  provider_type?: string;
  [key: string]: any;
}

interface TrainingConfigurationProps {
  baseModelOptions: BaseModelOption[];
  modelConfig: ModelConfig | null;
  isTraining: boolean;
  updateTrainingParams: (params: TrainingConfig) => void;
  status: string;
  trainSuspended: boolean;
  handleResetProgress: () => void;
  handleTrainingAction: () => Promise<void>;
  trainActionLoading: boolean;
  setSelectedInfo: React.Dispatch<React.SetStateAction<boolean>>;
  trainingParams: TrainingConfig;
  cudaAvailable: boolean;
}

const synthesisModeOptions = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' }
];

const TrainingConfiguration: React.FC<TrainingConfigurationProps> = ({
  baseModelOptions,
  modelConfig,
  isTraining,
  updateTrainingParams,
  trainingParams,
  status,
  handleResetProgress,
  trainSuspended,
  trainActionLoading,
  handleTrainingAction,
  setSelectedInfo,
  cudaAvailable
}) => {
  const [openThinkingModel, setOpenThinkingModel] = useState<boolean>(false);
  const [showThinkingWarning, setShowThinkingWarning] = useState<boolean>(false);
  const thinkingModelConfig = useModelConfigStore((state) => state.thinkingModelConfig);

  const disabledChangeParams = useMemo(() => {
    return isTraining || trainSuspended;
  }, [isTraining, trainSuspended]);

  const thinkingConfigComplete = useMemo(() => {
    return (
      !!thinkingModelConfig.thinking_model_name &&
      !!thinkingModelConfig.thinking_api_key &&
      !!thinkingModelConfig.thinking_endpoint
    );
  }, [thinkingModelConfig]);

  const trainButtonText = useMemo(() => {
    return isTraining
      ? 'Stop Training'
      : status === 'trained'
        ? 'Retrain'
        : trainSuspended
          ? 'Resume Training'
          : 'Start Training';
  }, [isTraining, status, trainSuspended]);

  const trainButtonIcon = useMemo(() => {
    return isTraining ? (
      trainActionLoading ? (
        <Spin className="h-5 w-5 mr-2" />
      ) : (
        <StopIcon className="h-5 w-5 mr-2" />
      )
    ) : (
      <PlayIcon className="h-5 w-5 mr-2" />
    );
  }, [isTraining, trainActionLoading]);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold tracking-tight text-gray-900">
          Training Configuration
        </h2>
        <button
          className="p-1.5 rounded-full bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-700 transition-colors"
          onClick={() => setSelectedInfo(true)}
          title="Learn more about training process"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
            />
          </svg>
        </button>
      </div>
      <p className="text-gray-600 mb-6 leading-relaxed">
        {`Configure how your Second Me will be trained using your memory data and identity. Then click 'Start Training'.`}
      </p>

      <div className="space-y-6">
        <div className="flex flex-col gap-10">
          <div className="flex flex-col gap-2">
            <h4 className="text-base font-semibold text-gray-800 flex items-center">
              Step 1: Choose Support Model for Data Synthesis
            </h4>
            {!modelConfig?.provider_type ? (
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <label className="block text-sm font-medium text-red-500 mb-1">
                    None Support Model for Data Synthesis
                  </label>
                  <button
                    className="ml-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors cursor-pointer relative z-10"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      window.dispatchEvent(new CustomEvent(EVENT.SHOW_MODEL_CONFIG_MODAL));
                    }}
                  >
                    Configure Support Model
                  </button>
                </div>
                <span className="text-xs text-gray-500">
                  Model used for processing and synthesizing your memory data
                </span>
              </div>
            ) : (
              <div className="flex items-center relative w-full rounded-lg bg-white py-2 text-left">
                <div className="flex items-center">
                  <span className="text-sm font-medium text-gray-700">Model Used : &nbsp;</span>
                  {modelConfig.provider_type === 'openai' ? (
                    <OpenAiModelIcon className="h-5 w-5 mr-2 text-green-600" />
                  ) : (
                    <CustomModelIcon className="h-5 w-5 mr-2 text-blue-600" />
                  )}
                  <span className="font-medium">
                    {modelConfig.provider_type === 'openai' ? 'OpenAI' : 'Custom Model'}
                  </span>
                  <button
                    className={classNames(
                      'ml-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors cursor-pointer relative z-10',
                      disabledChangeParams && 'opacity-50 !cursor-not-allowed'
                    )}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();

                      if (disabledChangeParams) {
                        message.warning('Cancel the current training run to configure the model');

                        return;
                      }

                      window.dispatchEvent(new CustomEvent(EVENT.SHOW_MODEL_CONFIG_MODAL));
                    }}
                  >
                    Configure Model for Data Synthesis
                  </button>
                </div>
                <span className="ml-auto text-xs text-gray-500">
                  Model used for processing and synthesizing your memory data
                </span>
              </div>
            )}
            <div className="flex flex-col gap-3">
              <div className="font-medium">Data Synthesis Mode</div>
              <Radio.Group
                disabled={disabledChangeParams}
                onChange={(e) =>
                  updateTrainingParams({
                    ...trainingParams,
                    data_synthesis_mode: e.target.value
                  })
                }
                optionType="button"
                options={synthesisModeOptions}
                value={trainingParams.data_synthesis_mode}
              />

              <span className="text-xs text-gray-500">
                Low: Fast data synthesis. Medium: Balanced synthesis and speed. High: Rich
                synthesis, slower speed.
              </span>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <h4 className="text-base font-semibold text-gray-800 mb-1">
                Step 2: Choose Base Model for Training Second Me
              </h4>
              <span className="text-xs text-gray-500">
                Base model for training your Second Me. Choose based on your available system
                resources.
              </span>
            </div>
            <Listbox
              disabled={disabledChangeParams}
              onChange={(value) => updateTrainingParams({ model_name: value })}
              value={trainingParams.model_name}
            >
              <div className="relative mt-1">
                <Listbox.Button
                  className={classNames(
                    'relative w-full cursor-pointer rounded-lg bg-white py-2 pl-3 pr-10 text-left border border-gray-300 focus:outline-none focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-blue-300',
                    disabledChangeParams && 'opacity-50 !cursor-not-allowed'
                  )}
                >
                  <span className="block truncate">
                    {baseModelOptions.find((option) => option.value === trainingParams.model_name)
                      ?.label || 'Select a model...'}
                  </span>
                  <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                    <ColumnArrowIcon className="h-5 w-5 text-gray-400" />
                  </span>
                </Listbox.Button>
                <Transition
                  as={Fragment}
                  leave="transition ease-in duration-100"
                  leaveFrom="opacity-100"
                  leaveTo="opacity-0"
                >
                  <Listbox.Options className="absolute mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 z-[1] focus:outline-none">
                    {baseModelOptions.map((option) => (
                      <Listbox.Option
                        key={option.value}
                        className={({ active }) =>
                          `relative cursor-pointer select-none py-2 pl-10 pr-4 ${active ? 'bg-blue-100 text-blue-900' : 'text-gray-900'}`
                        }
                        value={option.value}
                      >
                        {({ selected }) => (
                          <>
                            <span
                              className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}
                            >
                              {option.label}
                            </span>
                            {selected ? (
                              <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-blue-600">
                                <DoneIcon className="h-5 w-5" />
                              </span>
                            ) : null}
                          </>
                        )}
                      </Listbox.Option>
                    ))}
                  </Listbox.Options>
                </Transition>
              </div>
            </Listbox>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center">
              <h4 className="text-base font-semibold text-gray-800 mb-1">
                Step 3: Configure Advanced Training Parameters
              </h4>
              <div className="text-xs text-gray-500">
                Adjust these parameters to control training quality and performance. Recommended
                settings will ensure stable training.
              </div>
            </div>
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-2">
                <div className="flex gap-3 items-center">
                  <div className="font-medium">Learning Rate</div>
                  <Tooltip title="Lower values provide stable but slower learning, while higher values accelerate learning but risk overshooting optimal parameters, potentially causing training instability.">
                    <QuestionCircleOutlined className="cursor-pointer" />
                  </Tooltip>
                </div>
                <InputNumber
                  className="!w-[300px]"
                  disabled={disabledChangeParams}
                  max={0.005}
                  min={0.00003}
                  onChange={(value) => {
                    if (value == null) {
                      return;
                    }

                    updateTrainingParams({ ...trainingParams, learning_rate: value });
                  }}
                  status={
                    trainingParams.learning_rate == 0.005 || trainingParams.learning_rate == 0.00003
                      ? 'warning'
                      : undefined
                  }
                  step={0.0001}
                  value={trainingParams.learning_rate}
                />
                <div className="text-xs text-gray-500">
                  Enter a value between 0.00003 and 0.005 (recommended: 0.0001)
                </div>
              </div>
              <div className="flex flex-col gap-2">
                <div className="flex gap-3 items-center">
                  <div className="font-medium">Number of Epochs</div>
                  <Tooltip title="Controls how many complete passes the model makes through your entire dataset during training. More epochs allow deeper pattern recognition and memory integration but significantly increase training time and computational resources required.">
                    <QuestionCircleOutlined className="cursor-pointer" />
                  </Tooltip>
                </div>
                <InputNumber
                  className="!w-[300px]"
                  disabled={disabledChangeParams}
                  max={10}
                  min={1}
                  onChange={(value) => {
                    if (value == null) {
                      return;
                    }

                    updateTrainingParams({ ...trainingParams, number_of_epochs: value });
                  }}
                  status={
                    trainingParams.number_of_epochs == 10 || trainingParams.number_of_epochs == 1
                      ? 'warning'
                      : undefined
                  }
                  step={1}
                  value={trainingParams.number_of_epochs}
                />
                <div className="text-xs text-gray-500">
                  Enter an integer between 1 and 10 (recommended: 3)
                </div>
              </div>
              <div className="flex flex-col gap-2">
                <div className="flex gap-3 items-center">
                  <div className="font-medium">Concurrency Threads</div>
                  <Tooltip title="Defines the number of parallel processing streams used during data synthesis. Higher values can reduce overall training time but increase system resource consumption and may trigger API rate limits, potentially causing training failures.">
                    <QuestionCircleOutlined className="cursor-pointer" />
                  </Tooltip>
                </div>
                <InputNumber
                  className="!w-[300px]"
                  disabled={disabledChangeParams}
                  max={10}
                  min={1}
                  onChange={(value) => {
                    if (value == null) {
                      return;
                    }

                    updateTrainingParams({ ...trainingParams, concurrency_threads: value });
                  }}
                  status={
                    trainingParams.concurrency_threads == 10 ||
                    trainingParams.concurrency_threads == 1
                      ? 'warning'
                      : undefined
                  }
                  step={1}
                  value={trainingParams.concurrency_threads}
                />
                <div className="text-xs text-gray-500">
                  Enter an integer between 1 and 10 (recommended: 2)
                </div>
              </div>

              <div className="flex flex-col gap-2 mt-4">
                <div className="flex gap-3 items-center">
                  <div className="font-medium">Enable CUDA GPU Acceleration</div>
                  <Tooltip title="When enabled, training will use CUDA GPU acceleration if available on your system. This can significantly speed up training but requires compatible NVIDIA hardware and drivers.">
                    <QuestionCircleOutlined className="cursor-pointer" />
                  </Tooltip>
                </div>
                <div className="flex items-center">
                  <label className="inline-flex items-center cursor-pointer relative">
                    <input
                      checked={trainingParams.use_cuda}
                      className="sr-only peer"
                      disabled={disabledChangeParams || !cudaAvailable}
                      onChange={(e) => {
                        updateTrainingParams({ ...trainingParams, use_cuda: e.target.checked });
                      }}
                      type="checkbox"
                    />
                    <div
                      className={`w-11 h-6 ${!cudaAvailable ? 'bg-gray-300' : 'bg-gray-200'} peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all ${cudaAvailable ? 'peer-checked:bg-blue-600' : 'peer-checked:bg-gray-400'}`}
                    />
                    <span
                      className={`ms-3 text-sm font-medium ${!cudaAvailable ? 'text-gray-500' : 'text-gray-700'}`}
                    >
                      {trainingParams.use_cuda ? 'Enabled' : 'Disabled'}
                    </span>
                  </label>
                </div>
                <div className="text-xs text-gray-500">
                  {cudaAvailable
                    ? 'Enable for faster training on NVIDIA GPUs.'
                    : 'CUDA acceleration is not available on this system.'}
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-3">
            <div className="text-base font-semibold text-gray-800 flex items-center">
              Step 4: Configure Advanced Behavior
            </div>

            <div className="flex mr-auto gap-2 items-center ">
              <Checkbox
                checked={trainingParams.is_cot}
                disabled={disabledChangeParams}
                onChange={(e) => {
                  e.stopPropagation();

                  if (!thinkingConfigComplete) {
                    setShowThinkingWarning(true);

                    if (!showThinkingWarning) {
                      setTimeout(() => setShowThinkingWarning(false), 2000);
                    }

                    return;
                  }

                  updateTrainingParams({ ...trainingParams, is_cot: e.target.checked });
                }}
              />
              <div
                className={classNames(
                  `text-sm font-medium px-4 py-2 bg-white border rounded-md cursor-pointer transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]`,
                  showThinkingWarning
                    ? 'border-red-500 text-red-600 bg-red-50 shadow-[0_0_0_2px_rgba(220,38,38,0.4)] animate-pulse'
                    : 'border-gray-300 text-gray-700 hover:bg-gray-50',
                  disabledChangeParams && 'opacity-50 !cursor-not-allowed'
                )}
                onClick={() => {
                  if (disabledChangeParams) return;

                  setOpenThinkingModel(true);
                }}
              >
                Thinking Model 
              </div>
              <Tooltip title="Chain of Thought (CoT) enables the model to perform step-by-step reasoning during training. This improves the quality of responses by allowing the model to 'think' through complex questions before answering, resulting in more accurate and logically coherent outputs.">
                    <QuestionCircleOutlined className="cursor-pointer ml-2" />
              </Tooltip>
            </div>
          </div>
        </div>

        <div className="flex justify-end items-center gap-4 pt-4 border-t mt-4">
          {isTraining && (
            <div className="flex items-center text-amber-600 bg-amber-50 px-3 py-2 rounded-md border border-amber-200">
              <StopIcon className="h-5 w-5 mr-2" />
              <span className="font-medium">Full stop only when the current step is complete</span>
            </div>
          )}

          {trainButtonText === 'Resume Training' && (
            <button
              className={`inline-flex items-center justify-center px-4 py-2 bg-red-600 hover:bg-red-700 border border-transparent text-sm font-medium rounded-md shadow-sm text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
              onClick={() => {
                if (!isTraining) {
                  message.warning('Please do not shutdown your computer during training');
                }

                handleResetProgress();
              }}
            >
              <StopIcon className="h-5 w-5 mr-2" />
              Reset Training
            </button>
          )}
          <button
            className={`inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${isTraining ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'}
            ${!isTraining && !modelConfig?.provider_type ? 'bg-gray-300 hover:bg-gray-400 cursor-not-allowed' : 'cursor-pointer'}
            focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
            disabled={!isTraining && !modelConfig?.provider_type}
            onClick={() => {
              if (!isTraining) {
                message.warning('Please do not shutdown your computer during training');
              }

              handleTrainingAction();
            }}
          >
            {trainButtonIcon}
            {trainButtonText}
          </button>
        </div>
      </div>

      <ThinkingModelModal onClose={() => setOpenThinkingModel(false)} open={openThinkingModel} />
    </div>
  );
};

export default TrainingConfiguration;
