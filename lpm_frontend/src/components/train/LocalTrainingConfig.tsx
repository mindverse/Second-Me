'use client';

import type React from 'react';
import { Fragment, useMemo, useState, useEffect } from 'react';
import { Listbox, Transition } from '@headlessui/react';
import { QuestionCircleOutlined } from '@ant-design/icons';
import { InputNumber, message, Radio, Tooltip } from 'antd';
import type { LocalTrainingParams } from '@/service/train';
import { EVENT } from '../../utils/event';
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

interface LocalTrainingConfigProps {
  baseModelOptions: BaseModelOption[];
  modelConfig: ModelConfig | null;
  isTraining: boolean;
  updateTrainingParams: (params: Partial<LocalTrainingParams>) => void;
  status: string;
  trainSuspended: boolean;
  trainingParams: LocalTrainingParams;
  cudaAvailable: boolean;
}

const synthesisModeOptions = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' }
];

const LocalTrainingConfig: React.FC<LocalTrainingConfigProps> = ({
  baseModelOptions,
  modelConfig,
  isTraining,
  updateTrainingParams,
  trainingParams,
  status,
  trainSuspended,
  cudaAvailable
}) => {
  const [openThinkingModel, setOpenThinkingModel] = useState<boolean>(false);
  const [showThinkingWarning, setShowThinkingWarning] = useState<boolean>(false);
  const thinkingModelConfig = useModelConfigStore((state) => state.thinkingModelConfig);

  useEffect(() => {
    if (baseModelOptions.length > 0) {
      const currentModelIsValid = baseModelOptions.some(
        (m) => m.value === trainingParams.model_name
      );

      if (!trainingParams.model_name || !currentModelIsValid) {
        const defaultModel = baseModelOptions[0].value;

        if (trainingParams.model_name !== defaultModel) {
          updateTrainingParams({
            model_name: defaultModel
          });
        }
      }
    } else {
      if (trainingParams.model_name !== '') {
        updateTrainingParams({
          model_name: ''
        });
      }
    }
  }, [baseModelOptions, trainingParams, updateTrainingParams]);

  // Initialize language parameter if not set
  useEffect(() => {
    if (!trainingParams.language) {
      updateTrainingParams({
        ...trainingParams,
        language: 'english'
      });
    }
  }, [trainingParams, updateTrainingParams]);

  useEffect(() => {
    // 如果 is_cot 不是 true，则设置为 true
    if (trainingParams.is_cot !== true) {
      updateTrainingParams({ ...trainingParams, is_cot: true });
    }
  }, [trainingParams.is_cot, updateTrainingParams]);

  const disabledChangeParams = useMemo(() => {
    return isTraining || trainSuspended;
  }, [isTraining, trainSuspended]);

  const thinkingConfigComplete = useMemo(() => {
    return (
      !!thinkingModelConfig.thinking_model_name &&
      !!thinkingModelConfig.thinking_endpoint
    );
  }, [thinkingModelConfig]);

  return (
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
              Low: Fast data synthesis. Medium: Balanced synthesis and speed. High: Rich speed.
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
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
            onChange={(value) => {
              if (value !== trainingParams.model_name) {
                updateTrainingParams({
                  model_name: value
                });
              }
            }}
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
                    ?.label ||
                    (baseModelOptions.length > 0 ? baseModelOptions[0].label : 'Select a model...')}
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
                  {baseModelOptions.length === 0 ? (
                    <div className="py-2 px-4 text-center text-sm text-gray-500">
                      No models available.
                    </div>
                  ) : (
                    baseModelOptions.map((option) => (
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
                    ))
                  )}
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

            <div className="flex flex-col gap-2 mt-4">
              <div className="flex gap-3 items-center">
                <div className="font-medium">Training Language</div>
                <Tooltip title="Select the language for training data synthesis and model responses. This affects how the AI processes and generates content in your chosen language.">
                  <QuestionCircleOutlined className="cursor-pointer" />
                </Tooltip>
              </div>
              <Listbox
                disabled={disabledChangeParams}
                onChange={(value) => {
                  if (value !== trainingParams.language) {
                    updateTrainingParams({
                      ...trainingParams,
                      language: value
                    });
                  }
                }}
                value={trainingParams.language || 'english'}
              >
                <div className="relative mt-1">
                  <Listbox.Button
                    className={classNames(
                      'relative w-[300px] cursor-pointer rounded-lg bg-white py-2 pl-3 pr-10 text-left border border-gray-300 focus:outline-none focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-blue-300',
                      disabledChangeParams && 'opacity-50 !cursor-not-allowed'
                    )}
                  >
                    <span className="block truncate">
                      {trainingParams.language === 'chinese' ? 'Chinese (中文)' : 'English'}
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
                    <Listbox.Options className="absolute mt-1 max-h-60 w-[300px] overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 z-[1] focus:outline-none">
                      <Listbox.Option
                        className={({ active }) =>
                          `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                            active ? 'bg-blue-100 text-blue-900' : 'text-gray-900'
                          }`
                        }
                        value="english"
                      >
                        {({ selected }) => (
                          <>
                            <span
                              className={`block truncate ${
                                selected ? 'font-medium' : 'font-normal'
                              }`}
                            >
                              English
                            </span>
                            {selected ? (
                              <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-blue-600">
                                <DoneIcon className="h-5 w-5" />
                              </span>
                            ) : null}
                          </>
                        )}
                      </Listbox.Option>
                      <Listbox.Option
                        className={({ active }) =>
                          `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                            active ? 'bg-blue-100 text-blue-900' : 'text-gray-900'
                          }`
                        }
                        value="chinese"
                      >
                        {({ selected }) => (
                          <>
                            <span
                              className={`block truncate ${
                                selected ? 'font-medium' : 'font-normal'
                              }`}
                            >
                              Chinese (中文)
                            </span>
                            {selected ? (
                              <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-blue-600">
                                <DoneIcon className="h-5 w-5" />
                              </span>
                            ) : null}
                          </>
                        )}
                      </Listbox.Option>
                    </Listbox.Options>
                  </Transition>
                </div>
              </Listbox>
              <div className="text-xs text-gray-500">
                Choose the primary language for training data and model responses
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-3">
          <div className="text-base font-semibold text-gray-800 flex items-center">
            Step 4: Configure Thinking Model
          </div>

          <div className="flex mr-auto gap-2 items-center ">
            {/* 移除复选框，但保留配置按钮 */}
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
              Configure Thinking Model
            </div>
            <Tooltip
              title="Configure how your Second Me thinks through complex questions. This is a required feature."
              placement="top"
            >
              <QuestionCircleOutlined className="text-gray-400 cursor-help" />
            </Tooltip>
          </div>
        </div>

        {/* Step 5: Configure Data Filtering */}
        <div className="flex flex-col gap-3">
          <div className="text-base font-semibold text-gray-800 flex items-center">
            Step 5: Configure Data Filtering
          </div>

          <div className="flex flex-col gap-4">
            {/* Model Selection */}
            <div className="flex flex-col gap-2">
              <div className="text-sm font-medium text-gray-700">Filtering Model</div>
              <Listbox
                value={trainingParams.data_filtering_model || 'gemma:2b'}
                onChange={(value) => updateTrainingParams({ data_filtering_model: value })}
                disabled={disabledChangeParams}
              >
                <div className="relative">
                  <Listbox.Button
                    className={classNames(
                      'relative w-full cursor-pointer rounded-md bg-white py-2 pl-3 pr-10 text-left border focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500',
                      disabledChangeParams
                        ? 'opacity-50 cursor-not-allowed border-gray-200'
                        : 'border-gray-300 hover:border-gray-400'
                    )}
                  >
                    <span className="block truncate">
                      {trainingParams.data_filtering_model || 'gemma:2b'}
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
                    <Listbox.Options className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                      {[
                        { value: 'gemma:2b', label: 'Gemma 2B' },
                        { value: 'gemma2:9b', label: 'Gemma2 9B' },
                        { value: 'gemma3:4b', label: 'Gemma3 4B' },
                        { value: 'gemma3:12b', label: 'Gemma3 12B' },
                        { value: 'gemma:7b', label: 'Gemma 7B' }
                      ].map((model) => (
                        <Listbox.Option
                          key={model.value}
                          className={({ active }) =>
                            `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                              active ? 'bg-blue-100 text-blue-900' : 'text-gray-900'
                            }`
                          }
                          value={model.value}
                        >
                          {({ selected }) => (
                            <>
                              <span
                                className={`block truncate ${
                                  selected ? 'font-medium' : 'font-normal'
                                }`}
                              >
                                {model.label}
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
              <div className="text-xs text-gray-500">
                Choose the Ollama model for intelligent data filtering and quality assessment
              </div>
            </div>

            {/* Workers Configuration */}
            <div className="flex flex-col gap-2">
              <div className="text-sm font-medium text-gray-700">Number of Workers</div>
              <InputNumber
                min={1}
                max={10}
                value={trainingParams.data_filtering_workers || 5}
                onChange={(value) => updateTrainingParams({ data_filtering_workers: value || 5 })}
                disabled={disabledChangeParams}
                className="w-full"
                placeholder="5"
              />
              <div className="text-xs text-gray-500">
                Number of concurrent workers for data filtering (1-10, default: 5)
              </div>
            </div>

            {/* Keep Ratio Configuration */}
            <div className="flex flex-col gap-2">
              <div className="text-sm font-medium text-gray-700">Keep Ratio</div>
              <InputNumber
                min={0.1}
                max={1.0}
                step={0.1}
                value={trainingParams.data_filtering_keep_ratio || 0.8}
                onChange={(value) => updateTrainingParams({ data_filtering_keep_ratio: value || 0.8 })}
                disabled={disabledChangeParams}
                className="w-full"
                placeholder="0.8"
                formatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                parser={(value) => Number(value!.replace('%', '')) / 100}
              />
              <div className="text-xs text-gray-500">
                Percentage of data to keep after filtering (10%-100%, default: 80%)
              </div>
            </div>
          </div>
        </div>
      </div>

      <ThinkingModelModal onClose={() => setOpenThinkingModel(false)} open={openThinkingModel} />
    </div>
  );
};

export default LocalTrainingConfig;
