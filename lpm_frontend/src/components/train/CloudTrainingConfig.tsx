'use client';

import type React from 'react';
import { Fragment, useEffect, useMemo, useState } from 'react';
import { Listbox, Transition } from '@headlessui/react'; // Removed HeadlessRadioGroup
import { QuestionCircleOutlined } from '@ant-design/icons';
import { InputNumber, message, Tooltip, Spin, Radio } from 'antd';
import type { RadioChangeEvent } from 'antd'; // Added RadioChangeEvent for typing
import type { CloudTrainingParams } from '@/service/train';
import { EVENT } from '../../utils/event';
import OpenAiModelIcon from '../svgs/OpenAiModelIcon';
import CustomModelIcon from '../svgs/CustomModelIcon';
import AlibabaCloudIcon from '../svgs/AlibabaCloudIcon';
import ColumnArrowIcon from '../svgs/ColumnArrowIcon';
import DoneIcon from '../svgs/DoneIcon';
import ThinkingModelModal from '../ThinkingModelModal';
import { useCloudProviderStore } from '@/store/useCloudProviderStore';
import { useModelConfigStore } from '@/store/useModelConfigStore';
import classNames from 'classnames';
import CloudProviderModal from '../modelConfigModal/CloudProviderModal';
import { listAvailableModels } from '@/service/cloudService';
import type { CloudModel } from '@/service/cloudService';
import type { IModelConfig } from '@/service/modelConfig';

interface CloudTrainingConfigProps {
  modelConfig: IModelConfig | null;
  isTraining: boolean;
  updateTrainingParams: (params: Partial<CloudTrainingParams>) => void;
  status: string;
  trainSuspended: boolean;
  trainingParams: CloudTrainingParams;
  cudaAvailable: boolean;
}

const synthesisModeOptions = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' }
];

const defaultCloudTrainingParams: CloudTrainingParams = {
  model_name: '',
  base_model: '',
  data_synthesis_mode: 'medium',
  hyper_parameters: {
    learning_rate: 0.0001,
    n_epochs: 3
  },
  language: 'english',
  is_cot: true // 添加 is_cot 默认值
};

const CloudTrainingConfig: React.FC<CloudTrainingConfigProps> = ({
  modelConfig,
  isTraining,
  updateTrainingParams,
  trainingParams,
  trainSuspended
}) => {
  const [openThinkingModel, setOpenThinkingModel] = useState<boolean>(false);
  const [openCloudProviderModal, setOpenCloudProviderModal] = useState<boolean>(false);
  const [availableCloudModels, setAvailableCloudModels] = useState<CloudModel[]>([]);
  const [loadingModels, setLoadingModels] = useState<boolean>(false);
  const cloudConfig = useCloudProviderStore((state) => state.cloudConfig);
  const updateCloudConfig = useCloudProviderStore((state) => state.updateCloudConfig);
  const fetchCloudConfig = useCloudProviderStore((state) => state.fetchCloudConfig);

  // 获取 thinkingModelConfig
  const thinkingModelConfig = useModelConfigStore((store) => store.thinkingModelConfig);

  // 检查 Think Model 配置是否完整
  const thinkingConfigComplete = useMemo(() => {
    return (
      !!thinkingModelConfig?.thinking_model_name && !!thinkingModelConfig?.thinking_endpoint
      // 不再检查 thinking_api_key
    );
  }, [thinkingModelConfig]);

  // 初始化参数，确保所有参数都有默认值
  useEffect(() => {
    // 检查参数并设置默认值（如果缺少）
    const mergedParams = {
      ...defaultCloudTrainingParams,
      ...trainingParams,
      hyper_parameters: {
        ...defaultCloudTrainingParams.hyper_parameters,
        ...(trainingParams.hyper_parameters || {})
      },
      language: trainingParams.language || defaultCloudTrainingParams.language,
      is_cot: true // 确保 is_cot 始终为 true
    };

    // 只有在值不同时才更新
    if (JSON.stringify(mergedParams) !== JSON.stringify(trainingParams)) {
      updateTrainingParams(mergedParams);
    }
  }, []);

  const disabledChangeParams = useMemo(() => {
    return isTraining || trainSuspended;
  }, [isTraining, trainSuspended]);

  const hasCloudServiceApiKey = useMemo(() => {
    return !!(
      (modelConfig && modelConfig.cloud_service_api_key) ||
      (cloudConfig && cloudConfig.cloud_service_api_key)
    );
  }, [modelConfig, cloudConfig]);

  useEffect(() => {
    // This effect manages cloudConfig.provider_type based on available API key
    // Only update when the provider_type or API key has actually changed (not during input)
    const hasApiKey = !!(
      (modelConfig && modelConfig.cloud_service_api_key) ||
      (cloudConfig && cloudConfig.cloud_service_api_key && cloudConfig.provider_type)
    );

    if (hasApiKey && cloudConfig.provider_type === 'alibaba') {
      // Only update if not already correctly set
      const currentApiKey =
        modelConfig?.cloud_service_api_key || cloudConfig.cloud_service_api_key || '';
      if (cloudConfig.cloud_service_api_key !== currentApiKey) {
        updateCloudConfig({
          ...cloudConfig,
          provider_type: 'alibaba',
          cloud_service_api_key: currentApiKey
        });
      }
    } else if (!hasApiKey && cloudConfig.provider_type !== '') {
      updateCloudConfig({
        ...cloudConfig,
        provider_type: '',
        cloud_service_api_key: ''
      });
    }
  }, [modelConfig?.cloud_service_api_key, cloudConfig.provider_type]);

  useEffect(() => {
    const fetchModels = async () => {
      if (hasCloudServiceApiKey) {
        setLoadingModels(true);
        try {
          const res = await listAvailableModels();
          if (res.data.code === 0 && Array.isArray(res.data.data)) {
            const fetchedModels: CloudModel[] = res.data.data.map((item: any) => ({
              model_id: item.id,
              model_name: item.id
            }));
            setAvailableCloudModels(fetchedModels);

            if (fetchedModels.length > 0) {
              let validCloudModelName = false;
              if (trainingParams.base_model) {
                validCloudModelName = fetchedModels.some(
                  (m) => m.model_id === trainingParams.base_model
                );
              }

              const currentModelIsValid = fetchedModels.some(
                (m) => m.model_id === trainingParams.model_name
              );

              let modelToUse = '';

              if (validCloudModelName) {
                modelToUse = trainingParams.base_model as string;
              } else if (currentModelIsValid) {
                modelToUse = trainingParams.model_name;
              } else {
                modelToUse = fetchedModels[0].model_id;
              }

              if (
                trainingParams.model_name !== modelToUse ||
                trainingParams.base_model !== modelToUse
              ) {
                updateTrainingParams({
                  ...trainingParams,
                  model_name: modelToUse,
                  base_model: modelToUse
                });
              }
            } else {
              if (trainingParams.model_name !== '' || trainingParams.base_model !== '') {
                updateTrainingParams({
                  ...trainingParams,
                  model_name: '',
                  base_model: ''
                });
              }
            }
          } else {
            message.error(res.data.message || 'Failed to fetch available cloud models');
            setAvailableCloudModels([]);
            if (trainingParams.model_name !== '' || trainingParams.base_model !== '') {
              updateTrainingParams({
                ...trainingParams,
                model_name: '',
                base_model: ''
              });
            }
          }
        } catch (error: any) {
          // Added type for error
          message.error(error.message || 'Error fetching cloud models');
          setAvailableCloudModels([]);

          if (trainingParams.model_name !== '' || trainingParams.base_model !== '') {
            updateTrainingParams({
              ...trainingParams,
              model_name: '',
              base_model: ''
            });
          }

          console.error('Failed to fetch cloud models:', error);
        }
        setLoadingModels(false);
      } else {
        setAvailableCloudModels([]);
        if (trainingParams.model_name !== '' || trainingParams.base_model !== '') {
          updateTrainingParams({
            ...trainingParams,
            model_name: '',
            base_model: ''
          });
        }
      }
    };

    fetchModels();
  }, [hasCloudServiceApiKey, cloudConfig.provider_type, cloudConfig.cloud_service_api_key]); // 添加依赖项

  // useEffect(() => {
  //   if (trainingParams.cloud_model_name && trainingParams.model_name !== trainingParams.cloud_model_name) {
  //     updateTrainingParams({
  //       ...trainingParams,
  //       model_name: trainingParams.cloud_model_name
  //     });
  //   }
  // }, [trainingParams, updateTrainingParams]);

  const cloudModelOptions = useMemo(() => {
    return availableCloudModels.map((model) => ({
      value: model.model_id,
      label: model.model_name
    }));
  }, [availableCloudModels]);

  const handleSynthesisModeChange = (e: RadioChangeEvent) => {
    // Now data_synthesis_mode is part of CloudTrainingParams
    updateTrainingParams({
      ...trainingParams,
      data_synthesis_mode: e.target.value
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-10">
        {/* Step 1: Choose Cloud Provider */}
        <div className="flex flex-col gap-2">
          <h4 className="text-base font-semibold text-gray-800 flex items-center">
            Step 1: Choose Cloud Provider
          </h4>
          {!hasCloudServiceApiKey ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <label className="block text-sm font-medium text-red-500 mb-1">
                  None Cloud Provider Configured
                </label>
                <button
                  className="ml-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors cursor-pointer relative z-10"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setOpenCloudProviderModal(true);
                  }}
                >
                  Configure Cloud Provider
                </button>
              </div>
              <span className="text-xs text-gray-500">
                Configure your cloud training with Alibaba Cloud Model Studio
              </span>
            </div>
          ) : (
            <div className="flex flex-col rounded-lg bg-white py-2 text-left">
              <div className="flex items-center">
                <span className="text-sm font-medium text-gray-700">
                  Cloud Service Used : &nbsp;
                </span>
                <AlibabaCloudIcon className="h-5 w-5 mr-2 text-orange-600" />
                <span className="font-medium">Alibaba Cloud Model Studio</span>
                <button
                  className={classNames(
                    'ml-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600 transition-colors cursor-pointer relative z-10',
                    disabledChangeParams && 'opacity-50 !cursor-not-allowed'
                  )}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();

                    if (disabledChangeParams) {
                      message.warning(
                        'Cancel the current training run to configure the cloud provider'
                      );

                      return;
                    }

                    setOpenCloudProviderModal(true);
                  }}
                >
                  Configure Cloud Provider for Model Training
                </button>
              </div>
              <div className="mt-2 text-xs text-gray-500">
                This will provide faster training but requires API keys and may incur charges based
                on token usage
              </div>
            </div>
          )}
        </div>

        {/* Step 2: Choose Support Model for Data Synthesis */}
        <div className="flex flex-col gap-2">
          <h4 className="text-base font-semibold text-gray-800 flex items-center">
            Step 2: Choose Support Model for Data Synthesis
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
              onChange={handleSynthesisModeChange} // Use typed handler
              optionType="button"
              options={synthesisModeOptions}
              value={
                trainingParams.data_synthesis_mode || defaultCloudTrainingParams.data_synthesis_mode
              }
            />
            <span className="text-xs text-gray-500">
              Low: Fast data synthesis. Medium: Balanced synthesis and speed. High: Rich speed.
            </span>
          </div>
        </div>

        {/* Step 3: Choose Base Model for Training Second Me */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <h4 className="text-base font-semibold text-gray-800 mb-1">
              Step 3: Choose Base Model for Training Second Me
            </h4>
            <span className="text-xs text-gray-500">Base model for training your Second Me.</span>
          </div>
          {loadingModels ? (
            <div className="flex justify-center items-center h-10">
              <Spin />
            </div>
          ) : (
            <Listbox
              disabled={disabledChangeParams || !hasCloudServiceApiKey || loadingModels}
              onChange={(value) => {
                if (value !== trainingParams.base_model) {
                  updateTrainingParams({
                    ...trainingParams,
                    model_name: value,
                    base_model: value
                  });
                }
              }}
              value={trainingParams.base_model || trainingParams.model_name || ''}
            >
              <div className="relative mt-1">
                <Listbox.Button
                  className={classNames(
                    'relative w-full cursor-pointer rounded-lg bg-white py-2 pl-3 pr-10 text-left border border-gray-300 focus:outline-none focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-blue-300',
                    disabledChangeParams || !hasCloudServiceApiKey || loadingModels
                      ? 'opacity-50 !cursor-not-allowed'
                      : 'hover:bg-gray-50'
                  )}
                >
                  <span className="block truncate">
                    {(() => {
                      const currentOption = cloudModelOptions.find(
                        (option) => option.value === trainingParams.base_model
                      );

                      if (!currentOption && trainingParams.model_name) {
                        const modelOption = cloudModelOptions.find(
                          (option) => option.value === trainingParams.model_name
                        );
                        if (modelOption) return modelOption.label;
                      }

                      return (
                        currentOption?.label ||
                        (cloudModelOptions.length > 0
                          ? cloudModelOptions[0].label
                          : 'No models available')
                      );
                    })()}
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
                    {cloudModelOptions.length === 0 ? (
                      <div className="py-2 px-4 text-center text-sm text-gray-500">
                        No models available. Check API key or provider.
                      </div>
                    ) : (
                      cloudModelOptions.map((option) => (
                        <Listbox.Option
                          key={option.value}
                          className={({ active }) =>
                            `relative cursor-pointer select-none py-2 pl-10 pr-4 ${active ? 'bg-blue-100 text-blue-900' : 'text-gray-900'}`
                          }
                          value={option.value}
                        >
                          {({ selected }) => {
                            const isSelected =
                              selected ||
                              trainingParams.base_model === option.value ||
                              trainingParams.model_name === option.value;
                            return (
                              <>
                                <span
                                  className={`block truncate ${isSelected ? 'font-medium' : 'font-normal'}`}
                                >
                                  {option.label}
                                </span>
                                {isSelected ? (
                                  <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-blue-600">
                                    <DoneIcon className="h-5 w-5" />
                                  </span>
                                ) : null}
                              </>
                            );
                          }}
                        </Listbox.Option>
                      ))
                    )}
                  </Listbox.Options>
                </Transition>
              </div>
            </Listbox>
          )}
          {!hasCloudServiceApiKey && (
            <p className="text-xs text-red-500">
              Please configure Cloud Provider in Step 1 to see available models.
            </p>
          )}
          {hasCloudServiceApiKey && !loadingModels && cloudModelOptions.length === 0 && (
            <p className="text-xs text-orange-500">
              No base models available for the selected cloud provider or API key.
            </p>
          )}
        </div>

        {/* Step 4: Configure Advanced Training Parameters */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <h4 className="text-base font-semibold text-gray-800 mb-1">
              Step 4: Configure Advanced Training Parameters
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
                  if (value == null) return;
                  updateTrainingParams({
                    ...trainingParams,
                    hyper_parameters: {
                      ...trainingParams.hyper_parameters,
                      learning_rate: value
                    }
                  });
                }}
                status={
                  trainingParams.hyper_parameters?.learning_rate === 0.005 ||
                  trainingParams.hyper_parameters?.learning_rate === 0.00003
                    ? 'warning'
                    : undefined
                }
                step={0.0001}
                value={
                  trainingParams.hyper_parameters?.learning_rate ||
                  (defaultCloudTrainingParams.hyper_parameters?.learning_rate ?? 0.0001)
                }
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
                  if (value == null) return;
                  updateTrainingParams({
                    ...trainingParams,
                    hyper_parameters: {
                      ...trainingParams.hyper_parameters,
                      n_epochs: value
                    }
                  });
                }}
                status={
                  trainingParams.hyper_parameters?.n_epochs === 10 ||
                  trainingParams.hyper_parameters?.n_epochs === 1
                    ? 'warning'
                    : undefined
                }
                step={1}
                value={
                  trainingParams.hyper_parameters?.n_epochs ||
                  (defaultCloudTrainingParams.hyper_parameters?.n_epochs ?? 3)
                }
              />
              <div className="text-xs text-gray-500">
                Enter an integer between 1 and 10 (recommended: 3)
              </div>
            </div>
            <div className="flex flex-col gap-2">
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
                value={trainingParams.language || defaultCloudTrainingParams.language}
              >
                <div className="relative mt-1">
                  <Listbox.Button
                    className={classNames(
                      'relative w-[300px] cursor-pointer rounded-lg bg-white py-2 pl-3 pr-10 text-left border border-gray-300 focus:outline-none focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-blue-300',
                      disabledChangeParams && 'opacity-50 !cursor-not-allowed'
                    )}
                  >
                    <span className="block truncate">
                      {(trainingParams.language || defaultCloudTrainingParams.language) ===
                      'chinese'
                        ? 'Chinese (中文)'
                        : 'English'}
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

        {/* Step 5: Configure Thinking Model */}
        <div className="flex flex-col gap-3">
          <div className="text-base font-semibold text-gray-800 flex items-center">
            Step 5: Configure Thinking Model
          </div>

          <div className="flex mr-auto gap-2 items-center">
            <div
              className={classNames(
                `text-sm font-medium px-4 py-2 bg-white border rounded-md cursor-pointer transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)]`,
                disabledChangeParams
                  ? 'opacity-50 !cursor-not-allowed'
                  : 'border-gray-300 text-gray-700 hover:bg-gray-50'
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

        {/* Step 6: Configure Data Filtering */}
        <div className="flex flex-col gap-3">
          <div className="text-base font-semibold text-gray-800 flex items-center">
            Step 6: Configure Data Filtering
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
      <CloudProviderModal
        open={openCloudProviderModal}
        onClose={() => {
          setOpenCloudProviderModal(false);
          setTimeout(() => {
            fetchCloudConfig();
          }, 100);
        }}
        cloudConfig={cloudConfig} // Pass current cloudConfig from store
        updateCloudConfig={updateCloudConfig} // Pass update function from store
        saveCloudConfig={async () => {
          // Implement saveCloudConfig to align with modal props
          // This function in the modal is expected to save the config (e.g., call an API)
          // Here, we assume the modal itself handles the API call via its internal handleUpdate
          // and then calls this callback. We just need to ensure the local state/store is updated.
          fetchCloudConfig(); // Re-fetch config to update the store and related states
        }}
      />
    </div>
  );
};

export default CloudTrainingConfig;
