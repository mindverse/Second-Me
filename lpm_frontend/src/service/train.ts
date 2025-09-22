import { Request } from '../utils/request';
import type { CommonResponse, EmptyResponse } from '../types/responseModal';

interface ProcessInfo {
  cmdline: string[];
  cpu_percent: string;
  create_time: string;
  memory_percent: string;
  pid: string;
}

export interface ServiceStatusRes {
  process_info?: ProcessInfo;
  is_running?: boolean;
}

interface StartTrainResponse {
  progress_id: string;
}

export type StepStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'suspended';
export type StageStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'suspended';

interface TrainStep {
  completed: boolean;
  current_file?: string;
  name: string;
  status: StepStatus;
  path?: string;
  have_output?: boolean;
}

interface TrainStage {
  name: string;
  progress: number;
  status: StageStatus;
  steps: TrainStep[];
  current_step: string | null;
}
export interface TrainStepJson {
  content: object[];
  file_type: 'json';
}

export interface TrainStepParquet {
  columns: string[];
  content: object[];
  file_type: 'parquet';
}
export type TrainStepOutput = TrainStepJson | TrainStepParquet;

export type StageName =
  | 'downloading_the_base_model'
  | 'activating_the_memory_matrix'
  | 'synthesize_your_life_narrative'
  | 'prepare_training_data_for_deep_comprehension'
  | 'training_to_create_second_me';

export type StageDisplayName =
  | 'Downloading the Base Model'
  | 'Activating the Memory Matrix'
  | 'Synthesize Your Life Narrative'
  | 'Prepare Training Data for Deep Comprehension'
  | 'Training to create Second Me';

export interface TrainProgress {
  stages: TrainStage[];
  overall_progress: number;
  current_stage: StageName;
  status: StageStatus;
}

export interface TrainAdvanceParams {
  is_cot?: boolean;
}

export interface TrainingParams {
  concurrency_threads?: number;
  data_synthesis_mode?: string;
  learning_rate?: number;
  number_of_epochs?: number;
  use_cuda?: boolean;
}

export interface TrainBaseParams {
  model_name: string;
}

export type TrainingConfig = TrainingParams & TrainAdvanceParams & TrainBaseParams;

export const startTrain = (config: TrainingConfig) => {
  return Request<CommonResponse<StartTrainResponse>>({
    method: 'post',
    url: '/api/trainprocess/start',
    data: config
  });
};

export const getTrainProgress = (config: TrainingConfig) => {
  return Request<CommonResponse<TrainProgress>>({
    method: 'get',
    url: `/api/trainprocess/progress/${config.model_name}`
  });
};

export const resetProgress = () => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'post',
    url: `/api/trainprocess/progress/reset`
  });
};

export const stopTrain = () => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'post',
    url: `/api/trainprocess/stop`
  });
};

export const retrain = (config: TrainingConfig) => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'post',
    url: `/api/trainprocess/retrain`,
    data: config
  });
};

export const startService = (config: TrainingConfig) => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'post',
    url: `/api/kernel2/llama/start`,
    data: config
  });
};

export const getServiceStatus = () => {
  return Request<CommonResponse<ServiceStatusRes>>({
    method: 'get',
    url: `/api/kernel2/llama/status`
  });
};

export const stopService = () => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'post',
    url: `/api/kernel2/llama/stop`
  });
};

export const getTrainingParams = () => {
  return Request<CommonResponse<TrainingConfig>>({
    method: 'get',
    url: `/api/trainprocess/training_params`
  });
};

export const checkCudaAvailability = () => {
  return Request<
    CommonResponse<{
      cuda_available: boolean;
      cuda_info: {
        device_count?: number;
        current_device?: number;
        device_name?: string;
      };
    }>
  >({
    method: 'get',
    url: '/api/kernel2/cuda/available'
  });
};

export const getStepOutputContent = (stepName: string) => {
  return Request<CommonResponse<TrainStepOutput>>({
    method: 'get',
    url: `/api/trainprocess/step_output_content?step_name=${stepName}`
  });
};
