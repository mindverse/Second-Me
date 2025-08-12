'use client';

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import InfoModal from '@/components/InfoModal';
import type {
  TrainingConfig,
  LocalTrainingParams,
  CloudTrainingParams,
  TrainingParamsResponse
} from '@/service/train';
import {
  startTrain,
  stopTrain,
  retrain,
  getTrainingParams,
  checkCudaAvailability,
  resetProgress,
  checkStopStatus
} from '@/service/train';
import {
  startCloudTraining,
  getCloudTrainingProgress,
  stopCloudTraining,
  resetCloudTrainingProgress,
  checkCloudTrainingPauseStatus,
  resumeCloudTraining,
  checkCloudTrainingStopStatus
} from '@/service/cloudService';
import type { CloudProgressData } from '@/service/cloudService';
import { useTrainingStore } from '@/store/useTrainingStore';
import { getMemoryList } from '@/service/memory';
import { message, Modal, Button } from 'antd';
import { useModelConfigStore } from '@/store/useModelConfigStore';
import CelebrationEffect from '@/components/Celebration';
import TrainingLog from '@/components/train/TrainingLog';
import TrainingProgress from '@/components/train/TrainingProgress';
import TrainingConfiguration from '@/components/train/TrainingConfiguration';
import { ROUTER_PATH } from '@/utils/router';

interface TrainInfo {
  name: string;
  description: string;
  features: string[];
}

const trainInfo: TrainInfo = {
  name: 'Training Process',
  description:
    'Transform your memories into a personalized AI model through a multi-stage training process',
  features: [
    'Automated multi-stage training process',
    'Real-time progress monitoring',
    'Detailed training logs',
    'Training completion notification',
    'Model performance metrics'
  ]
};

const POLLING_INTERVAL = 3000;

interface TrainingDetail {
  message: string;
  timestamp: string;
}

const baseModelOptions = [
  {
    value: 'Qwen3-0.6B',
    label: 'Qwen3-0.6B (8GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-1.7B',
    label: 'Qwen3-1.7B (16GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-4B',
    label: 'Qwen3-4B (32GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-8B',
    label: 'Qwen3-8B (64GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-14B',
    label: 'Qwen3-14B (96GB+ RAM Recommended)'
  },
  {
    value: 'Qwen3-32B',
    label: 'Qwen3-32B (192GB+ RAM Recommended)'
  }
];

// Title and explanation section
const pageTitle = 'Training Process';
const pageDescription =
  'Transform your memories into a personalized AI model that thinks and communicates like you.';

export default function TrainingPage(): JSX.Element {
  const checkTrainStatus = useTrainingStore((state) => state.checkTrainStatus);
  const resetTrainingState = useTrainingStore((state) => state.resetTrainingState);
  const trainingError = useTrainingStore((state) => state.error);
  const setStatus = useTrainingStore((state) => state.setStatus);
  const fetchModelConfig = useModelConfigStore((state) => state.fetchModelConfig);
  const modelConfig = useModelConfigStore((store) => store.modelConfig);
  const thinkingModelConfig = useModelConfigStore((store) => store.thinkingModelConfig);
  const status = useTrainingStore((state) => state.status);
  const trainingProgress = useTrainingStore((state) => state.trainingProgress);
  const serviceStarted = useTrainingStore((state) => state.serviceStarted);

  const router = useRouter();

  const [selectedInfo, setSelectedInfo] = useState<boolean>(false);
  const isTraining = useTrainingStore((state) => state.isTraining);
  const setIsTraining = useTrainingStore((state) => state.setIsTraining);
  const [localTrainingParams, setLocalTrainingParams] = useState<LocalTrainingParams>(
    {} as LocalTrainingParams
  );
  const [cloudTrainingParams, setCloudTrainingParams] = useState<CloudTrainingParams>(
    {} as CloudTrainingParams
  );
  const [trainActionLoading, setTrainActionLoading] = useState(false);
  const [showCelebration, setShowCelebration] = useState(false);
  const [showMemoryModal, setShowMemoryModal] = useState(false);

  const [trainingType, _setTrainingType] = useState<'local' | 'cloud'>('local');
  const setTrainingType = useCallback((type: 'local' | 'cloud') => {
    console.log('Setting training type to:', type);
    _setTrainingType(type);
    // 每次设置训练类型时都保存到localStorage
    localStorage.setItem('trainingType', type);
  }, [_setTrainingType]);
  const [isHydrated, setIsHydrated] = useState(false);
  const [cloudProgress, setCloudProgress] = useState<CloudProgressData | null>(null);
  const [cloudJobId, setCloudJobId] = useState<string | null>(null);
  const [cloudTrainSuspended, setCloudTrainSuspended] = useState(false);
  const [cloudTrainingStatus, setCloudTrainingStatus] = useState<
    'idle' | 'training' | 'trained' | 'failed' | 'suspended' | 'pending'
  >('idle');
  // Track pause request state to avoid state inconsistency
  const [isPauseRequested, setIsPauseRequested] = useState(false);
  // Track pause status polling
  const [pauseStatus, setPauseStatus] = useState<'success' | 'pending' | 'failed' | null>(null);
  const pausePollingRef = useRef<NodeJS.Timeout | null>(null);
  // Track polling retry attempts
  const [pollingRetryCount, setPollingRetryCount] = useState(0);
  const maxPollingRetries = 3;
  // Pause timeout detection
  const pauseTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pauseTimeoutDuration = 30000; // 30 seconds timeout for pause operations

  const cleanupEventSourceRef = useRef<(() => void) | undefined>();
  const containerRef = useRef<HTMLDivElement>(null);
  const firstLoadRef = useRef<boolean>(true);
  const pollingStopRef = useRef<boolean>(false);
  const cloudPollingRef = useRef<NodeJS.Timeout | null>(null);

  const [cudaAvailable, setCudaAvailable] = useState<boolean>(false);
  const trainSuspended = useTrainingStore((state) => state.trainSuspended);
  const setTrainSuspended = useTrainingStore((state) => state.setTrainSuspended);

  // 检查 Think Model 配置是否完整
  const thinkingConfigComplete = useMemo(() => {
    return (
      !!thinkingModelConfig?.thinking_model_name &&
      !!thinkingModelConfig?.thinking_endpoint
      // 不再检查 thinking_api_key
    );
  }, [thinkingModelConfig]);

  const startCloudTrainingPolling = (isResume = false) => {
    console.log('startCloudTrainingPolling called with isResume:', isResume);
    setStatus('training');
    setIsTraining(true);
    setCloudTrainSuspended(false);
    setCloudTrainingStatus('training');
    pollingStopRef.current = false; // Reset polling stop flag
    
    // Only reset cloud progress if not resuming
    if (!isResume) {
      console.log('Resetting cloud progress because isResume is false');
      setCloudProgress(null);
    } else {
      console.log('Preserving cloud progress because isResume is true');
    }
    
    pollCloudProgress();
  };

  // New function to poll cloud training progress
  const pollCloudProgress = async () => {
    if (pollingStopRef.current) return;

    try {
      const res = await getCloudTrainingProgress();
      if (pollingStopRef.current) return; // Check again in case it was stopped during API call

      if (res.data.code === 0) {
        const progressData = res.data.data.progress;
        const currentJobId = res.data.data.job_id;

        // If we have progress data, set it regardless of status
        if (progressData && typeof progressData === 'object') {
          setCloudProgress(progressData);
          
          // 检查是否有活跃的云端训练
          const isActiveCloudTraining = ['in_progress', 'suspended', 'completed', 'failed', 'pending'].includes(progressData.status);
          
          if (currentJobId) {
            setCloudJobId(currentJobId);
          }

          // Set status based on progress data status
          if (progressData.status === 'in_progress') {
            // During training, just set the status without checking pause status
            setIsTraining(true);
            setStatus('training');
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('training');
          } else if (progressData.status === 'completed') {
            setStatus('trained');
            setIsTraining(false);
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('trained');
          } else if (progressData.status === 'failed') {
            setStatus('training'); // Keep as 'training' since ModelStatus doesn't have 'failed'
            setIsTraining(false);
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('failed');
          } else if (progressData.status === 'suspended') {
            setStatus('training');
            setIsTraining(false);
            setCloudTrainSuspended(true);
            setCloudTrainingStatus('suspended');
          } else if (progressData.status === 'pending') {
            // 检查是否真的是暂停中状态，还是仅仅是没有训练
            if (progressData.job_id || progressData.current_stage) {
              // 有job_id或current_stage，说明是真正的暂停中状态
              setStatus('training');
              setIsTraining(false); // 与本地训练一致，设置为false
              setCloudTrainSuspended(false);
              setCloudTrainingStatus('pending'); // 修改为'pending'
              setIsPauseRequested(true); // 设置暂停请求标志
              setPauseStatus('pending'); // 设置暂停状态为pending
              setTrainActionLoading(true); // 与本地训练一致，设置为true
            } else {
              // 没有job_id和current_stage，说明没有训练在进行
              setStatus('seed_identity');
              setIsTraining(false);
              setCloudTrainSuspended(false);
              setCloudTrainingStatus('');
              setIsPauseRequested(false);
              setPauseStatus(null);
              setTrainActionLoading(false);
            }
          } else {
            // For any other status, still show progress
            setStatus('training');
            setIsTraining(false);
          }

          // Continue polling if training is still in progress
          if (progressData.status === 'in_progress' && !pollingStopRef.current) {
            setPollingRetryCount(0); // Reset retry count on success
            cloudPollingRef.current = setTimeout(pollCloudProgress, POLLING_INTERVAL);
          }
          
          return true;
        }
      }
      
      // 如果没有有效的进度数据但API调用成功，继续轮询
      // 这确保了即使后端尚未准备好进度数据，前端也会继续检查
      if (!pollingStopRef.current) {
        console.log('No valid progress data yet, continuing polling');
        cloudPollingRef.current = setTimeout(pollCloudProgress, POLLING_INTERVAL);
      }
    } catch (error) {
      console.error('Error polling cloud training progress:', error);

      // Handle network errors with intelligent retry
      if (pollingRetryCount < maxPollingRetries) {
        setPollingRetryCount((prev) => prev + 1);
        message.warning(
          `Network error checking cloud training status (${pollingRetryCount + 1}/${maxPollingRetries}). Retrying...`
        );

        if (!pollingStopRef.current) {
          // Exponential backoff for retries
          const retryDelay = POLLING_INTERVAL * Math.pow(2, pollingRetryCount);
          cloudPollingRef.current = setTimeout(pollCloudProgress, retryDelay);
        }
      } else {
        message.error(
          'Unable to check cloud training status after multiple retries. Please refresh the page.'
        );
        setPollingRetryCount(0);
        stopCloudPolling();
      }
    }
  };

  const stopCloudPolling = () => {
    pollingStopRef.current = true;

    if (cloudPollingRef.current) {
      clearTimeout(cloudPollingRef.current);
      cloudPollingRef.current = null;
    }
  };

  // Poll pause status until it's no longer pending
  const pollPauseStatus = () => {
    if (pausePollingRef.current) {
      clearTimeout(pausePollingRef.current);
    }

    pausePollingRef.current = setTimeout(async () => {
      try {
        const result = await stopCloudTraining();
        const status = result.data.data.status;
        setPauseStatus(status);

        if (status === 'pending') {
          // Continue polling if still pending
          pollPauseStatus();
        } else if (status === 'success') {
          // Successfully stopped
          setIsTraining(false);
          setCloudTrainSuspended(true);
          setCloudTrainingStatus('suspended');
          setIsPauseRequested(false);
          setPauseStatus(null);
          stopCloudPolling();
          setTrainActionLoading(false); // 与本地训练一致，设置为false
          message.success('Cloud training stopped successfully');

          // Clear pause timeout if it exists
          if (pauseTimeoutRef.current) {
            clearTimeout(pauseTimeoutRef.current);
            pauseTimeoutRef.current = null;
          }
        } else if (status === 'failed') {
          // Failed to stop
          setIsPauseRequested(false);
          setPauseStatus(null);
          setTrainActionLoading(false);
          message.error('Failed to stop cloud training');

          // Clear pause timeout if it exists
          if (pauseTimeoutRef.current) {
            clearTimeout(pauseTimeoutRef.current);
            pauseTimeoutRef.current = null;
          }
        }
      } catch (error) {
        console.error('Error polling pause status:', error);
        setIsPauseRequested(false);
        setPauseStatus(null);
        setTrainActionLoading(false);
        message.error('Error checking pause status');

        // Clear pause timeout if it exists
        if (pauseTimeoutRef.current) {
          clearTimeout(pauseTimeoutRef.current);
          pauseTimeoutRef.current = null;
        }
      }
    }, 5000); // Poll every 5 seconds
  };

  // Check cloud training pause status on page load
  const checkCloudPauseStatus = async (): Promise<'pending' | 'success' | 'failed' | null> => {
    try {
      const res = await checkCloudTrainingPauseStatus();

      if (res.data.code === 0 && res.data.data) {
        const status = res.data.data.status;

        if (status === 'pending') {
          // If pause is still pending, show the pending state and start polling
          setIsPauseRequested(true);
          setPauseStatus('pending');
          pollPauseStatus();
          console.log('Found pending pause operation, starting status polling...');
          return 'pending';
        } else if (status === 'success') {
          // Pause was completed successfully - update UI to show suspended state
          setIsTraining(false);
          setCloudTrainSuspended(true);
          setCloudTrainingStatus('suspended');
          setIsPauseRequested(false);
          setPauseStatus(null);
          stopCloudPolling();
          console.log(
            'Previous pause operation completed successfully, UI updated to suspended state'
          );
          return 'success';
        } else if (status === 'failed') {
          // Pause failed, clear any pending state
          setIsPauseRequested(false);
          setPauseStatus(null);
          console.log('Previous pause operation failed');
          return 'failed';
        }
      }
      return null;
    } catch (error) {
      // Silently handle errors - this is just a status check
      console.log('No pending pause operation found or error checking pause status:', error);
      return null;
    }
  };

  useEffect(() => {
    fetchModelConfig();
  }, []);

  useEffect(() => {
    // Check CUDA availability once on load
    checkCudaAvailability()
      .then((res) => {
        if (res.data.code === 0) {
          const { cuda_available, cuda_info } = res.data.data;

          setCudaAvailable(cuda_available);

          if (cuda_available) {
            console.log('CUDA is available:', cuda_info);
          } else {
            console.log('CUDA is not available on this system');
          }
        } else {
          message.error(res.data.message || 'Failed to check CUDA availability');
        }
      })
      .catch((err) => {
        console.error('CUDA availability check failed', err);
        message.error('CUDA availability check failed');
      });
  }, []);

  // Handle hydration for client-side rendering
  useEffect(() => {
    // Set hydrated state to true once component is mounted on client
    setIsHydrated(true);

    // 从localStorage读取训练类型，这是决定性的
    const savedTrainingType = localStorage.getItem('trainingType');
    if (savedTrainingType === 'local' || savedTrainingType === 'cloud') {
      setTrainingType(savedTrainingType);
      console.log('Restored training type from localStorage:', savedTrainingType);
    } else {
      // 如果没有保存的训练类型，默认为本地训练
      setTrainingType('local');
      console.log('No saved training type, defaulting to local');
    }
  }, []);

  // 检查训练状态
  useEffect(() => {
    // Skip if not hydrated yet
    if (!isHydrated) return;

    // Check if user has at least 3 memories
    const checkMemoryCount = async () => {
      try {
        const memoryResponse = await getMemoryList();

        if (memoryResponse.data.code === 0) {
          const memories = memoryResponse.data.data;

          if (memories.length < 3) {
            // Show modal instead of direct redirect
            setShowMemoryModal(true);
            return;
          }
        }
      } catch (error) {
        console.error('Error checking memory count:', error);
      }
      
      // 检查本地和云端训练状态，但不改变训练类型
      await Promise.allSettled([
        checkLocalTrainingStatus(), 
        checkCloudTrainingProgress()
      ]);
      
      console.log('Training status check complete');
    };

    checkMemoryCount();
  }, [isHydrated]);

  useEffect(() => {
    const savedLogs = localStorage.getItem('trainingLogs');

    setTrainingDetails(savedLogs ? JSON.parse(savedLogs) : []);
  }, []);

  // Scroll to the bottom of the page
  const scrollPageToBottom = () => {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: 'smooth'
    });
    // Set that it's no longer the first load
    firstLoadRef.current = false;
  };

  const updateLocalTrainingParams = (params: Partial<LocalTrainingParams>) => {
    setLocalTrainingParams((state: LocalTrainingParams) => ({ ...state, ...params }));
  };

  const updateCloudTrainingParams = (params: Partial<CloudTrainingParams>) => {
    setCloudTrainingParams((state: CloudTrainingParams) => ({ ...state, ...params }));
  };

  const getDetails = () => {
    // Use EventSource to get logs
    const eventSource = new EventSource(`/api/trainprocess/logs`);

    eventSource.onmessage = (event) => {
      // Don't try to parse as JSON, just use the raw text data directly
      const logMessage = event.data;

      setTrainingDetails((prev) => {
        const newLogs = [
          ...prev.slice(-500), // Keep more log entries (500 instead of 100)
          {
            message: logMessage,
            timestamp: new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit'
            })
          }
        ];

        // Save logs to localStorage for persistence between page refreshes
        localStorage.setItem('trainingLogs', JSON.stringify(newLogs));

        return newLogs;
      });
    };

    eventSource.onerror = (error) => {
      console.error('EventSource failed:', error);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  };

  const updateTrainLog = () => {
    if (cleanupEventSourceRef.current) {
      cleanupEventSourceRef.current();
    }

    cleanupEventSourceRef.current = getDetails();
  };

  // Handler function for stopping training
  const handleStopTraining = async () => {
    // Prevent multiple pause requests
    if (isPauseRequested) {
      message.warning('Pause request already in progress, please wait...');
      return false;
    }

    setIsPauseRequested(true);
    try {
      setTrainActionLoading(true);
      const res = await stopTrain();

      if (res.data.code === 0) {
        // Start polling for stop status
        pollLocalPauseStatus();

        // Set timeout for pause operation
        pauseTimeoutRef.current = setTimeout(() => {
          if (isPauseRequested) {
            setIsPauseRequested(false);
            setPauseStatus(null);
            setTrainActionLoading(false);
            message.error('Pause operation timed out. Please try again.');
          }
        }, pauseTimeoutDuration);

        return true;
      } else {
        message.error(res.data.message || 'Failed to stop training');
        setIsPauseRequested(false);
        setTrainActionLoading(false);
        return false;
      }
    } catch (error) {
      console.error('Error stopping training:', error);
      message.error('Failed to stop training');
      setIsPauseRequested(false);
      setTrainActionLoading(false);
      return false;
    }
  };

  // Poll local training pause status until it's no longer pending
  const pollLocalPauseStatus = () => {
    if (pausePollingRef.current) {
      clearTimeout(pausePollingRef.current);
    }

    pausePollingRef.current = setTimeout(async () => {
      try {
        const res = await checkStopStatus();

        if (res.data.code === 0 && res.data.data) {
          const status = res.data.data.status;
          setPauseStatus(status);

          if (status === 'pending') {
            // Continue polling if still pending
            pollLocalPauseStatus();
          } else if (status === 'success') {
            // Successfully stopped
            setIsTraining(false);
            setTrainSuspended(true);
            setIsPauseRequested(false);
            setPauseStatus(null);
            stopPolling();
            setTrainActionLoading(false);
            message.success('Training paused successfully');

            // Clear pause timeout if it exists
            if (pauseTimeoutRef.current) {
              clearTimeout(pauseTimeoutRef.current);
              pauseTimeoutRef.current = null;
            }
          } else {
            // 'failed'
            message.error('Failed to pause training');
            setIsPauseRequested(false);
            setPauseStatus(null);
            setTrainActionLoading(false);

            // Clear pause timeout if it exists
            if (pauseTimeoutRef.current) {
              clearTimeout(pauseTimeoutRef.current);
              pauseTimeoutRef.current = null;
            }
          }
        } else {
          // API error
          setIsPauseRequested(false);
          setPauseStatus(null);
          setTrainActionLoading(false);
          message.error(res.data.message || 'Failed to check pause status');

          // Clear pause timeout if it exists
          if (pauseTimeoutRef.current) {
            clearTimeout(pauseTimeoutRef.current);
            pauseTimeoutRef.current = null;
          }
        }
      } catch (error) {
        console.error('Error polling pause status:', error);
        setIsPauseRequested(false);
        setPauseStatus(null);
        setTrainActionLoading(false);
        message.error('Error checking pause status');

        // Clear pause timeout if it exists
        if (pauseTimeoutRef.current) {
          clearTimeout(pauseTimeoutRef.current);
          pauseTimeoutRef.current = null;
        }
      }
    }, 3000); // Poll every 3 seconds
  };

  // Helper function to check if cloud training is in the critical stage
  const isInCriticalStage = (progressData: CloudProgressData | null): boolean => {
    if (!progressData) return false;

    // Check if we're in the "Training to create Second Me" stage
    const currentStage = progressData.current_stage;

    // Check by current_stage field
    if (currentStage === 'training_to_create_second_me') {
      return true;
    }

    // Also check if any stage has the "Training to create Second Me" name and is in progress
    if (progressData.stages) {
      return progressData.stages.some(
        (stage) => stage.name === 'Training to create Second Me' && stage.status === 'in_progress'
      );
    }

    return false;
  };

  // Handle cloud training stop with confirmation for critical stage
  const handleStopCloudTraining = async (): Promise<boolean> => {
    // Prevent multiple pause requests
    if (isPauseRequested) {
      message.warning('Pause request already in progress, please wait...');
      return false;
    }

    // Check if we're in the critical final stage
    if (isInCriticalStage(cloudProgress)) {
      return new Promise((resolve) => {
        Modal.confirm({
          title: 'Cloud Training Critical Stage Warning',
          content: (
            <div className="space-y-3">
              <p className="text-amber-600 font-medium">
                ⚠️ You are currently in the &quot;Training to create Second Me&quot; stage of cloud
                training.
              </p>
              <p>
                This critical stage does not support checkpoint resume functionality. If you stop
                cloud training now:
              </p>
              <ul className="list-disc pl-6 space-y-1">
                <li>All cloud training progress will be lost</li>
                <li>Alibaba Cloud training costs are non-refundable</li>
                <li>You will need to restart the entire cloud training process</li>
              </ul>
              <p className="font-medium">Are you sure you want to stop cloud training?</p>
            </div>
          ),
          okText: 'Yes, Stop Training',
          cancelText: 'Cancel',
          okType: 'danger',
          onOk: async () => {
            setIsPauseRequested(true);
            try {
              setTrainActionLoading(true); // 与本地训练一致，设置为true
              const res = await stopCloudTraining();

              if (res.data.code === 0 && res.data.data) {
                const status = res.data.data.status;
                setPauseStatus(status);

                if (status === 'pending') {
                  // Start polling for status updates
                  pollPauseStatus();
                  resolve(true);
                } else if (status === 'success') {
                  // Immediately update state when pause is successful
                  setIsTraining(false);
                  setCloudTrainSuspended(true);
                  setCloudTrainingStatus('suspended');
                  setIsPauseRequested(false);
                  setPauseStatus(null);
                  stopCloudPolling();
                  setTrainActionLoading(false); // 与本地训练一致，设置为false
                  message.success('Cloud training stopped successfully');
                  resolve(true);
                } else {
                  // 'failed'
                  message.error('Failed to stop cloud training');
                  setIsPauseRequested(false);
                  setPauseStatus(null);
                  resolve(false);
                }
              } else {
                message.error(res.data.message || 'Failed to stop cloud training');
                setIsPauseRequested(false);
                setPauseStatus(null);
                resolve(false);
              }
            } catch (error) {
              console.error('Error stopping cloud training:', error);
              message.error('Failed to stop cloud training');
              setIsPauseRequested(false);
              setPauseStatus(null);
              resolve(false);
            }
          },
          onCancel: () => resolve(false)
        });
      });
    }

    // Normal stopping for non-critical stages (supports checkpoint resume)
    setIsPauseRequested(true);
    try {
      setTrainActionLoading(true); // 与本地训练一致，设置为true
      const res = await stopCloudTraining();

      if (res.data.code === 0 && res.data.data) {
        const status = res.data.data.status;
        setPauseStatus(status);

        if (status === 'pending') {
          // Start polling for status updates
          pollPauseStatus();
          return true;
        } else if (status === 'success') {
          // Immediately update state when pause is successful
          setIsTraining(false);
          setCloudTrainSuspended(true);
          setCloudTrainingStatus('suspended');
          setIsPauseRequested(false);
          setPauseStatus(null);

          // Clear pause timeout if it exists
          if (pauseTimeoutRef.current) {
            clearTimeout(pauseTimeoutRef.current);
            pauseTimeoutRef.current = null;
          }

          // Stop polling immediately since pause is confirmed
          stopCloudPolling();
          setTrainActionLoading(false); // 与本地训练一致，设置为false
          message.success('Cloud training stopped successfully');

          return true;
        } else {
          // 'failed'
          message.error('Failed to stop cloud training');
          setIsPauseRequested(false);
          setPauseStatus(null);
          setTrainActionLoading(false);
          return false;
        }
      }

      message.error(res.data.message || 'Failed to stop cloud training');
      setIsPauseRequested(false);
      setPauseStatus(null);
      setTrainActionLoading(false);

      return false;
    } catch (error) {
      console.error('Error stopping cloud training:', error);
      message.error('Failed to stop cloud training');
      setIsPauseRequested(false);
      setPauseStatus(null);
      setTrainActionLoading(false);

      return false;
    }
  };

  // Handle cloud training reset
  const handleResetCloudProgress = async () => {
    setTrainActionLoading(true);

    try {
      const res = await resetCloudTrainingProgress();
      if (res.data.code === 0) {
        setCloudTrainSuspended(false);
        setCloudProgress(null);
        setCloudJobId(null);
        setCloudTrainingStatus('idle');
        setStatus('training');
        resetTrainingState();

        // Clear pause state
        setIsPauseRequested(false);
        setPauseStatus(null);
        if (pausePollingRef.current) {
          clearTimeout(pausePollingRef.current);
          pausePollingRef.current = null;
        }

        localStorage.removeItem('trainingLogs');
        localStorage.removeItem('hasShownCloudTrainingComplete');

        // Request progress after reset to get the initialized progress state
        try {
          const progressRes = await getCloudTrainingProgress();
          if (progressRes.data.code === 0) {
            const progressData = progressRes.data.data.progress;
            if (progressData) {
              console.log('Setting cloud progress from handleResetCloudProgress:', progressData);
              setCloudProgress(progressData);
              // Update job ID if available
              if (progressRes.data.data.job_id) {
                setCloudJobId(progressRes.data.data.job_id);
              }
              console.log('Retrieved initial cloud training progress after reset:', progressData);
            }
          } else {
            console.log(
              'No initial progress found after reset, which is expected for a fresh start'
            );
          }
        } catch (progressError) {
          console.error('Error getting initial progress after reset:', progressError);
          // This is not a critical error, just log it
        }
      } else {
        message.error(res.data.message || 'Failed to reset cloud training progress');
      }
    } catch (error) {
      console.error('Error resetting cloud training progress:', error);
      message.error('Failed to reset cloud training progress');
    } finally {
      setTrainActionLoading(false);
    }
  };

  // Handle resume cloud training
  const handleResumeCloudTraining = async () => {
    // 检查 Think Model 是否已配置
    if (!thinkingConfigComplete) {
      message.error('Please configure Thinking Model before resuming training');
      return;
    }
    
    setIsTraining(true);
    setCloudTrainSuspended(false);
    setCloudTrainingStatus('training');

    // Clear pause state
    setIsPauseRequested(false);
    setPauseStatus(null);
    if (pausePollingRef.current) {
      clearTimeout(pausePollingRef.current);
      pausePollingRef.current = null;
    }

    try {
      // 确保 is_cot 设置为 true
      const cloudParams = {
        ...cloudTrainingParams,
        is_cot: true // 强制设置为 true
      };
      
      // Call the resume API endpoint
      const response = await resumeCloudTraining();

      if (response.data.code === 0) {
        // Resume successful, restart polling
        setStatus('training');
        startCloudTrainingPolling(true); // Pass true to indicate this is a resume operation
        message.success('Cloud training resumed successfully');
        return true;
      } else {
        throw new Error(response.data.message || 'Failed to resume cloud training');
      }
    } catch (error) {
      console.error('Error resuming cloud training:', error);
      setIsTraining(false);
      setCloudTrainSuspended(true);
      message.error('Failed to resume cloud training');
      return false;
    }
  };

  // Resume local training
  const handleResumeLocalTraining = async () => {
    // 检查 Think Model 是否已配置
    if (!thinkingConfigComplete) {
      message.error('Please configure Thinking Model before resuming training');
      return;
    }
    
    setTrainActionLoading(true);
    try {
      // Use startTrain to resume from the current state
      const res = await startTrain({
        ...localTrainingParams,
        is_cot: true, // 确保 is_cot 设置为 true
        // Convert to legacy format for compatibility
        local_model_name: localTrainingParams.model_name,
        cloud_model_name: cloudTrainingParams.model_name
      });

      if (res.data.code === 0) {
        setTrainSuspended(false);
        setIsTraining(true);
        startGetTrainingProgress(true); // 传入true表示这是恢复训练
        message.success('Training resumed successfully');

        // Save training type preference to localStorage
        localStorage.setItem('trainingType', 'local');
      } else {
        message.error(res.data.message || 'Failed to resume training');
      }
    } catch (error: unknown) {
      console.error('Error resuming training:', error);
      message.error('Failed to resume training');
    } finally {
      setTrainActionLoading(false);
    }
  };

  const handleResetProgress = () => {
    setTrainActionLoading(true);

    resetProgress()
      .then((res) => {
        if (res.data.code === 0) {
          setTrainSuspended(false);
          resetTrainingState();
          localStorage.removeItem('trainingLogs');
        } else {
          throw new Error(res.data.message || 'Failed to reset progress');
        }
      })
      .catch((error) => {
        console.error('Error resetting progress:', error);
      })
      .finally(() => {
        setTrainActionLoading(false);
      });
  };

  // Start new training
  const handleStartNewTraining = async () => {
    // 检查 Think Model 是否已配置
    if (!thinkingConfigComplete) {
      message.error('Please configure Thinking Model before starting training');
      return;
    }
    
    setIsTraining(true);
    // Clear training logs
    setTrainingDetails([]);
    localStorage.removeItem('trainingLogs');
    // Reset training status to initial state
    resetTrainingState();

    try {
      console.log('Using startTrain API to train new model:', localTrainingParams.model_name);
      const res = await startTrain({
        ...localTrainingParams,
        is_cot: true, // 确保 is_cot 设置为 true
        // Convert to legacy format for compatibility
        local_model_name: localTrainingParams.model_name,
        cloud_model_name: cloudTrainingParams.model_name
      } as TrainingConfig);

      if (res.data.code === 0) {
        // Save training configuration and start polling
        localStorage.setItem('localTrainingParams', JSON.stringify(localTrainingParams));
        scrollPageToBottom();
        startGetTrainingProgress();
      } else {
        message.error(res.data.message || 'Failed to start training');
        setIsTraining(false);
      }
    } catch (error: unknown) {
      console.error('Error starting training:', error);
      setIsTraining(false);

      if (error instanceof Error) {
        message.error(error.message || 'Failed to start training');
      } else {
        message.error('Failed to start training');
      }
    }
  };

  // Start local training
  const handleStartLocalTraining = async () => {
    // 检查 Think Model 是否已配置
    if (!thinkingConfigComplete) {
      message.error('Please configure Thinking Model before starting training');
      return;
    }
    
    setTrainActionLoading(true);
    setIsTraining(true);
    // Clear training logs
    setTrainingDetails([]);
    localStorage.removeItem('trainingLogs');
    // Reset training status to initial state
    resetTrainingState();

    try {
      console.log('Using startTrain API to train new model:', localTrainingParams.model_name);
      const res = await startTrain({
        ...localTrainingParams,
        is_cot: true, // 确保 is_cot 设置为 true
        // Convert to legacy format for compatibility
        local_model_name: localTrainingParams.model_name,
        cloud_model_name: cloudTrainingParams.model_name
      } as TrainingConfig);

      if (res.data.code === 0) {
        // Save training configuration and start polling
        localStorage.setItem('localTrainingParams', JSON.stringify(localTrainingParams));
        scrollPageToBottom();
        startGetTrainingProgress();
      } else {
        message.error(res.data.message || 'Failed to start training');
        setIsTraining(false);
      }
    } catch (error: unknown) {
      console.error('Error starting training:', error);
      setIsTraining(false);

      if (error instanceof Error) {
        message.error(error.message || 'Failed to start training');
      } else {
        message.error('Failed to start training');
      }
    } finally {
      setTrainActionLoading(false);
    }
  };

  const handleRetrainModel = async () => {
    setTrainActionLoading(true); // 添加加载状态
    setIsTraining(true);
    // Clear training logs
    setTrainingDetails([]);
    localStorage.removeItem('trainingLogs');
    // Reset training status to initial state
    resetTrainingState();

    try {
      const res = await retrain({
        ...localTrainingParams,
        // Convert to legacy format for compatibility
        local_model_name: localTrainingParams.model_name,
        cloud_model_name: cloudTrainingParams.model_name
      } as TrainingConfig);

      if (res.data.code === 0) {
        // Save training configuration and start polling
        localStorage.setItem('localTrainingParams', JSON.stringify(localTrainingParams));
        scrollPageToBottom();
        startGetTrainingProgress();
      } else {
        message.error(res.data.message || 'Failed to retrain model');
        setIsTraining(false);
      }
    } catch (error: unknown) {
      console.error('Error retraining model:', error);
      setIsTraining(false);

      if (error instanceof Error) {
        message.error(error.message || 'Failed to retrain model');
      } else {
        message.error('Failed to retrain model');
      }
    } finally {
      setTrainActionLoading(false); // 无论成功或失败，都重置加载状态
    }
  };

  const handleRetrainCloudModel = async () => {
    setTrainActionLoading(true);
    try {
      // Reset cloud training progress first
      const resetRes = await resetCloudTrainingProgress();

      if (resetRes.data.code === 0) {
        // Then start a new training session
        await handleStartCloudTraining();
      } else {
        message.error(resetRes.data.message || 'Failed to reset cloud training progress');
        setTrainActionLoading(false);
      }
    } catch (error) {
      console.error('Error retraining cloud model:', error);
      message.error('Failed to retrain cloud model');
      setTrainActionLoading(false);
    }
  };

  const handleTrainingAction = async () => {
    // Save training type preference to localStorage
    localStorage.setItem('trainingType', trainingType);

    if (trainingType === 'cloud') {
      if (isTraining) {
        // Pause cloud training
        await handleStopCloudTraining();
      } else if (cloudTrainSuspended) {
        // Resume cloud training
        await handleResumeCloudTraining();
      } else if (cloudTrainingStatus === 'completed') {
        // Retrain cloud model
        await handleRetrainCloudModel();
      } else {
        // Start new cloud training
        await handleStartCloudTraining();
      }
    } else {
      if (isTraining) {
        // Pause local training
        await handleStopTraining();
      } else if (trainSuspended) {
        // Resume local training
        await handleResumeLocalTraining();
      } else if (status === 'trained') {
        // Retrain local model
        await handleRetrainModel();
      } else {
        // Start new local training
        await handleStartLocalTraining();
      }
    }
  };

  useEffect(() => {
    return () => {
      stopPolling();
      stopCloudPolling();
    };
  }, []);

  const pollProgress = () => {
    if (pollingStopRef.current) return;

    checkTrainStatus()
      .then((res) => {
        if (pollingStopRef.current) return;

        if (res && res.data && res.data.data && res.data.data.progress) {
          const progressData = res.data.data.progress;

          // Handle different training statuses
          if (progressData.status === 'completed') {
            // Training completed successfully
            setStatus('trained');
            setIsTraining(false);
            setTrainSuspended(false);

            // Show celebration effect
            const hasShownTrainingComplete = localStorage.getItem('hasShownTrainingComplete');

            if (hasShownTrainingComplete !== 'true') {
              setTimeout(() => {
                setShowCelebration(true);
                localStorage.setItem('hasShownTrainingComplete', 'true');
              }, 1000);
            }
            stopPolling();
          } else if (progressData.status === 'failed') {
            // Training failed
            setStatus('training');
            setIsTraining(false);
            setTrainSuspended(false);
            stopPolling();
            message.error('Training failed. Please check the logs for details.');
          } else if (progressData.status === 'suspended') {
            // Training suspended
            setStatus('training');
            setIsTraining(false);
            setTrainSuspended(true);
            setIsPauseRequested(false);
            setPauseStatus(null);
            stopPolling();
            message.info('Training has been paused.');

            // Clear pause timeout if it exists
            if (pauseTimeoutRef.current) {
              clearTimeout(pauseTimeoutRef.current);
              pauseTimeoutRef.current = null;
            }
          }
        }
      })
      .catch((error) => {
        console.error('Error polling training progress:', error);
      })
      .finally(() => {
        if (!pollingStopRef.current) {
          setTimeout(pollProgress, POLLING_INTERVAL);
        }
      });
  };

  const getTrainingButtonText = () => {
    if (trainingType === 'cloud') {
      if (isTraining) {
        return 'Pause Cloud Training';
      } else if (cloudTrainSuspended) {
        return 'Resume Cloud Training';
      } else if (cloudTrainingStatus === 'completed') {
        return 'Retrain Cloud Model';
      } else {
        return 'Start Cloud Training';
      }
    } else {
      if (isTraining) {
        return 'Pause Training';
      } else if (trainSuspended) {
        return 'Resume Training';
      } else if (status === 'trained') {
        return 'Retrain Model';
      } else {
        return 'Start Training';
      }
    }
  };

  const getButtonLoadingState = () => {
    if (trainingType === 'cloud') {
      return trainActionLoading || isPauseRequested;
    } else {
      return trainActionLoading || isPauseRequested;
    }
  };

  const getButtonDisabledState = () => {
    if (trainingType === 'cloud') {
      return trainActionLoading || isPauseRequested || pauseStatus === 'pending';
    } else {
      return trainActionLoading || isPauseRequested || pauseStatus === 'pending';
    }
  };

  const getButtonType = () => {
    if (trainingType === 'cloud') {
      if (isTraining) {
        return 'default';
      } else if (cloudTrainSuspended) {
        return 'primary';
      } else {
        return 'primary';
      }
    } else {
      if (isTraining) {
        return 'default';
      } else if (trainSuspended) {
        return 'primary';
      } else {
        return 'primary';
      }
    }
  };

  const renderTrainingProgress = () => {
    return (
      <div className="space-y-6">
        {/* Training Progress Component */}
        <TrainingProgress
          status={status}
          trainingProgress={trainingProgress}
          trainingType={trainingType}
          cloudProgressData={cloudProgress}
          cloudJobId={cloudJobId}
        />
      </div>
    );
  };

  const renderTrainingLog = () => {
    return (
      <div className="space-y-6">
        {/* Training Log Console */}
        <TrainingLog trainingDetails={trainingDetails} />
      </div>
    );
  };

  // Check local training status including suspended state
  const checkLocalTrainingStatus = async () => {
    try {
      const res = await checkTrainStatus();

      if (res && res.data && res.data.data && res.data.data.progress) {
        const progressData = res.data.data.progress;

        // If local training is suspended, update the UI accordingly
        if (progressData.status === 'suspended') {
          setTrainingType('local');
          setStatus('training');
          setIsTraining(false);
          setTrainSuspended(true);
          console.log('Local training detected as suspended');
        }
        // If local training is in progress
        else if (progressData.status === 'in_progress') {
          setTrainingType('local');
          setStatus('training');
          setIsTraining(true);
          setTrainSuspended(false);
          console.log('Local training detected as in_progress');
          startGetTrainingProgress();
        }
        // If local training is completed
        else if (progressData.status === 'completed') {
          setTrainingType('local');
          setStatus('trained');
          setIsTraining(false);
          setTrainSuspended(false);
          console.log('Local training detected as completed');
        }
        // If local training failed
        else if (progressData.status === 'failed') {
          setTrainingType('local');
          setStatus('training');
          setIsTraining(false);
          setTrainSuspended(false);
          console.log('Local training detected as failed');
        }

        // Save the training type to localStorage
        localStorage.setItem('trainingType', 'local');
      }
    } catch (error) {
      console.log('No active local training found or error checking status:', error);
    }
  };

  // Check cloud training progress on page load
  const checkCloudTrainingProgress = async () => {
    try {
      const res = await getCloudTrainingProgress();

      if (res.data.code === 0) {
        const progressData = res.data.data.progress;
        const currentJobId = res.data.data.job_id;

        // If we have progress data, set it regardless of status
        if (progressData && typeof progressData === 'object') {
          setCloudProgress(progressData);
          
          // 检查是否有活跃的云端训练
          const isActiveCloudTraining = ['in_progress', 'suspended', 'completed', 'failed', 'pending'].includes(progressData.status);
          
          if (currentJobId) {
            setCloudJobId(currentJobId);
          }

          // Set status based on progress data status
          if (progressData.status === 'in_progress') {
            // During training, just set the status without checking pause status
            setIsTraining(true);
            setStatus('training');
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('training');
            startCloudTrainingPolling(true); // Start polling to continuously check progress
          } else if (progressData.status === 'completed') {
            setStatus('trained');
            setIsTraining(false);
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('trained');
          } else if (progressData.status === 'failed') {
            setStatus('training'); // Keep as 'training' since ModelStatus doesn't have 'failed'
            setIsTraining(false);
            setCloudTrainSuspended(false);
            setCloudTrainingStatus('failed');
          } else if (progressData.status === 'suspended') {
            setStatus('training');
            setIsTraining(false);
            setCloudTrainSuspended(true);
            setCloudTrainingStatus('suspended');
          } else if (progressData.status === 'pending') {
            // 检查是否真的是暂停中状态，还是仅仅是没有训练
            if (progressData.job_id || progressData.current_stage) {
              // 有job_id或current_stage，说明是真正的暂停中状态
              setStatus('training');
              setIsTraining(false); // 与本地训练一致，设置为false
              setCloudTrainSuspended(false);
              setCloudTrainingStatus('pending'); // 修改为'pending'
              setIsPauseRequested(true); // 设置暂停请求标志
              setPauseStatus('pending'); // 设置暂停状态为pending
              setTrainActionLoading(true); // 与本地训练一致，设置为true
            } else {
              // 没有job_id和current_stage，说明没有训练在进行
              setStatus('seed_identity');
              setIsTraining(false);
              setCloudTrainSuspended(false);
              setCloudTrainingStatus('');
              setIsPauseRequested(false);
              setPauseStatus(null);
              setTrainActionLoading(false);
            }
          } else {
            // For any other status, still show progress
            setStatus('training');
            setIsTraining(false);
          }

          // Return true only if there's an active cloud training
          return isActiveCloudTraining;
        }
      }
      return false;
    } catch (error) {
      // Silently handle errors for initial check - cloud training might not be active
      console.log('No active cloud training found or error checking cloud progress:', error);
      return false;
    }
  };

  // Start polling training progress
  const startPolling = () => {
    if (pollingStopRef.current) {
      return;
    }

    // Start new polling
    checkTrainStatus()
      .then(() => {
        if (pollingStopRef.current) {
          return;
        }

        setTimeout(() => {
          startPolling();
        }, POLLING_INTERVAL);
      })
      .catch((error) => {
        console.error('Training status check failed:', error);
        stopPolling(); // Stop polling when error occurs
        setIsTraining(false);
        message.error('Training status check failed, monitoring stopped');
      });
  };

  // Start polling for training progress
  const startGetTrainingProgress = (isResume = false) => {
    // Reset polling stop flag
    pollingStopRef.current = false;

    // Set training status
    setStatus('training');
    setIsTraining(true);
    setTrainSuspended(false);

    // Reset pause-related states
    setIsPauseRequested(false);
    setPauseStatus(null);
    if (pausePollingRef.current) {
      clearTimeout(pausePollingRef.current);
      pausePollingRef.current = null;
    }
    if (pauseTimeoutRef.current) {
      clearTimeout(pauseTimeoutRef.current);
      pauseTimeoutRef.current = null;
    }
    
    // Only reset training state if not resuming
    if (!isResume) {
      resetTrainingState();
    }

    // Start polling
    startPolling();
  };

  // Stop polling
  const stopPolling = () => {
    pollingStopRef.current = true;
  };

  useEffect(() => {
    if (status === 'trained' || trainingError) {
      stopPolling();
      setIsTraining(false);

      const hasShownTrainingComplete = localStorage.getItem('hasShownTrainingComplete');

      if (hasShownTrainingComplete !== 'true' && status === 'trained' && !trainingError) {
        setTimeout(() => {
          setShowCelebration(true);
          localStorage.setItem('hasShownTrainingComplete', 'true');
        }, 1000);
      }
    }
  }, [status, trainingError]);

  // Handle memory modal confirmation
  const handleMemoryModalConfirm = () => {
    setShowMemoryModal(false);
    router.push(ROUTER_PATH.TRAIN_MEMORIES);
  };

  const handleStartCloudTraining = async () => {
    // 检查 Think Model 是否已配置
    if (!thinkingConfigComplete) {
      message.error('Please configure Thinking Model before starting training');
      return;
    }
    
    setTrainActionLoading(true);
    setIsTraining(true);
    setStatus('training');
    setCloudTrainingStatus('training');

    setTrainingDetails([]);
    localStorage.removeItem('trainingLogs');
    localStorage.removeItem('hasShownCloudTrainingComplete'); // Reset celebration flag

    resetTrainingState();
    setCloudProgress(null); // Reset cloud progress
    setCloudJobId(null); // Reset cloud job ID
    setCloudTrainSuspended(false); // Reset suspension state

    // Clear pause state
    setIsPauseRequested(false);
    setPauseStatus(null);
    if (pausePollingRef.current) {
      clearTimeout(pausePollingRef.current);
      pausePollingRef.current = null;
    }

    try {
      const res = await startCloudTraining({
        base_model: cloudTrainingParams.base_model,
        training_type: cloudTrainingParams.training_type || 'efficient_sft',
        data_synthesis_mode: cloudTrainingParams.data_synthesis_mode || 'medium',
        language: cloudTrainingParams.language || 'english',
        is_cot: true, // 确保 is_cot 设置为 true
        data_filtering_model: cloudTrainingParams.data_filtering_model || 'gemma:2b',
        data_filtering_workers: cloudTrainingParams.data_filtering_workers || 5,
        data_filtering_keep_ratio: cloudTrainingParams.data_filtering_keep_ratio || 0.8,
        hyper_parameters: {
          n_epochs: cloudTrainingParams.hyper_parameters?.n_epochs,
          learning_rate: cloudTrainingParams.hyper_parameters?.learning_rate
        }
      });

      if (res.data.code === 0) {
        const data = res.data.data as { job_id?: string };
        const returnedJobId = data.job_id;
        if (returnedJobId) {
          setCloudJobId(returnedJobId);
        }

        localStorage.setItem('cloudTrainingParams', JSON.stringify(cloudTrainingParams));
        localStorage.setItem('trainingType', 'cloud'); // Save training type preference
        scrollPageToBottom();
        setTrainingType('cloud');
        startCloudTrainingPolling();
      } else {
        message.error(res.data.message || 'Failed to start cloud training');
        setIsTraining(false);
        setCloudTrainingStatus('idle');
      }
    } catch (error) {
      console.error('Error starting cloud training:', error);
      setIsTraining(false);

      if (error instanceof Error) {
        message.error(error.message || 'Failed to start cloud training');
      } else {
        message.error('Failed to start cloud training');
      }
    } finally {
      setTrainActionLoading(false);
    }
  };

  // Switch between local and cloud training
  const handleSwitchTrainingType = (type: 'local' | 'cloud') => {
    setTrainingType(type);
    // Save training type preference to localStorage
    localStorage.setItem('trainingType', type);

    // Reset training states when switching
    setIsTraining(false);
    setTrainSuspended(false);
    setCloudTrainSuspended(false);
    setIsPauseRequested(false);
    setPauseStatus(null);

    // Clear any polling or timeouts
    stopPolling();
    if (pausePollingRef.current) {
      clearTimeout(pausePollingRef.current);
      pausePollingRef.current = null;
    }
    if (pauseTimeoutRef.current) {
      clearTimeout(pauseTimeoutRef.current);
      pauseTimeoutRef.current = null;
    }

    // Check if there's an active training for the selected type
    if (type === 'local') {
      checkLocalTrainingStatus();
    } else {
      checkCloudTrainingProgress();
    }
  };

  useEffect(() => {
    // Monitor training status changes and manage log connections
    if (trainingProgress.status === 'in_progress') {
      setIsTraining(true);

      if (firstLoadRef.current) {
        scrollPageToBottom();

        // On first load, start polling and get training progress.
        startGetTrainingProgress();
      }
    }
    // If training is completed or failed, stop polling
    else if (
      trainingProgress.status === 'completed' ||
      trainingProgress.status === 'failed' ||
      trainingProgress.status === 'suspended'
    ) {
      stopPolling();
      setIsTraining(false);
    }
  }, [trainingProgress]);

  useEffect(() => {
    if (isTraining) {
      updateTrainLog();
    }
  }, [isTraining]);

  // Cleanup when component unmounts
  useEffect(() => {
    return () => {
      stopPolling();
      if (cloudPollingRef.current) {
        // Ensure cloud polling is stopped
        clearTimeout(cloudPollingRef.current);
      }
      if (pauseTimeoutRef.current) {
        // Clear pause timeout
        clearTimeout(pauseTimeoutRef.current);
      }
      if (pausePollingRef.current) {
        // Clear pause polling
        clearTimeout(pausePollingRef.current);
      }
      pollingStopRef.current = true; // Set flag to stop any ongoing polling
    };
  }, []);

  const [trainingDetails, setTrainingDetails] = useState<TrainingDetail[]>([]);

  //get training params
  useEffect(() => {
    getTrainingParams()
      .then((res) => {
        if (res.data.code === 0) {
          const data = res.data.data;

          // Set separate local and cloud training params
          setLocalTrainingParams(data.local);
          setCloudTrainingParams(data.cloud);

          localStorage.setItem('localTrainingParams', JSON.stringify(data.local));
          localStorage.setItem('cloudTrainingParams', JSON.stringify(data.cloud));
          localStorage.setItem('trainingParams', JSON.stringify(data));
        } else {
          throw new Error(res.data.message);
        }
      })
      .catch((error) => {
        console.error(error.message);
      });
  }, []);

  return (
    <div ref={containerRef} className="h-full overflow-auto">
      {/* Memory count warning modal */}
      <Modal
        cancelText="Stay Here"
        okText="Go to Memories Page"
        onCancel={() => setShowMemoryModal(false)}
        onOk={handleMemoryModalConfirm}
        open={showMemoryModal}
        title="More Memories Needed"
      >
        <p>You need to add at least 3 memories before you can train your model.</p>
        <p>Would you like to go to the memories page to add more?</p>
      </Modal>

      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* Page Title and Description */}
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">{pageTitle}</h1>
          <p className="text-gray-600 max-w-3xl">{pageDescription}</p>
        </div>
        {/* Training Configuration Component */}
        <TrainingConfiguration
          baseModelOptions={baseModelOptions}
          cudaAvailable={cudaAvailable}
          handleResetProgress={
            trainingType === 'cloud' ? handleResetCloudProgress : handleResetProgress
          }
          handleTrainingAction={handleTrainingAction}
          isTraining={isTraining}
          modelConfig={modelConfig}
          thinkingModelConfig={thinkingModelConfig}
          thinkingConfigComplete={thinkingConfigComplete}
          setSelectedInfo={setSelectedInfo}
          status={status}
          trainActionLoading={trainActionLoading}
          trainSuspended={trainingType === 'cloud' ? cloudTrainSuspended : trainSuspended}
          localTrainingParams={localTrainingParams}
          cloudTrainingParams={cloudTrainingParams}
          updateLocalTrainingParams={updateLocalTrainingParams}
          updateCloudTrainingParams={updateCloudTrainingParams}
          trainingType={trainingType}
          setTrainingType={handleSwitchTrainingType}
          cloudTrainingStatus={cloudTrainingStatus}
          isPauseRequested={isPauseRequested}
          pauseStatus={pauseStatus}
        />

        {/* Only show training progress after training starts */}
        {(status === 'training' || status === 'trained') && renderTrainingProgress()}

        {/* Always show training log regardless of training status */}
        {renderTrainingLog()}

        {/* stage2 and L2 Panels - show when training is complete or model is running */}

        <InfoModal
          content={
            <div className="space-y-4">
              <p className="text-gray-600">{trainInfo.description}</p>
              <div>
                <h4 className="font-medium mb-2">Key Features:</h4>
                <ul className="list-disc pl-5 space-y-1">
                  {trainInfo.features.map((feature, index) => (
                    <li key={index} className="text-gray-600">
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          }
          onClose={() => setSelectedInfo(false)}
          open={!!selectedInfo && !!trainInfo}
          title={trainInfo.name}
        />

        {/* Training completion celebration effect */}
        <CelebrationEffect isVisible={showCelebration} onClose={() => setShowCelebration(false)} />
      </div>
    </div>
  );
}
