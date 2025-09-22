import type { IThinkingModelParams } from '@/service/modelConfig';
import { updateThinkingConfig } from '@/service/modelConfig';
import { useModelConfigStore } from '@/store/useModelConfigStore';
import { Input, message, Modal } from 'antd';
import { useEffect, useState } from 'react';

interface IProps {
  open: boolean;
  onClose: () => void;
}

const ThinkingModelModal = (props: IProps) => {
  const { open, onClose: handleCancel } = props;

  const fetchModelConfig = useModelConfigStore((store) => store.fetchModelConfig);
  const [thinkingModelParams, setThinkingModelParams] = useState<IThinkingModelParams>(
    {} as IThinkingModelParams
  );
  const updateThinkingModelConfig = useModelConfigStore((store) => store.updateThinkingModelConfig);
  const thinkingModelConfig = useModelConfigStore((store) => store.thinkingModelConfig);

  useEffect(() => {
    if (open) {
      fetchModelConfig();
    }
  }, [open]);

  useEffect(() => {
    // If backend returned data
    if (thinkingModelConfig) {
      const updatedParams = {...thinkingModelConfig};
      
      // Check fields, if null/undefined then set default values
      if (updatedParams.thinking_model_name === null || updatedParams.thinking_model_name === undefined) {
        updatedParams.thinking_model_name = "deepseek-reasoner";
      }
      
      if (updatedParams.thinking_endpoint === null || updatedParams.thinking_endpoint === undefined) {
        updatedParams.thinking_endpoint = "https://api.deepseek.com";
      }
      
      setThinkingModelParams(updatedParams);
    } else {
      // Set complete default values when no configuration exists
      setThinkingModelParams({
        thinking_model_name: "deepseek-reasoner",
        thinking_endpoint: "https://api.deepseek.com",
        thinking_api_key: ""
      });
    }
  }, [thinkingModelConfig]);

  const handleUpdate = () => {
    const thinkingConfigComplete =
      !!thinkingModelParams.thinking_model_name &&
      !!thinkingModelParams.thinking_api_key &&
      !!thinkingModelParams.thinking_endpoint;

    if (!thinkingConfigComplete) {
      message.error('Please fill in all thinking model configuration fields');

      return;
    }

    updateThinkingConfig(thinkingModelParams)
      .then((res) => {
        if (res.data.code == 0) {
          updateThinkingModelConfig(thinkingModelParams);
          handleCancel();
        } else {
          throw new Error(res.data.message);
        }
      })
      .catch((error) => {
        console.error(error.message || 'Failed to update model config');
      });
  };

  return (
    <Modal
      centered
      onCancel={handleCancel}
      onOk={() => {
        handleUpdate();
      }}
      open={open}
    >
      <div className="flex flex-col gap-2 mb-4">
        <div className="text-xl leading-6 font-semibold text-gray-900">Thinking model</div>
        <div className="text-sm font-medium text-gray-700">Currently only supports DeepSeek</div>
      </div>
      <div className="p-4 border rounded-lg hover:shadow-md transition-shadow">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model Name</label>
            <Input
              autoComplete="off"
              className="w-full"
              disabled
              onChange={(e) =>
                setThinkingModelParams({
                  ...thinkingModelParams,
                  thinking_model_name: e.target.value
                })
              }
              value={thinkingModelParams.thinking_model_name}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
            {/* form is to disable autoComplete */}
            <form autoComplete="off">
              <Input.Password
                autoComplete="off"
                className="w-full"
                onChange={(e) =>
                  setThinkingModelParams({
                    ...thinkingModelParams,
                    thinking_api_key: e.target.value
                  })
                }
                value={thinkingModelParams.thinking_api_key}
              />
            </form>
          </div>
        </div>

        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">API Endpoint</label>
          <Input
            autoComplete="off"
            className="w-full"
            disabled
            onChange={(e) =>
              setThinkingModelParams({
                ...thinkingModelParams,
                thinking_endpoint: e.target.value
              })
            }
            value={thinkingModelParams.thinking_endpoint}
          />
        </div>
      </div>
    </Modal>
  );
};

export default ThinkingModelModal;
