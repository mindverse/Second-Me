import { ExclamationCircleOutlined } from '@ant-design/icons';
import { deleteLoadInfo } from '@/service/info';
import { getMemoryList } from '@/service/memory';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import { useTrainingStore } from '@/store/useTrainingStore';
import { Button, Input, message, Modal } from 'antd';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import ModelConfigModal from '@/components/modelConfigModal';
import { EVENT } from '@/utils/event';
import { tabs } from './constants';
import CollapseIcon from '@/components/svgs/CollapseIcon';
import ArrowIcon from '@/components/svgs/ArrowIcon';
import SettingsIcon from '@/components/svgs/SettingsIcon';
import TrashIcon from '@/components/svgs/TrashIcon';
import DocumentIcon from '@/components/svgs/DocumentIcon';
import classNames from 'classnames';
import { ROUTER_PATH } from '@/utils/router';
import { useModelConfigStore } from '@/store/useModelConfigStore';

const Menu = () => {
  const pathname = usePathname();
  const router = useRouter();
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const loadInfo = useLoadInfoStore((state) => state.loadInfo);
  const clearLoadInfo = useLoadInfoStore((state) => state.clearLoadInfo);
  const serviceStarted = useTrainingStore((state) => state.serviceStarted);

  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState('');
  const [deleteConfirmLoading, setDeleteConfirmLoading] = useState(false);
  const [showModelConfig, setShowModelConfig] = useState(false);

  const isTraining = useTrainingStore((state) => state.isTraining);
  const trainSuspended = useTrainingStore((state) => state.trainSuspended);

  const disabledChangeParams = useMemo(() => {
    return isTraining || trainSuspended;
  }, [isTraining, trainSuspended]);

  const isRegistered = useMemo(() => {
    return loadInfo?.status === 'online';
  }, [loadInfo]);

  const name = useMemo(() => {
    return loadInfo?.name || 'Second Me';
  }, [loadInfo]);

  const statusJudge = async (path: string, e: any) => {
    // Check for training page access - require at least 3 memories
    if (path === ROUTER_PATH.TRAIN_TRAINING) {
      try {
        const memoryResponse = await getMemoryList();

        if (memoryResponse.data.code === 0) {
          const memories = memoryResponse.data.data;

          if (memories.length < 3) {
            e.preventDefault();
            message.error({
              content: 'You need to add at least 3 memories before training',
              duration: 3
            });
            // Redirect to memories page
            router.push(ROUTER_PATH.TRAIN_MEMORIES);

            return;
          }
        }
      } catch (error) {
        console.error('Error checking memory count:', error);
      }
    }

    // Check for playground and applications access
    if (
      path === ROUTER_PATH.PLAYGROUND ||
      path.startsWith(ROUTER_PATH.PLAYGROUND) ||
      path === ROUTER_PATH.APPLICATIONS
    ) {
      if (!serviceStarted) {
        e.preventDefault();
        message.info({
          content: 'Please start your model service first',
          duration: 2
        });
      }
    }
  };

  const handleDeleteSecondMe = () => {
    if (!loadInfo) return;

    setDeleteConfirmText('');
    setDeleteModalVisible(true);
  };

  const handleConfirmDelete = async () => {
    if (deleteConfirmText !== 'DELETE SECOND ME') return;

    setDeleteConfirmLoading(true);

    try {
      // Get upload data from localStorage
      const res = await deleteLoadInfo(loadInfo!.name);

      if (res.data.code !== 0) {
        message.error(res.data.message);
      } else {
        localStorage.clear();
        // Clear the loadInfo store
        clearLoadInfo();
        useModelConfigStore.getState().deleteModelConfig();
        message.success(`${loadInfo!.name} and all related data deleted successfully`);
        router.push(ROUTER_PATH.HOME);
      }
    } catch (error) {
      console.error('Error deleting Second Me:', error);
      message.error('Failed to delete Second Me');
    } finally {
      setDeleteConfirmLoading(false);
      setDeleteModalVisible(false);
    }
  };

  useEffect(() => {
    const toggleModelConfig = () => {
      setShowModelConfig((prevState) => !prevState);
    };

    addEventListener(EVENT.SHOW_MODEL_CONFIG_MODAL, toggleModelConfig);

    return () => {
      removeEventListener(EVENT.SHOW_MODEL_CONFIG_MODAL, toggleModelConfig);
    };
  }, []);

  return (
    <>
      <div
        className={`${
          isSidebarCollapsed ? 'w-16' : 'w-64'
        } bg-white/60 backdrop-blur-sm border-r border-gray-800/5 flex-shrink-0 transition-all duration-300`}
      >
        <div className="h-full flex flex-col">
          <div className="relative px-3 py-4 border-b border-gray-800/5 flex items-center justify-center">
            <div className="flex items-center space-x-3 h-10 px-3">
              <div className="w-8 h-8 rounded-full shadow-md overflow-hidden flex-shrink-0 bg-blue-100 flex items-center justify-center">
                <span className="text-blue-600 font-medium">
                  {name ? name[0].toUpperCase() : 'S'}
                </span>
              </div>
              {!isSidebarCollapsed && (
                <h1 className="text-base font-medium text-gray-700 truncate my-auto">{name}</h1>
              )}
            </div>
            <button
              className="absolute right-0 top-[calc(50%+20px)] -translate-y-1/2 -mr-3 w-6 h-6 rounded-full bg-white shadow-[0_2px_4px_rgba(0,0,0,0.1)] flex items-center justify-center hover:bg-gray-50 transition-colors"
              onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            >
              <CollapseIcon className="w-3 h-3 text-gray-400" collapsed={!isSidebarCollapsed} />
            </button>
          </div>

          <nav className="flex-1 px-3 py-2 overflow-y-auto space-y-4">
            {tabs.map((tab) => (
              <div key={tab.path} className="space-y-1">
                {/* Main tab item */}
                <div className="flex items-center">
                  <Link
                    className={`group flex flex-1 items-center justify-between gap-2 px-3 py-2 text-sm rounded-lg transition-all duration-200 ${
                      pathname.startsWith(tab.path)
                        ? 'bg-blue-50 text-blue-600 font-medium'
                        : 'text-gray-600 hover:bg-blue-50/50 hover:text-blue-600'
                    }`}
                    href={tab.path}
                    onClick={(e) => {
                      // For Train Second Me, show tutorial instead of navigation
                      if (tab.name !== 'Train Second Me') {
                        statusJudge(tab.path, e);
                      }
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`transition-colors duration-200 ${
                          pathname.startsWith(tab.path)
                            ? 'text-blue-600'
                            : 'text-gray-400 group-hover:text-blue-500'
                        }`}
                      >
                        {tab.icon}
                      </span>
                      <span
                        className={classNames(
                          'font-medium transition-opacity line-clamp-1',
                          isSidebarCollapsed ? 'opacity-0' : 'opacity-100'
                        )}
                      >
                        {tab.name}
                      </span>
                    </div>

                    {/* Dropdown arrow for items with submenus */}
                    {!isSidebarCollapsed && tab.subTabs && (
                      <button className="p-1 rounded-md hover:bg-blue-50/50 mr-1">
                        <ArrowIcon className={pathname.startsWith(tab.path) ? 'rotate-90' : ''} />
                      </button>
                    )}
                  </Link>
                </div>

                {/* Sub Tabs with improved styling */}
                {!isSidebarCollapsed && !!tab.subTabs.length && pathname.startsWith(tab.path) && (
                  <div className="ml-4 space-y-1 mt-1 border-l-2 border-blue-100 pl-2">
                    {tab.subTabs.map((subTab) => (
                      <div
                        key={subTab.path}
                        className={`group cursor-pointer flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-all duration-200 ${
                          pathname === subTab.path
                            ? 'bg-blue-50 text-blue-600 font-medium'
                            : 'text-gray-500 hover:bg-blue-50/50 hover:text-blue-600'
                        }`}
                        onClick={() => {
                          if (tab.path === ROUTER_PATH.APPLICATIONS && !isRegistered) {
                            dispatchEvent(new Event(EVENT.SHOW_REGISTER_MODAL));

                            return;
                          }

                          router.push(subTab.path);
                        }}
                      >
                        {subTab.icon && (
                          <span
                            className={`transition-colors duration-200 ${
                              pathname === subTab.path
                                ? 'text-blue-600'
                                : 'text-gray-400 group-hover:text-blue-500'
                            }`}
                          >
                            {subTab.icon}
                          </span>
                        )}
                        <span>{subTab.name}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </nav>

          <div className="px-4 py-3 border-t-2 border-gray-800/10 space-y-2">
            <button
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 rounded-md transition-colors hover:bg-gray-800/5"
              onClick={() =>
                window.open('https://secondme.gitbook.io/secondme/getting-started', '_blank')
              }
            >
              <DocumentIcon className="w-4 h-4 shrink-0" />
              <span
                className={classNames(
                  'font-medium transition-opacity line-clamp-1 flex-1 text-start',
                  isSidebarCollapsed ? 'opacity-0' : 'opacity-100'
                )}
              >
                Tutorial
              </span>
            </button>

            <button
              className={classNames(
                'w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 rounded-md transition-colors hover:bg-gray-800/5',
                disabledChangeParams && 'opacity-50 cursor-not-allowed'
              )}
              onClick={() => {
                if (disabledChangeParams) {
                  message.warning('Cancel the current training run to configure the model');

                  return;
                }

                setShowModelConfig(true);
              }}
            >
              <SettingsIcon className="w-4 h-4 shrink-0" />
              <span
                className={classNames(
                  'font-medium transition-opacity line-clamp-1 flex-1 text-start',
                  isSidebarCollapsed ? 'opacity-0' : 'opacity-100'
                )}
              >
                Support Model Config
              </span>
            </button>

            <button
              className="w-full flex items-center justify-start  gap-2 px-3 py-2 text-sm text-red-600 hover:text-red-700 rounded-md transition-colors hover:bg-red-50"
              onClick={() => handleDeleteSecondMe()}
            >
              <TrashIcon className="w-4 h-4 shrink-0" />
              <span
                className={classNames(
                  'font-medium transition-opacity line-clamp-1 flex-1 text-start',
                  isSidebarCollapsed ? 'opacity-0' : 'opacity-100'
                )}
              >
                Delete Second Me
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Delete Second Me Confirmation Modal */}
      <Modal
        footer={[
          <Button key="cancel" onClick={() => setDeleteModalVisible(false)}>
            Cancel
          </Button>,
          <Button
            key="delete"
            danger
            disabled={deleteConfirmText !== 'DELETE SECOND ME'}
            loading={deleteConfirmLoading}
            onClick={handleConfirmDelete}
            type="primary"
          >
            Delete
          </Button>
        ]}
        onCancel={() => setDeleteModalVisible(false)}
        open={deleteModalVisible}
        title={
          <div className="text-red-600 flex items-center gap-2">
            <ExclamationCircleOutlined /> Delete {name}
          </div>
        }
      >
        <div className="space-y-4">
          <p>
            This action will permanently delete {name} and all associated data. This action cannot
            be undone.
          </p>
          <div className="bg-red-50 p-3 border border-red-200 rounded-md">
            <p className="text-sm text-red-700">
              To confirm deletion, please type <span className="font-bold">DELETE SECOND ME</span>{' '}
              in the field below:
            </p>
          </div>
          <Input
            onChange={(e) => setDeleteConfirmText(e.target.value)}
            placeholder="Type DELETE SECOND ME to confirm"
            status={deleteConfirmText && deleteConfirmText !== 'DELETE SECOND ME' ? 'error' : ''}
            value={deleteConfirmText}
          />
          {deleteConfirmText && deleteConfirmText !== 'DELETE SECOND ME' && (
            <p className="text-red-500 text-sm">
              Text does not match. Please type exactly: DELETE SECOND ME
            </p>
          )}
        </div>
      </Modal>

      <ModelConfigModal onClose={() => setShowModelConfig(false)} open={showModelConfig} />
    </>
  );
};

export default Menu;
