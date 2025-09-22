import type { CommonResponse, EmptyResponse } from '../types/responseModal';
import { Request } from '../utils/request';

// Memory processing status
export type EmbeddingStatus = 'INITIALIZED' | 'PROCESSING' | 'SUCCESS' | 'FAILED';
export type ExtractStatus = 'INITIALIZED' | 'PROCESSING' | 'SUCCESS' | 'FAILED';

// Basic interface definition for Memory
export interface MemoryFile {
  id: string;
  name: string;
  title: string;
  create_time: string;
  document_size: number;
  embedding_status: EmbeddingStatus;
  extract_status: ExtractStatus;
  insight: string | null;
  mime_type: string;
  raw_content: string;
  summary: string | null;
  url: string;
  user_description: string;
}

interface MetaData {
  description: string;
  name: string;
}

interface UploadMemoryRes {
  created_at: string;
  id: string;
  meta_data: MetaData;
  name: string;
  path: string;
  type: string;
}

export const getMemoryList = () => {
  return Request<CommonResponse<MemoryFile[]>>({
    method: 'get',
    url: ' /api/documents/list'
  });
};

export const uploadMemory = (formData: FormData) => {
  return Request<CommonResponse<UploadMemoryRes>>({
    method: 'post',
    url: '/api/memories/file',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
};

export const deleteMemory = (id: string) => {
  return Request<EmptyResponse>({
    method: 'delete',
    url: `/api/memories/file/id/${id}`
  });
};
