import copy
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple

import openai
import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.file_data import Document
from lpm_kernel.file_data.document_repository import DocumentRepository
from .prompts import Prompts

logger = get_train_process_logger()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

load_dotenv(override=True)


class TopicGenerate():
    _input_keys: List[str] = ["noteMemory", "topics", "preferredLanguage"]
    _output_keys: List[str] = ["topics"]
    _must_keys: List[str] = ["noteMemory", "topics", "preferredLanguage"]

    Topic_Generate_DEFAULT_PROMPT = {
        "topics_generate": Prompts.Topics_Generate_SYSTEM_PROMPT,
        "topics_generate_zh": Prompts.Topics_Generate_zh_SYSTEM_PROMPT
    }

    model_params = {
        "max_tokens": 20000,
        "top_p": 0.9,
        "temperature": 0.9,
        "request_timeout": 60,
        "response_format": {"type": "json_object"},
        "max_retries": 3
    }

    @property
    def _api_type(self):
        return "topics_generate"

    @property
    def input_keys(self) -> List[str]:
        return self._input_keys

    @property
    def output_keys(self) -> List[str]:
        return self._output_keys

    @property
    def must_keys(self) -> List[str]:
        return self._must_keys

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_threads = 8

        self.class_name = self.__class__.__name__
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model_params.update(**kwargs)

        self.langfuse_dict = {
            k: self._get_langfuse_prompt(k, v) for k, v in self.Topic_Generate_DEFAULT_PROMPT.items()
        }

        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()

        self.llms: Dict[str, openai.OpenAI] = {
            k: openai.OpenAI(
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
            for k, _ in self.Topic_Generate_DEFAULT_PROMPT.items()
        }
        self.model_name = self.user_llm_config.chat_model_name

    def _get_langfuse_prompt(self, prompt_key, default_prompt) -> Dict[str, Any]:
        # 直接使用默认提示词，不再尝试从 langfuse 获取
        system_prompt = default_prompt
        logging.info(f"Using local default prompt for {prompt_key}")
        return {
            "lf_prompt": None,
            "system_prompt": system_prompt
        }

    def _preprocess(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
            处理掉无关信息，只保留topicName和topicDescription给到大模型
        """
        pre_topics = []
        for topic in topics:
            pre_topics.append({
                "topicName": topic["topicName"],
                "topicDescription": topic["topicDescription"]
            })
        return pre_topics

    def _process_memory_format(self, memories: List[dict]) -> str:
        """
        将memory数据转换为NLP格式的字符串

        Args:
            memories: 包含note的memory列表，格式为[{"type": "note", "content": ..., "id": ...}]

        Returns:
            str: 格式化后的memory字符串
        """
        processed_memories = ""
        for memory in memories:
            memory_id = memory.get("id", "")
            content = memory.get("content", "")

            processed_memories += f"Note {memory_id} Content: {content}\n"

        return processed_memories

    def _postprocess(self, topics: List[Dict[str, Any]], updated_topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        后处理函数，将新生成的topics与原有topics合并

        Args:
            topics: 原有的topics列表
            updated_topics: 新生成的topics列表

        Returns:
            List[Dict]: 合并后的完整topics列表
        """
        final_topics = copy.deepcopy(topics)  # 深拷贝原有topics

        # 处理新生成的topics
        for topic in updated_topics:
            # 检查是否已存在相同名称的topic
            existing_topic = next((t for t in final_topics if t["topicName"] == topic["topicName"]), None)

            if existing_topic:
                # 如果存在，更新description，保留原有的topicId
                existing_topic["topicDescription"] = topic["topicDescription"]
                if "topicId" not in existing_topic:
                    existing_topic["topicId"] = ""
            else:
                # 如果不存在，创建新的topic
                new_topic = {
                    "topicName": topic["topicName"],
                    "topicId": "",  # 新topic的topicId为空字符串
                    "topicDescription": topic["topicDescription"],
                    "relatedNotes": [],
                }
                final_topics.append(new_topic)

        # 确保所有topic都有topicId字段
        for topic in final_topics:
            if "topicId" not in topic:
                topic["topicId"] = ""

        return final_topics

    def _clean_json_response(self, response: str) -> str:
        """
        清理LLM返回结果中的markdown代码块标记

        Args:
            response: LLM返回的原始字符串

        Returns:
            str: 清理后的JSON字符串
        """
        # 移除开头的```json或```
        if response.strip().startswith('```json'):
            response = response.strip()[7:]  # 移除```json
        elif response.strip().startswith('```'):
            response = response.strip()[3:]  # 移除```

        # 移除结尾的```
        if response.strip().endswith('```'):
            response = response.strip()[:-3]  # 移除结尾的```

        return response.strip()

    def _process_batch(self, pre_topics, batch, preferredLanguage):
        """
        处理一批memory数据，返回更新后的topics和memory关联关系

        Args:
            pre_topics: 预处理后的topics列表，只包含topicName和topicDescription
            batch: 一批memory数据
            preferredLanguage: 首选语言

        Returns:
            Tuple[List[Dict], Dict]: 更新后的topics列表和memory关联关系
        """
        # 将batch转换为NLP格式
        processed_batch = self._process_memory_format(batch)
        logging.info(f"processed_batch: {processed_batch}")

        # 获取prompt
        if preferredLanguage == "简体中文/Simplified Chinese" or "chinese":
            message = Prompts.return_topics_generate_prompt(
                system_prompt=self.langfuse_dict["topics_generate_zh"]["system_prompt"],
                cur_topics=pre_topics,  # 使用预处理后的topics list
                memory_content=processed_batch,
                prefer_lang=preferredLanguage
            )
            llm = self.llms["topics_generate_zh"]
        else:
            message = Prompts.return_topics_generate_prompt(
                system_prompt=self.langfuse_dict["topics_generate"]["system_prompt"],
                cur_topics=pre_topics,  # 使用预处理后的topics list
                memory_content=processed_batch,
                prefer_lang=preferredLanguage
            )
            llm = self.llms["topics_generate"]

        updated_topics = []
        # 为每个topic维护独立的memory映射
        topic_memory_relations = defaultdict(lambda: defaultdict(list))

        # 调用LLM生成内容
        try:
            answer = llm.chat.completions.create(
                model=self.model_name,
                messages=message,
            )
            result = answer.choices[0].message.content

            # 解析结果
            try:
                logging.info(f"llm return: result: {result}")
                # 清理LLM返回结果中的markdown标记
                cleaned_result = self._clean_json_response(result)
                logging.info(f"cleaned result: {cleaned_result}")

                data = json.loads(cleaned_result)

                # 处理LLM返回的结果
                if "topics" in data:
                    for topic in data["topics"]:
                        if "topicName" in topic and "topicDescription" in topic:
                            logging.info(f"-------Current topic: {topic}--------")
                            updated_topics.append({
                                "topicName": topic["topicName"],
                                "topicDescription": topic["topicDescription"]
                            })

                            # 如果LLM返回了memory关联信息
                            if "relatedMemories" in topic:
                                for memory in topic["relatedMemories"]:
                                    memory_id = memory.get("id")

                                    # 使用正则表达式提取纯数字ID 防止LLM不稳定导致的id错误
                                    if memory_id:
                                        # 匹配连续的数字
                                        numeric_id = re.search(r'\d+', str(memory_id))
                                        if numeric_id:
                                            memory_id = numeric_id.group()

                                    memory_type = memory.get("type")  # "note" 或 "chat"
                                    if memory_id and memory_type:
                                        # 为当前topic添加memory关联
                                        topic_memory_relations[topic["topicName"]][memory_type].append(memory_id)

                logging.info(f"topic_memory_relations: {topic_memory_relations}")
                return updated_topics, topic_memory_relations

            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response: {e}")
                return [], {}

        except Exception as e:
            logging.error(f"Error in LLM processing: {e}")
            return [], {}

    def _incremental_update(self, noteMemory: List[Document],
                            topics: List[Dict[str, Any]], preferredLanguage: str) -> dict[str, list[dict[str, Any]]]:
        """
        增量更新topics

        Args:
            noteMemory: 笔记记忆列表
            topics: 现有的topics列表
            preferredLanguage: 首选语言

        Returns:
            List[Dict]: 更新后的topics列表
        """
        # 预处理
        pre_topics = self._preprocess(topics)

        # 合并所有memory
        all_memories = []
        for doc in noteMemory:
            all_memories.append({"type": "note", "content": doc.raw_content, "id": doc.id})

        batch_size = min(20, len(all_memories))
        memory_batches = [all_memories[i:i + batch_size] for i in range(0, len(all_memories), batch_size)]
        logging.info(f"将{len(all_memories)}条记忆分成{len(memory_batches)}批，每批最多{batch_size}条")

        # 用于存储每个topic的memory关联关系
        topic_memory_relations = defaultdict(lambda: {"note": set(), "chat": set()})
        all_updated_topics = []

        # 处理每一批memory - 添加进度条
        with tqdm(total=len(memory_batches), desc="处理memory批次", unit="批") as pbar:
            for i, batch in enumerate(memory_batches):
                pbar.set_description(f"处理第{i + 1}/{len(memory_batches)}批 ({len(batch)}条记忆)")
                # print(f"Current batch: {batch}")

                updated_topics, batch_relations = self._process_batch(pre_topics, batch, preferredLanguage)

                # 更新每个topic的memory关联关系
                for topic in updated_topics:
                    topic_name = topic["topicName"]
                    # 使用当前topic的memory映射更新关联关系
                    if topic_name in batch_relations:
                        for memory_type, memory_ids in batch_relations[topic_name].items():
                            topic_memory_relations[topic_name][memory_type].update(memory_ids)

                # 更新pre_topics，包含新生成的topics
                pre_topics.extend(updated_topics)
                all_updated_topics.extend(updated_topics)

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "已处理记忆": f"{(i + 1) * batch_size if (i + 1) * batch_size <= len(all_memories) else len(all_memories)}/{len(all_memories)}",
                    "生成topics": len(set(topic["topicName"] for topic in all_updated_topics))
                })

        # 后处理，将memory关联关系添加到topics中
        final_topics = self._postprocess(topics, all_updated_topics)

        # 更新每个topic的relatedNotes和relatedChats，将set转换回list
        for topic in final_topics:
            topic_name = topic["topicName"]
            topic["relatedNotes"] = list(topic_memory_relations[topic_name]["note"])

        final_results = {}
        final_results["topics"] = final_topics
        return final_results

    # @observe(name="topics_generate")
    def topics_generate(self, topics, preferredLanguage: str = "English/English") -> Dict[str, Any]:

        start_time = time.time()

        doc_repository = DocumentRepository()
        documents = doc_repository.list()

        updated_topics = topics

        if len(documents) == 0:
            logging.warning("note is empty, cannot update topics!")
        else:
            logging.info(f"Let's start to generate topics...")
            updated_topics = self._incremental_update(documents, topics, preferredLanguage)
            end_time = time.time()
            logging.info(f"topics generate cost: {end_time:.2f} - {start_time:.2f} seconds")

        result_json = json.dumps(updated_topics, ensure_ascii=False, indent=2)
        os.makedirs("resources/data/stage2/topics", exist_ok=True)
        with open("resources/data/stage2/topics/topic.json", "w", encoding="utf-8") as f:
            f.write(result_json)

        return updated_topics


if __name__ == "__main__":
    topic_generate = TopicGenerate()
    topics = []
    preferredLanguage = "zh"
    result = topic_generate.topics_generate(topics=topics, preferredLanguage=preferredLanguage)
