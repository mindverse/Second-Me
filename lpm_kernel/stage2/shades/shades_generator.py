import re, os
from typing import List, Tuple, Dict, Any
import copy
import math
import time
import tiktoken
import traceback
import itertools
import statistics
import json
import random
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm  # 添加tqdm导入
from dotenv import load_dotenv
# from langfuse.decorators import observe

import openai
from requests.exceptions import Timeout
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, OrderedDict
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.file_data import Document
from lpm_kernel.file_data.document_repository import DocumentRepository

from lpm_kernel.configs.logging import get_train_process_logger

from .prompts import Prompts

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = get_train_process_logger()

# 加载环境变量
load_dotenv(override=True)


class ShadeGenerate():
    _input_keys: List[str] = ["topics", "shades", "preferredLanguage"]
    _output_keys: List[str] = ["shades"]
    _must_keys: List[str] = ["topics", "shades", "preferredLanguage"]

    Shade_Generate_DEFAULT_PROMPT = {
        "shades_generate": Prompts.Shades_Generate_SYSTEM_PROMPT,
        "shades_update": Prompts.Shades_Update_SYSTEM_PROMPT,
        "shades_generate_zh": Prompts.Shades_Generate_zh_SYSTEM_PROMPT,
        "shades_update_zh": Prompts.Shades_Update_zh_SYSTEM_PROMPT
    }

    model_params = {
        "max_tokens": 30000,
        "top_p": 0.9,
        "temperature": 0.9,
        "request_timeout": 100,
        "response_format": {"type": "json_object"},
        "max_retries": 1
    }

    @property
    def _api_type(self):
        return "shades_generate"

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
        self.prompts = Prompts()
        self.max_threads = 8
        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()

        self.llms: Dict[str, openai.OpenAI] = {
            k: openai.OpenAI(
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
            for k, _ in self.Shade_Generate_DEFAULT_PROMPT.items()
        }
        self.model_name = self.user_llm_config.chat_model_name

        self.class_name = self.__class__.__name__
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model_params.update(**kwargs)

        self.langfuse_dict = {
            k: self._get_langfuse_prompt(k, v) for k, v in self.Shade_Generate_DEFAULT_PROMPT.items()
        }

    def _get_langfuse_prompt(self, prompt_key, default_prompt) -> Dict[str, Any]:
        # 直接使用默认提示词，不再尝试从 langfuse 获取
        system_prompt = default_prompt
        logger.info(f"Using local default prompt for {prompt_key}")
        return {
            "lf_prompt": None,
            "system_prompt": system_prompt
        }

    def _preprocess_topics_format(self, topics: List[dict], shades: List[dict]) -> tuple[str | Any, str | Any]:
        processed_topics = ""
        for topic in topics:
            cur_topic = f"topicName: {topic['topicName']}, topicDescription: {topic['topicDescription']}\n"
            processed_topics += cur_topic
        processed_shades = ""
        if shades == []:
            processed_shades = "No shades"
        else:
            for shade in shades:
                cur_shade = f"shadeName: {shade['shadeName']}, shadeIcon: {shade['shadeIcon']}, shadeDescription: {shade['shadeDescription']}\n"
                cur_shade += f"sourceTopics: {shade['sourceTopics']}\n"
                cur_shade += f"confidenceLevel: {shade['confidenceLevel']}\n"
                processed_shades += cur_shade
        return processed_topics, processed_shades

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

        # Fix unquoted shadeIcon values
        import re
        # Pattern to match "shadeIcon": followed by an unquoted emoji or text
        # This handles cases like "shadeIcon": 🏙️, or "shadeIcon": ❤️
        pattern = r'"shadeIcon":\s*([^",\s][^,}\]]*?)(?=,|\s*[}\]])'
        
        def replace_icon(match):
            icon_value = match.group(1).strip()
            # If the icon is not already quoted, quote it
            if not (icon_value.startswith('"') and icon_value.endswith('"')):
                return f'"shadeIcon": "{icon_value}"'
            return match.group(0)
        
        response = re.sub(pattern, replace_icon, response)

        # Additional JSON cleaning - fix common issues
        # Fix trailing commas before closing brackets/braces
        response = re.sub(r',(\s*[}\]])', r'\1', response)
        
        # Note: Removed the general unquoted string fix as it was causing issues with array elements
        # The shadeIcon-specific fix should be sufficient for this use case

        return response.strip()

    # @observe(name="shades_generate")
    def shades_generate(self, topics, shades, preferredLanguage: str = "简体中文/Simplified Chinese") -> Dict[str, Any]:

        if isinstance(topics, dict):
            topics = topics.get("topics", [])
        if isinstance(shades, dict):
            shades = shades.get("shades", [])

        input_topics, input_shades = self._preprocess_topics_format(topics, shades)  # 预处理topics 处理为NLP字符串 方便模型理解
        logger.info(f"input topics: {input_topics}")
        logger.info(f"input shades: {input_shades}")

        # 添加进度条来跟踪shade生成进度
        total_steps = 4  # 预处理、LLM调用、后处理、去重
        with tqdm(total=total_steps, desc="生成shade", unit="步") as pbar:

            # 步骤1: 预处理完成
            pbar.set_description("预处理topics和shades")
            pbar.update(1)

            # 判断当前是cold start 还是update
            if shades == []:
                logger.info("Shades Cold Start!")
                pbar.set_description("冷启动生成shade")

                if preferredLanguage == "简体中文/Simplified Chinese" or "chinese":
                    # 获取prompt
                    message = Prompts.return_shades_generate_prompt(
                        system_prompt=self.langfuse_dict["shades_generate_zh"]["system_prompt"],
                        topics_list=input_topics,
                        prefer_lang=preferredLanguage
                    )
                    # 启服务
                    llm = self.llms["shades_generate_zh"]
                else:
                    message = Prompts.return_shades_generate_prompt(
                        system_prompt=self.langfuse_dict["shades_generate"]["system_prompt"],
                        topics_list=input_topics,
                        prefer_lang=preferredLanguage
                    )
                    # 启服务
                    llm = self.llms["shades_generate"]

                    # 步骤2: 调用LLM生成内容
                pbar.set_description("调用LLM生成shade")
                try:
                    answer = llm.chat.completions.create(
                        model=self.model_name,
                        messages=message,
                    )
                    result = answer.choices[0].message.content
                    if result is None:
                        logger.error(f"LLM return Nothing!")
                        return {"shades": []}

                    # process llm result
                    try:
                        logger.info(f"llm return: result: {result}")
                        cleaned_result = self._clean_json_response(result)
                        pre_shades = json.loads(cleaned_result)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse LLM response: {e}")
                        raise e
                except Exception as e:
                    logger.error(f"Cannot fetch LLM correctly: {e}")
                    raise e
                pbar.update(1)

            else:
                logger.info("Shades Update!")
                pbar.set_description("更新现有shade")

                if preferredLanguage == "简体中文/Simplified Chinese" or "chinese":
                    # 获取prompt
                    message = Prompts.return_shades_update_prompt(
                        system_prompt=self.langfuse_dict["shades_update_zh"]["system_prompt"],
                        cur_shades=input_shades,
                        topics_list=input_topics,
                        prefer_lang=preferredLanguage
                    )
                    # 启服务
                    llm = self.llms["shades_update_zh"]
                else:
                    message = Prompts.return_shades_update_prompt(
                        system_prompt=self.langfuse_dict["shades_update"]["system_prompt"],
                        cur_shades=input_shades,
                        topics_list=input_topics,
                        prefer_lang=preferredLanguage
                    )
                    # 启服务
                    llm = self.llms["shades_update"]

                    # 步骤2: 调用LLM生成内容
                pbar.set_description("调用LLM更新shade")
                try:
                    answer = llm.chat.completions.create(
                        model=self.model_name,
                        messages=message,
                    )
                    result = answer.choices[0].message.content

                    # process llm result
                    try:
                        logger.info(f"llm return: result: {result}")
                        cleaned_result = self._clean_json_response(result)
                        pre_shades = json.loads(cleaned_result)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse LLM response: {e}")
                        raise e
                except Exception as e:
                    logger.error(f"Cannot fetch LLM correctly: {e}")
                    raise e
                pbar.update(1)

            # 步骤3: 为当前shade匹配related Memory
            pbar.set_description("匹配相关记忆")
            for shade in pre_shades:
                # 获取当前shade的sourceTopics
                source_topics = shade["sourceTopics"]

                # 获取当前shade的relatedNotes
                related_notes = []
                for topic in source_topics:
                    for cur_topic in topics:
                        # print(cur_topic)
                        if cur_topic["topicName"] == topic:
                            related_notes.extend(cur_topic["relatedNotes"])
                shade["relatedNotes"] = related_notes
            pbar.update(1)

            # 步骤4: 去重处理
            pbar.set_description("处理重复记忆")
            final_result = {}
            final_result["shades"] = pre_shades
            # 为related memory  去重
            for shade in pre_shades:
                related_notes = shade["relatedNotes"]
                related_notes = list(set(related_notes))
                shade["relatedNotes"] = related_notes
            pbar.update(1)

            pbar.set_description("shade生成完成")

        logger.info(f"final_result: {final_result}")
        return final_result


class ShadeContentGenerate():
    _input_keys: List[str] = ["noteMemory", "topics", "shades", "preferredLanguage"]
    _output_keys: List[str] = ["shades"]
    _must_keys: List[str] = ["noteMemory", "topics", "shades", "preferredLanguage"]

    Shade_Content_Generate_DEFAULT_PROMPT = {
        "shades_content_generate": Prompts.Shades_Content_SYSTEM_PROMPT,
        "shades_content_generate_zh": Prompts.Shades_Content_zh_SYSTEM_PROMPT,
        "shades_content_update": Prompts.Shades_Content_Update_SYSTEM_PROMPT,
        "shades_content_update_zh": Prompts.Shades_Content_Update_zh_SYSTEM_PROMPT,
    }

    model_params = {
        "max_tokens": 30000,
        "top_p": 0.9,
        "temperature": 0.9,
        "request_timeout": 100,
        "response_format": {"type": "json_object"},
        "max_retries": 1
    }

    @property
    def _api_type(self):
        return "shades_content_generate"

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
        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()

        self.llms: Dict[str, openai.OpenAI] = {
            k: openai.OpenAI(
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
            for k, _ in self.Shade_Content_Generate_DEFAULT_PROMPT.items()
        }
        self.model_name = self.user_llm_config.chat_model_name

        self.class_name = self.__class__.__name__
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self.model_params.update(**kwargs)
        self.token_threshold = kwargs.get("token_threshold", 10000)  # 默认token阈值

        # 初始化langfuse字典
        self.langfuse_dict = {
            k: self._get_langfuse_prompt(k, v) for k, v in self.Shade_Content_Generate_DEFAULT_PROMPT.items()
        }

    def _get_langfuse_prompt(self, prompt_key, default_prompt) -> Dict[str, Any]:
        # 直接使用默认提示词，不再尝试从 langfuse 获取
        system_prompt = default_prompt
        logger.info(f"Using local default prompt for {prompt_key}")
        return {
            "lf_prompt": None,
            "system_prompt": system_prompt
        }

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

        # Fix unquoted shadeIcon values
        import re
        # Pattern to match "shadeIcon": followed by an unquoted emoji or text
        # This handles cases like "shadeIcon": 🏙️, or "shadeIcon": ❤️
        pattern = r'"shadeIcon":\s*([^",\s][^,}\]]*?)(?=,|\s*[}\]])'
        
        def replace_icon(match):
            icon_value = match.group(1).strip()
            # If the icon is not already quoted, quote it
            if not (icon_value.startswith('"') and icon_value.endswith('"')):
                return f'"shadeIcon": "{icon_value}"'
            return match.group(0)
        
        response = re.sub(pattern, replace_icon, response)

        # Additional JSON cleaning - fix common issues
        # Fix trailing commas before closing brackets/braces
        response = re.sub(r',(\s*[}\]])', r'\1', response)
        
        # Note: Removed the general unquoted string fix as it was causing issues with array elements
        # The shadeIcon-specific fix should be sufficient for this use case

        return response.strip()

    def _precess_batch(self, memories: List[Document], topics: List[dict], shades: List[dict]):
        """
        处理memory批次，根据token数量决定是否需要分批处理
        Args:
            memories: 记忆列表
            topics: 主题列表
            shades: 标签列表
        """
        # 首先处理所有memory
        processed_memories, processed_topics, processed_shades = self._process_memory_format(memories, topics, shades)
        memory_batches = []

        # 计算处理后的memory的token数量
        memory_tokens = len(self._tokenizer.encode(processed_memories))
        logger.info(f"Current shade related memories' token: {memory_tokens}")

        # 如果token数量小于阈值，直接返回处理后的结果
        if memory_tokens <= self.token_threshold:
            logger.info("Memory tokens within threshold, process all memories")
            memory_batches.append(processed_memories)
            return memory_batches, processed_topics, processed_shades

        # 如果token数量超过阈值，需要分批处理
        logger.info(f"Memory tokens exceed threshold ({self.token_threshold}), processing in batches")

        # 将每条记忆单独处理并计算token
        memory_str = ""
        memory_str_list = []
        memory_token_list = []

        # 处理每条记忆并计算token
        for memory in memories:
            if "noteId" in memory:  # 处理笔记记忆
                memory_str = f"Note {memory.get('noteId', '')} Summary: {memory.get('summary', '')}\n"

            memory_tokens = len(self._tokenizer.encode(memory_str))
            memory_str_list.append(memory_str)
            memory_token_list.append(memory_tokens)

        # 按token限制分组
        current_batch = []
        current_batch_tokens = 0

        for memory_str, memory_tokens in zip(memory_str_list, memory_token_list):
            # 如果当前记忆的token数已经超过阈值，需要单独作为一个batch
            if memory_tokens > self.token_threshold:
                if current_batch:  # 如果当前batch不为空，先保存
                    memory_batches.append("".join(current_batch))
                    current_batch = []
                    current_batch_tokens = 0
                # 将超长的记忆单独作为一个batch
                memory_batches.append(memory_str)
                continue

            # 如果加入当前记忆会超过阈值，保存当前batch并开始新的batch
            if current_batch_tokens + memory_tokens > self.token_threshold:
                memory_batches.append("".join(current_batch))
                current_batch = [memory_str]
                current_batch_tokens = memory_tokens
            else:
                current_batch.append(memory_str)
                current_batch_tokens += memory_tokens

        # 保存最后一个batch
        if current_batch:
            memory_batches.append("".join(current_batch))

        logger.info(f"Split into {len(memory_batches)} batches")
        for i, batch in enumerate(memory_batches):
            batch_tokens = len(self._tokenizer.encode(batch))
            logger.info(f"Batch {i + 1} tokens: {batch_tokens}")

        return memory_batches, processed_topics, processed_shades

    def _process_memory_format(self, memories: List[Document], topics: List[dict], shades: List[dict]) -> tuple[
        str | Any, str | Any, list[dict]]:
        processed_memories = ""
        for memory in memories:
            processed_memories += f"Note {memory.id} Summary: {memory.raw_content}\n"
        processed_topics = ""
        for topic in topics:
            processed_topics += f"Topic {topic.get('topicName', '')} Description: {topic.get('topicDescription', '')}\n"

        return processed_memories, processed_topics, shades

    def _call_llm(self, llm, message, shades: list[dict]):

        # 调用LLM生成内容
        try:
            answer = llm.chat.completions.create(
                model=self.model_name,
                messages=message,
            )
            result = answer.choices[0].message.content

            # process llm result
            try:
                logger.info(f"llm return: result: {result}")
                cleaned_result = self._clean_json_response(result)
                result = json.loads(cleaned_result)
                result = result[0]  # 理论上只有一个shade

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response: {e}")
                logger.info("LLM return parse error, use description as content")
                # 返回默认结构，避免result未定义
                result = {
                    "shadeContent": "",
                    "shadeContentThirdView": ""
                }

        except Exception as e:
            logger.error(f"Cannot fetch LLM correctly: {e}")
            logger.info("Fetch LLM failed, use description as content")
            # 返回默认结构，避免result未定义
            result = {
                "shadeContent": "",
                "shadeContentThirdView": ""
            }

        return result

    # @observe(name="shades_content_generate")
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # 入参获取
        noteMemory = inputs.get("noteMemory", [])
        topics = inputs.get("topics", [])
        shades = inputs.get("shades", [])
        preferredLanguage = inputs.get("preferredLanguage", "English")

        # 使用批处理功能处理memory
        memory_batches, processed_topics, processed_shades = self._precess_batch(noteMemory, topics, shades)

        logger.info("Multiple batches, process each batch")
        cur_shades = processed_shades
        total_batches = len(memory_batches)

        for batch_idx, memory_batch in enumerate(memory_batches, 1):
            logger.info(f"------ Processing batch {batch_idx}/{total_batches} -------")
            # 第一个batch使用generate，后续batch使用update
            if batch_idx == 1:
                logger.info("First batch, using generate prompt")
                if preferredLanguage == "简体中文/Simplified Chinese" or "chinese":
                    message = Prompts.return_shades_content_generate_prompt(
                        system_prompt=self.langfuse_dict["shades_content_generate_zh"]["system_prompt"],
                        topics_list=processed_topics,
                        cur_shade=cur_shades,
                        related_memories=memory_batch,
                        prefer_lang=preferredLanguage
                    )
                    llm = self.llms["shades_content_generate_zh"]
                else:
                    message = Prompts.return_shades_content_generate_prompt(
                        system_prompt=self.langfuse_dict["shades_content_generate"]["system_prompt"],
                        topics_list=processed_topics,
                        cur_shade=cur_shades,
                        related_memories=memory_batch,
                        prefer_lang=preferredLanguage
                    )
                    llm = self.llms["shades_content_generate"]
            else:
                logger.info(f"Subsequent batch {batch_idx}, using update prompt")
                if preferredLanguage == "简体中文/Simplified Chinese" or "chinese":
                    message = Prompts.return_shades_content_update_prompt(
                        system_prompt=self.langfuse_dict["shades_content_update_zh"]["system_prompt"],
                        cur_shade=cur_shades,
                        related_memories=memory_batch,
                        prefer_lang=preferredLanguage
                    )
                    llm = self.llms["shades_content_update_zh"]
                else:
                    message = Prompts.return_shades_content_update_prompt(
                        system_prompt=self.langfuse_dict["shades_content_update"]["system_prompt"],
                        cur_shade=cur_shades,
                        related_memories=memory_batch,
                        prefer_lang=preferredLanguage
                    )
                    llm = self.llms["shades_content_update"]

            cur_shades = self._call_llm(llm, message, processed_shades)
            processed_shades[0]["shadeContent"] = cur_shades.get("shadeContent", "")
            processed_shades[0]["shadeContentThirdView"] = cur_shades.get("shadeContentThirdView", "")

        logger.info(f"------ All batch Process completed -------")
        shades[0]["shadeContent"] = cur_shades.get("shadeContent", "")
        shades[0]["shadeContentThirdView"] = cur_shades.get("shadeContentThirdView", "")

        final_result = {}
        final_result["shades"] = shades
        return final_result


if __name__ == "__main__":

    doc_repository = DocumentRepository()
    documents = doc_repository.list()

    topics_path = "resources/data/stage2/topics/topic.json"
    with open(topics_path, "r") as f:
        topics_data = json.load(f)
        topics = topics_data.get("topics", [])

    # 测试shade generate
    shade_generate = ShadeGenerate()
    shades = []
    preferredLanguage = "简体中文/Simplified Chinese"
    shades_result = shade_generate.shades_generate(topics=topics, shades=shades, preferredLanguage=preferredLanguage)
    logger.info(f"shades generate result: {shades_result}")
    # 保存shades result
    shades_result_json = json.dumps(shades_result, ensure_ascii=False, indent=2)
    os.makedirs("resources/data/stage2/shades", exist_ok=True)
    with open("resources/data/stage2/shades/shades.json", "w", encoding="utf-8") as f:
        f.write(shades_result_json)

    with open("resources/data/stage2/shades/shades.json", "r", encoding="utf-8") as f:
        shades_result = json.load(f)
    final_result = shades_result
    shade_content_generate = ShadeContentGenerate()
    for idx, shade in enumerate(shades_result["shades"]):

        cur_note = []
        cur_topic = []

        for note in documents:
            if note.id in shade["relatedNotes"]:
                cur_note.append(note)
        for topic in topics:
            if topic["topicName"] in shade["sourceTopics"]:
                cur_topic.append(topic)
        shades_content_input = {
            "noteMemory": cur_note,
            "topics": cur_topic,
            "shades": [shade],
            "preferredLanguage": "简体中文/Simplified Chinese"
        }
        shades_content_result = shade_content_generate._call(shades_content_input)
        logger.info(f"shade content generate result: {shades_content_result}")
        final_result["shades"][idx]["shadeContent"] = shades_content_result["shades"][0]["shadeContent"]
        final_result["shades"][idx]["shadeContentThirdView"] = shades_content_result["shades"][0][
            "shadeContentThirdView"]
    # 保存shades content result
    with open("resources/data/stage2/shades/shades_content.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
