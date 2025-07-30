from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from openai import OpenAI
from tqdm import tqdm

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()


class BaseData(ABC):
    def __init__(self, is_cot: bool = True, max_workers: int = 10):
        self.max_workers = max_workers
        self.is_cot = is_cot

        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()

        if self.user_llm_config is None:
            self.client = None
            self.model_name = None
            self.reasoning_client = None
            self.reasoning_model_name = None
        else:
            self.client = OpenAI(
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
            self.model_name = self.user_llm_config.chat_model_name
            self.reasoning_client = OpenAI(
                api_key=self.user_llm_config.thinking_api_key,
                base_url=self.user_llm_config.thinking_endpoint,
            )
            self.reasoning_model_name = self.user_llm_config.thinking_model_name

    def preprocess(self):
        pass

    def build_messages(self):
        pass

    def build_responses(self, messages_list: List[List[Dict[str, Any]]]):

        def process_request(messages):
            try:
                if self.is_cot:
                    response = self.reasoning_client.chat.completions.create(
                        model=self.reasoning_model_name,
                        messages=messages,
                    )
                    result = response.choices[0].message
                    logger.info("===============================================")
                    logger.info(messages[1]["content"])
                    logger.info("-----------------------------------------------")
                    logger.info("<think>" + result.reasoning_content + "</think>")
                    logger.info("<answer>" + result.content + "</answer>")
                    logger.info("")
                    return "<think>" + result.reasoning_content + "</think>\n\n<answer>" + result.content + "</answer>"
                else:
                    responses = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=3000,
                    ).choices[0].message.content
                    return responses
            except Exception as e:
                logger.error(f"Raise ERROR: {e} WHEN GENERATE RESPONSE")
                raise

        with ThreadPoolExecutor(max_workers=max(min(self.max_workers, len(messages_list)), 1)) as executor:
            futures = [executor.submit(process_request, messages) for messages in messages_list]
            results = []

            for future in tqdm(futures, total=len(messages_list), desc="Generating responses"):
                result = future.result()
                results.append(result)

        return results

    @abstractmethod
    def run(self):
        pass

    def postprocess(self):
        pass
