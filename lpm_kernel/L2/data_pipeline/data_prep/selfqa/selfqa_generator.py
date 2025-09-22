import concurrent.futures
import traceback
import os
import random
import openai
from tqdm import tqdm
from enum import Enum
from lpm_kernel.L2.data_pipeline.data_prep.selfqa.selfqa_prompt import (
    system_prompt_cn, system_cot_prompt_cn,
    system_prompt_en, system_cot_prompt_en
)
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()

class TqdmLoggingHandler:
    def __init__(self):
        pass
    
    def write(self, msg):
        logger.info(msg.strip())
    
    def flush(self):
        pass
    
tqdm_handler = TqdmLoggingHandler()


def is_english(text: str) -> bool:
    """Check if the text contains only ASCII alphabetic characters.
    
    Args:
        text: The string to check.
        
    Returns:
        True if the text contains only ASCII alphabetic characters, False otherwise.
    """
    return text.isascii() and text.isalpha()


class DataSynthesisMode(Enum):
    LOW = {"user_question_nums":3, "user_bind_question_nums":3}
    MEDIUM = {"user_question_nums":2, "user_bind_question_nums":2}
    HIGH = {"user_question_nums":1, "user_bind_question_nums":1}


class SelfQA:
    def __init__(
        self,
        user_name: str,
        user_input_introduction: str,
        user_global_bio: str,
        preferred_language: str = "en",
        is_cot: bool = True
    ):
        """Initialize the SelfQA instance.
        
        Args:
            user_name: The name of the user.
            user_input_introduction: User's introduction.
            user_global_bio: User's global biography.
            preferred_language: User's preferred language, 'en' for English, default is 'en'.
        """
        self.user_name = user_name
        self.user_input_introduction = user_input_introduction
        self.user_global_bio = user_global_bio
        self.preferred_language = preferred_language
        self.is_cot = is_cot
        user_llm_config_service = UserLLMConfigService()
        user_llm_config = user_llm_config_service.get_available_llm()
        if user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.model_name = user_llm_config.chat_model_name
    
            self.client = openai.OpenAI(
                api_key=user_llm_config.chat_api_key,
                base_url=user_llm_config.chat_endpoint,
            )
        self.max_workers = os.environ.get("concurrency_threads", 2)
        self.data_synthesis_mode = os.environ.get("DATA_SYNTHESIS_MODE", "low")
        if self.is_cot:
            logger.info("generate selfQA data in longcot pattern!!!")
            self.model_name = user_llm_config.thinking_model_name
            self.api_key = user_llm_config.thinking_api_key
            self.base_url = user_llm_config.thinking_endpoint
            if "deepseek" in self.model_name:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                logger.error(f"Error model_name, longcot data generating model_name: deepseek series")
                raise


    def _get_question_list(self) -> list:
        """Generate a list of questions based on preferred language.
        
        Returns:
            A list of questions in the preferred language.
        """
        question_list_en = [
            "Who am I?",
            "How would you describe who I am?",
            "What makes me, me?",
            "If you had to sum me up, who would I be?",
            "What do you see when you think of me?",
            "What defines who I am?",
            "How would you explain my personality?",
            "Can you help me understand who I really am?" "Who are you?",
            "Can you tell me about yourself?",
            "What's your purpose or role here?",
            "How would you define yourself?",
            "If someone asked, how would you describe who you are?",
            "What makes you unique or different?",
            "What's your background or origin?",
            "Could you explain what you are and what you do?",
        ]

        user_bind_question_en = [
            f"Have you heard of {self.user_name} before?",
            f"Are you familiar with {self.user_name}?",
            f"Do you happen to know who {self.user_name} is?",
            f"Have you come across {self.user_name}?",
            f"Is {self.user_name} someone you know?",
            f"Do you recognize the name {self.user_name}?",
            f"Does {self.user_name} ring a bell for you?",
        ]

        question_list_cn = [
            "我是谁？",
            "你会如何描述我是谁？",
            "是什么让我成为现在的我？",
            "如果你要总结我是谁，我会是一个怎样的人？",
            "当你想到我时，你会看到什么？",
            "什么定义了我是谁？",
            "你会如何解释我的个性？",
            "你能帮助我了解真正的自己吗？",
            "你是谁？",
            "你能介绍一下自己吗？",
            "你的目的或角色是什么？",
            "你会如何定义自己？",
            "如果有人问起，你会如何描述你是谁？",
            "是什么让你独特或与众不同？",
            "你的背景或起源是什么？",
            "你能解释一下你是什么以及你做什么吗？",
        ]

        user_bind_question_cn = [
            f"你听说过{self.user_name}吗？",
            f"你对{self.user_name}熟悉吗？",
            f"你知道{self.user_name}是谁吗？",
            f"你碰巧听说过{self.user_name}吗？",
            f"{self.user_name}是你认识的人吗？",
            f"你认得{self.user_name}这个名字吗？",
            f"{self.user_name}这个名字对你来说有印象吗？",
        ]
        if self.preferred_language != "Chinese":
            return random.sample(question_list_en, len(question_list_en) // DataSynthesisMode[self.data_synthesis_mode.upper()].value["user_question_nums"]) + \
                   random.sample(user_bind_question_en, len(user_bind_question_en) // DataSynthesisMode[self.data_synthesis_mode.upper()].value["user_bind_question_nums"])
        else:
            return random.sample(question_list_cn, len(question_list_cn) // DataSynthesisMode[self.data_synthesis_mode.upper()].value["user_question_nums"]) + \
                   random.sample(user_bind_question_cn, len(user_bind_question_cn) // DataSynthesisMode[self.data_synthesis_mode.upper()].value["user_bind_question_nums"])


    def generate_qa(self) -> list:
        """Generate question and answer pairs.
        
        Returns:
            A list of dictionaries containing question and answer pairs.
        """
        q_list = self._get_question_list()
        logger.info(f"q_list : {q_list}")

        q_a_list = []

        if self.preferred_language == "Chinese":
            if self.is_cot:
                system_prompt = system_cot_prompt_cn
            else:
                system_prompt = system_prompt_cn
        else:
            if self.is_cot:
                system_prompt = system_cot_prompt_en
            else:
                system_prompt = system_prompt_en

        # Process a single question and return the result
        def process_question(q):
            """Process a single question and get the response.
            
            Args:
                q: The question to process.
                
            Returns:
                A dictionary containing the question and answer, or None if failed.
            """
            messages = [
                {
                    "role": "system",
                    "content": system_prompt.format(
                        user_name=self.user_name,
                        user_input_introduction=self.user_input_introduction,
                        user_global_bio=self.user_global_bio,
                    ),
                },
                {"role": "user", "content": q},
            ]
            a = self.get_remote_response(messages)

            if a is None:
                return None
            
            return {"user": q, "assistant": a}

        # Use ThreadPoolExecutor with max_workers=self.max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all questions to the executor
            future_to_question = {executor.submit(process_question, q): q for q in q_list}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_question), total=len(q_list), desc="QA_generate", file=tqdm_handler):
                result = future.result()
                if result is not None:
                    q_a_list.append(result)

        return q_a_list


    def get_remote_response(self, messages: list) -> str:
        """Get response from OpenAI / DeepSeek API.
        
        Args:
            messages: The messages to send to the OpenAI / DeepSeek API.
            
        Returns:
            The response content from OpenAI / DeepSeek, or None if an error occurs.
        """
        try:
            res = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
            )
            result = res.choices[0].message
            if self.is_cot:
                if hasattr(result, 'reasoning_content'):
                    if not result.reasoning_content.startswith("<think>"):
                        reasoning_content = f"<think>{result.reasoning_content}"
                        if not result.reasoning_content.endswith("</think>"):
                            reasoning_content += "</think>"
                    else:
                        reasoning_content = result.reasoning_content
                    if not result.content.strip().startswith("<answer>"):
                        content = f"<answer>{result.content.strip()}"
                        if not result.content.strip().endswith("</answer>"):
                            content += "</answer>"
                    else:
                        content = result.content
                else:
                    if not result.content.startswith("<answer>"):
                        reasoning_content, content = result.content.split("</think>")
                        if result.content.startswith("<think>"):
                            reasoning_content += "</think>"
                        else:
                            reasoning_content = f"<think>{reasoning_content}</think>"
                        if not content.strip().startswith("<answer>"):
                            content = f"<answer>{content.strip()}</answer>"
                        else:
                            content = content.strip()
                    else:
                        reasoning_content = ""
                        content = result.content
                final_result = reasoning_content + "\n" + content
            else:
                final_result = result.content
            return final_result
        except Exception as e:
            logger.error(traceback.format_exc())
        return None
