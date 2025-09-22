from itertools import islice
import concurrent.futures
import json
import os
import random
import re
from tqdm import tqdm
import openai
from enum import Enum
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from lpm_kernel.L2.data_pipeline.data_prep.preference.prompts import (
    CH_USR_TEMPLATES, CH_USR_COT_TEMPLATES,
    EN_USR_TEMPLATES, EN_USR_COT_TEMPLATES,
    CH_SYS_TEMPLATES, CH_SYS_COT_TEMPLATES,
    EN_SYS_TEMPLATES, EN_SYS_COT_TEMPLATES
)
import traceback
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


class LowMode(Enum):
    cluster_nums = 3


class MediumMode(Enum):
    cluster_nums = 2


class HighMode(Enum):
    cluster_nums = 1


class PreferenceQAGenerator:
    def __init__(self, filename: str, bio: str, preference_language: str, is_cot: bool = True):
        """Initialize the PreferenceQAGenerator class.
        
        Args:
            filename: Path to the input JSON file containing preference messages.
            bio: Biography or context information to use in prompt generation.
            preference_language: Language for prompts ("Chinese/中文" or otherwise English).
        """
        # Ensure the filename is actually a string
        if filename is None:
            raise ValueError("Filename cannot be None")
            
        self.filename = filename
        # Convert is_cot to bool if it's a string
        if isinstance(is_cot, str):
            self.is_cot = is_cot.lower() == 'true'
        else:
            self.is_cot = bool(is_cot)
            
        logger.info(f"PreferenceQAGenerator initialized with is_cot={self.is_cot}")
        
        with open(self.filename, "r", encoding="utf-8") as f:
            self.pre_msg = json.load(f)

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
        if self.is_cot:
            logger.info("generate pereference data in longcot pattern!!!")
            self.model_name = user_llm_config.thinking_model_name
            self.api_key = user_llm_config.thinking_api_key
            self.base_url = user_llm_config.thinking_endpoint
            if "deepseek" in self.model_name:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                logger.error(f"Error model_name, longcot data generating model_name: deepseek series")
                raise
            
        
        self.bio = bio
        self.question_list = []
        self.preference_language = preference_language
        self.prompt_templates = self._get_prompt_templates(preference_language)
        self.sys_templates = self._get_sys_templates(preference_language)
        self.max_workers = 1
        self.data_synthesis_mode = os.environ.get("DATA_SYNTHESIS_MODE", "low")


    def generate_response(self, sys: str, prompt: str, mode: str) -> str:
        """Generate a response using the OpenAI / DeepSeek API.
        
        Args:
            sys: The system prompt to use.
            prompt: The user prompt to send to the API.
            
        Returns:
            The generated response text or None if an error occurred.
        """
        def get_remote_response(sys: str, prompt: str, mode: str) -> str:
            """Get response from OpenAI / DeepSeek API.
            
            Args:
                sys: The system prompt to use.
                prompt: The user prompt to send to the API.
                
            Returns:
                The response content from OpenAI / DeepSeek, or None if an error occurs.
            """
            try:
                res = self.client.chat.completions.create(
                    messages=[
                            {"role": "system", "content": sys},
                            {"role": "user", "content": prompt},
                        ],
                    model=self.model_name,
                )
                result = res.choices[0].message
                if self.is_cot and mode == "answer":
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
        
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future = executor.submit(get_remote_response, sys, prompt, mode)
                response = future.result()
                return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None


    def clean_chunk(self, chunk: str) -> str:
        """Clean and process a text chunk.
        
        Args:
            chunk: The text chunk to clean.
            
        Returns:
            Cleaned text after processing.
        """
        after = chunk.split(":")[1:]
        after = ":".join(after)
        return after[1:]


    def _get_prompt_templates(self, preference_language: str) -> dict:
        """Return a dictionary of prompt templates based on language preference (w or w/o cot).
        
        Args:
            preference_language: The language preference ("Chinese/中文" or otherwise English).
            
        Returns:
            A dictionary of prompt templates.
        """
        if preference_language == "Chinese":
            if self.is_cot:
                return CH_USR_COT_TEMPLATES
            else:
                return CH_USR_TEMPLATES
        else:
            if self.is_cot:
                return EN_USR_COT_TEMPLATES
            else:
                return EN_USR_TEMPLATES


    def _get_sys_templates(self, preference_language: str) -> dict:
        """Return a dictionary of system templates based on language preference (w or w/o cot).
        
        Args:
            preference_language: The language preference ("Chinese/中文" or otherwise English).
            
        Returns:
            A dictionary of system templates.
        """
        if preference_language == "Chinese":
            if self.is_cot:
                return CH_SYS_COT_TEMPLATES
            else:
                return CH_SYS_TEMPLATES
        else:
            if self.is_cot:
                return EN_SYS_COT_TEMPLATES
            else:
                return EN_SYS_TEMPLATES


    def process_clusters(self, output_filename: str) -> None:
        """Process clusters and generate questions and answers.
        
        Args:
            output_filename: Path to save the generated Q&A pairs.
        """
        cluster_items = list(self.pre_msg.items())
        count = 0
        
        if self.data_synthesis_mode == "low":
            sample_num = max(1, len(cluster_items) // LowMode.cluster_nums.value) if 0 < len(cluster_items) < 3 else len(cluster_items) // LowMode.cluster_nums.value
            new_cluster_items = random.sample(cluster_items, sample_num)
        elif self.data_synthesis_mode == "medium":
            sample_num = max(1, len(cluster_items) // MediumMode.cluster_nums.value) if 0 < len(cluster_items) < 2 else len(cluster_items) // MediumMode.cluster_nums.value
            new_cluster_items = random.sample(cluster_items, sample_num)
        else: # high or other case
            new_cluster_items = cluster_items
            
        for _, cluster in tqdm(new_cluster_items, desc="preference_generate", file=tqdm_handler):
            chunk_concat = self._get_chunk_concat(cluster["contents"])

            tags = " ".join(cluster["tags"])

            if len(chunk_concat) < 20:
                continue
            count += 1
            
            n_cluster = len(cluster["contents"])
            if n_cluster > 1:
                logger.info(f"Cluster has {str(n_cluster)} chunks")

            prompt_question_template = self.prompt_templates["query"]
            prompt_answer_template = self.prompt_templates["answer"]
            sys_question = self.sys_templates["query"]
            sys_answer = self.sys_templates["answer"]

            try:
                gen_question = self.generate_response(
                    sys_question,
                    prompt_question_template.format(
                        bio=self.bio, chunks_concat=chunk_concat
                    ),
                    mode="question"
                )
                if self.is_cot:
                    question_match = re.search(r"<question>(.*?)</question>", gen_question, re.DOTALL)
                    gen_question = question_match.group(1).strip() if question_match else gen_question
                    if "<think>" in gen_question:
                        gen_question = gen_question.split("</think>")[1].strip()
            except Exception as e:
                logger.error(traceback.format_exc())
                continue
            try:
                gen_answer = self.generate_response(
                    sys_answer,
                    prompt_answer_template.format(
                        question=gen_question, bio=self.bio, chunks_concat=chunk_concat
                    ),
                    mode="answer"
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                continue
            
            self.question_list.append({"user": gen_question, "assistant": gen_answer})
            if n_cluster >= 20:
                self._generate_multiple_questions(cluster["contents"], chunk_concat)
            if count % 5 == 0:
                logger.info(f"Processed {count} clusters")

        with open(output_filename, "w") as json_file:
            json.dump(self.question_list, json_file, indent=4, ensure_ascii=False)


    def _get_chunk_concat(self, contents: list) -> str:
        """Concatenate and clean chunks of text.
        
        Args:
            contents: List of content chunks to concatenate.
            
        Returns:
            Concatenated text with formatting.
        """
        chunk_concat = ""
        for content in contents:
            chunk_content = content
            chunk_concat += chunk_content
            chunk_concat += "\n\n"
        return chunk_concat


    def _generate_multiple_questions(self, contents: list, chunk_concat: str) -> None:
        """Generate multiple questions and answers for larger clusters.
        
        Args:
            contents: List of content chunks.
            chunk_concat: Concatenated text chunks.
        """
        num_chunk_referred = 30
        n_repeat = max(1, int(len(contents) * 1 / num_chunk_referred))
        chunk_content_list = [
            self.clean_chunk(content)
            for content in contents
            if len(self.clean_chunk(content)) >= 80
        ]

        logger.info(f"Big cluster: n_repeat = {n_repeat}")

        for i in range(n_repeat):
            if i % 5 == 0 and i > 0:
                logger.info(f"Repeat {i} times")
            selected_chunks = random.sample(
                chunk_content_list, min(len(chunk_content_list), num_chunk_referred)
            )
            chunk_concat = "\n".join(selected_chunks)
            prompt_question_template = self.prompt_templates["query"]
            prompt_answer_template = self.prompt_templates["answer"]
            sys_question = self.sys_templates["query"]
            sys_answer = self.sys_templates["answer"]

            try:
                gen_question = self.generate_response(
                    sys_question,
                    prompt_question_template.format(
                        bio=self.bio, chunks_concat=chunk_concat
                    ),
                    mode="question"
                )
                if self.is_cot:
                    question_match = re.search(r"<question>(.*?)</question>", gen_question, re.DOTALL)
                    gen_question = question_match.group(1).strip() if question_match else gen_question
                gen_answer = self.generate_response(
                    sys_answer,
                    prompt_answer_template.format(
                        question=gen_question, chunks_concat=chunk_concat, bio=self.bio
                    ),
                    mode="answer"
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                continue
            self.question_list.append({"user": gen_question, "assistant": gen_answer})
        return
