import argparse
import concurrent.futures
import json
import logging
import os
import random
from typing import List

from openai import OpenAI
from tqdm import tqdm

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.base.database_operate import get_latest_global_bio
from lpm_kernel.base.stage3_prompt import QUESTION_WITHOUT_NOTES_USER_PROMPT_ZH, ANSWER_WITHOUT_NOTES_SYSTEM_PROMPT_ZH
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('synthetic_data_generation.log')
    ]
)


class NoNoteDataGenerator:
    def __init__(self, language: str):
        self.language = language

        self.bio = get_latest_global_bio().content_third_view

        self.bio = self.bio.replace("用户", "你")

        self.scenarios = {
            "决策": 0.2,
            "解释": 0.2,
            "习惯/流程": 0.2,
            "偏好比较": 0.2,
            "假设推演": 0.2,
            "故障排查": 0.1,
            "总结复述": 0.1,
            "建议指导": 0.2,
            "情感支撑": 0.4
        }
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

    def extract_questions_from_response(self, generated_text: str) -> List[str]:
        """
        Extract questions from the generated JSON response.

        Args:
            generated_text (str): The raw JSON response from the LLM

        Returns:
            List[str]: List of extracted questions
        """
        questions = []
        try:
            # Parse the JSON response
            content_json = json.loads(generated_text)

            # Extract questions from the JSON
            for key, value in content_json.items():
                if key.startswith("question_") and value:
                    questions.append(value)

            return questions
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from response: {generated_text[:100]}...")
            return []
        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return []

    def _generate_question_worker(self, _):
        scenario = random.choices(list(self.scenarios.keys()), list(self.scenarios.values()))[0]
        prompt = QUESTION_WITHOUT_NOTES_USER_PROMPT_ZH.format(scenario=scenario)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.7
            # extra_body={"metadata": {"tags": "lpm-pipeline-using"}},
        )
        generated_text = response.choices[0].message.content
        questions = self.extract_questions_from_response(generated_text)

        results = []
        for i, question in enumerate(questions):
            result = {
                "scenario": scenario,
                "generated_questions": question,
                "raw_response": generated_text
            }
            results.append(result)
            logging.info(f"Successfully added question {i + 1} for scenario: {scenario}")
            break

        return results

    def gen_no_notes_questions(self, length, max_workers=4,
                               output_path="resources/data/stage3/synthetic_data_no_notes_questions.json"):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._generate_question_worker, i) for i in range(length)]

            for future in tqdm(concurrent.futures.as_completed(futures), total=length,
                               desc="Generating questions without notes"):
                results.extend(future.result())

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        return results

    def _generate_answer_worker(self, question):
        try:
            prompt = ANSWER_WITHOUT_NOTES_SYSTEM_PROMPT_ZH.format(bio=self.bio)
            response = self.reasoning_client.chat.completions.create(
                model=self.reasoning_model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question["generated_questions"]}
                ],
                max_tokens=5000,
                temperature=0.7
            )
            result = response.choices[0].message

            answer = "<think>" + result.reasoning_content + "</think>\n\n<answer>" + result.content + "</answer>"
            question_copy = question.copy()
            question_copy["answer"] = answer

            # Print to console for monitoring
            logger.info("<think>" + result.reasoning_content + "</think>")
            logger.info("<answer>" + result.content + "</answer>")
            logger.info("")

            return question_copy
        except Exception as e:
            logging.error(f"Error generating answer for question '{question['generated_questions']}': {str(e)}")
            question_copy = question.copy()
            question_copy["answer"] = f"ERROR: {str(e)}"
            return question_copy

    def gen_no_notes_answers(self, questions, max_workers=5,
                             output_path="resources/data/stage3/synthetic_data_no_notes_answers.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_questions = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._generate_answer_worker, question) for question in questions]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(questions),
                               desc="Generating answers without notes"):
                try:
                    result = future.result()
                    if result:
                        processed_questions.append(result)
                except Exception as e:
                    logging.error(f"Exception in thread: {str(e)}")

        # Save all successfully processed questions
        if processed_questions:
            with open(output_path, "w") as f:
                json.dump(processed_questions, f, ensure_ascii=False, indent=4)

        return processed_questions


if __name__ == "__main__":
    with open(
            "resources/data/stage3/synthetic_data_with_notes_answers.json",
            "r") as f:
        with_notes_data = json.load(f)

    len_of_no_note_data = len(with_notes_data) // 8

    generator = NoNoteDataGenerator(
        language='zh'
    )

    # 生成问题
    logger.info(f"开始生成 {len_of_no_note_data} 个无笔记问题...")
    questions = generator.gen_no_notes_questions(
        length=len_of_no_note_data,
        max_workers=10
    )
    logger.info(f"成功生成 {len(questions)} 个问题")

    # 生成回答
    logger.info("开始为问题生成回答...")
    answers = generator.gen_no_notes_answers(
        questions=questions,
        max_workers=10
    )
    logger.info(f"成功生成 {len(answers)} 个回答")
    logger.info("数据生成完成!")
