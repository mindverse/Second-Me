import concurrent.futures
import json
import logging
import os
import random
import traceback
from typing import List, Dict, Any, Tuple

from openai import OpenAI
from tqdm import tqdm

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.base.database_operate import get_latest_global_bio
from lpm_kernel.base.stage3_prompt import QUESTION_WITH_NOTES_SYSTEM_PROMPT_ZH, ANSWER_WITH_NOTES_SYSTEM_PROMPT_ZH
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.file_data import Document
from lpm_kernel.file_data.document_repository import DocumentRepository

logger = get_train_process_logger()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('synthetic_data_generation.log')
    ]
)


class SyntheticDataGenerator:
    def __init__(self, language: str = "简体中文"):
        """
        Initializes the SyntheticDataGenerator.

        Args:
            language (str): Language configuration, e.g., 'zh' for Chinese, 'en' for English.
            model_name (str): The name of the language model.
            model_address (str): The API endpoint or address of the language model.
            api_key (str, optional): The API key for accessing the model. Defaults to None,
                                     can also be fetched from environment variables.
        """
        self.language = language

        self.bio = get_latest_global_bio().content_third_view

        with open(
                "resources/data/stage2/wiki/wiki_res.json",
                "r") as f:
            self.subjective_wiki_res = json.load(f)

        doc_repository = DocumentRepository()
        self.subjective_notes = doc_repository.list()

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

        self.sample_time = 5

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

    def __repr__(self):
        return f"SyntheticDataGenerator(language='{self.language}', model_name='{self.model_name}')"

    def get_note_by_id(self, note_id: int):
        """
        Get note by ID from objective or subjective notes.

        Args:
            note_id (int): The ID of the note to find
            note_type (str): Either 'objective' or 'subjective'

        Returns:
            Dict[str, Any]: The found note or empty dict if not found
        """
        try:
            notes = self.subjective_notes
            for note in notes:
                if note.id == note_id:
                    return note
            logger.warning(f"Note with ID {note_id} not found in notes")
            return {}
        except Exception as e:
            logger.error(f"Error getting note by ID: {e}")
            return {}

    def format_note(self, note: Document) -> str:
        """
        Format a note for use in prompts.

        Args:
            note (Document): The note to format

        Returns:
            str: The formatted note
        """
        try:
            if not note:
                return ""

            title = note.title
            insight = note.insight
            return f"【你之前浏览过的资料】{title}: {insight}"

        except Exception as e:
            logger.error(f"Error formatting note: {e}")
            return ""

    def process_wikis_and_notes(self) -> List[Tuple[str, List[str]]]:
        """
        Process all wikis and find related notes.

        Returns:
            List[Tuple[str, List[str]]]: List of tuples containing (wiki description, list of formatted notes)
        """
        result = []

        try:
            # Process subjective wikis
            for wiki in self.subjective_wiki_res:
                wiki_description = wiki.get('description', '')
                related_note_ids = wiki.get('related_notes', [])
                formatted_notes = []

                for note_id in related_note_ids:
                    note = self.get_note_by_id(note_id)
                    formatted_note = self.format_note(note)
                    if formatted_note:
                        formatted_notes.append(formatted_note)

                if wiki_description and formatted_notes:
                    result.append((wiki_description, formatted_notes))

            logger.info(f"Processed {len(result)} wikis with related notes")
            return result

        except Exception as e:
            logger.error(f"Error processing wikis and notes: {e}")
            return []

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

    def select_scenario(self) -> List[str]:
        """
        Select scenarios based on configured probabilities.

        Returns:
            List[str]: The selected scenarios
        """
        try:
            scenarios = list(self.scenarios.keys())
            probabilities = list(self.scenarios.values())
            selected_scenarios = random.choices(scenarios, weights=probabilities, k=self.sample_time)
            logger.info(f"Selected scenarios: {selected_scenarios}")
            return selected_scenarios
        except Exception as e:
            logger.error(f"Error selecting scenarios: {e}")
            default_scenario = list(self.scenarios.keys())[0]
            logger.info(f"Using default scenario: {default_scenario}")
            return [default_scenario]

    def generate_without_user_prompt(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generates synthetic data based on the provided prompt using the LLM API.

        Args:
            prompt (str): The prompt to send to the language model.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The sampling temperature.

        Returns:
            str: The generated text.
        """
        try:
            logger.info(f"Generating data with prompt: '{prompt[:100]}...' using {self.model_name}")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=temperature
                # extra_body={"metadata": {"tags": "lpm-pipeline-using"}},
            )

            generated_text = response.choices[0].message.content

            logger.info(f"Successfully generated data ({len(generated_text)} chars)")
            return generated_text

        except Exception as e:
            logger.error(f"Error generating data: {e}")
            return f"Error generating data: {str(e)}"

    def generate_with_user_prompt(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000,
                                  temperature: float = 0.7) -> str:
        """
        Generates synthetic data based on the provided prompt using the LLM API.

        Args:
            prompt (str): The prompt to send to the language model.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The sampling temperature.

        Returns:
            str: The generated text.
        """
        try:
            logger.info(f"Generating data with prompt: '{system_prompt[:100]}...' using {self.reasoning_model_name}")

            response = self.reasoning_client.chat.completions.create(
                model=self.reasoning_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
                # extra_body={"metadata": {"tags": "lpm-pipeline-using"}},
            )

            result = response.choices[0].message
            logger.info("<think>" + result.reasoning_content + "</think>")
            logger.info("<answer>" + result.content + "</answer>")
            logger.info("")
            return "<think>" + result.reasoning_content + "</think>\n\n<answer>" + result.content + "</answer>"
        except Exception as e:
            logger.error(f"Error generating data: {e}")
            return f"Error generating data: {str(e)}"

    def process_wiki_note_pair(self, wiki_note_pair):
        """
        Process a single wiki-note pair with all selected scenarios.

        Args:
            wiki_note_pair (Tuple): A tuple containing (wiki_desc, formatted_notes)

        Returns:
            List[Dict]: List of generated results
        """
        wiki_desc, formatted_notes = wiki_note_pair
        results = []

        # Select scenarios
        scenarios = self.select_scenario()

        for scenario in scenarios:
            try:
                # Concatenate notes
                notes_content = "\n\n".join(formatted_notes)

                prompt = QUESTION_WITH_NOTES_SYSTEM_PROMPT_ZH.format(
                    scenario=scenario,
                    notes_content=notes_content)

                # Generate content using LLM
                generated_content = self.generate_without_user_prompt(prompt)

                # Parse and validate the generated content
                try:
                    # Try to parse as JSON
                    content_json = json.loads(generated_content)

                    # Extract questions from JSON
                    questions = self.extract_questions_from_response(generated_content)[:1]

                    # Create separate result for each question
                    for i, question in enumerate(questions):
                        result = {
                            "wiki_description": wiki_desc,
                            "notes": formatted_notes,
                            "scenario": scenario,
                            "generated_questions": question,
                            "raw_response": generated_content
                        }
                        results.append(result)
                        logger.info(f"Successfully added question {i + 1} for scenario: {scenario}")

                except json.JSONDecodeError:
                    # If not valid JSON, save as raw text
                    result = {
                        "wiki_description": wiki_desc,
                        "notes": formatted_notes,
                        "scenario": scenario,
                        "generated_questions": None,
                        "raw_response": generated_content
                    }
                    results.append(result)
                    logger.warning(f"Generated content is not valid JSON: {generated_content[:100]}...")

            except Exception as e:
                logger.error(f"Error processing wiki-note pair with scenario {scenario}: {e}")
                logger.error(traceback.format_exc())
                continue

        return results

    def generate_synthetic_data(self, output_file: str) -> list[Any] | None:
        """
        Generate synthetic data by processing wikis, selecting scenarios, and generating content.
        Uses parallel processing with max_workers=5.

        Args:
            output_file (str): Path to save the generated data

        Returns:
            bool: Whether the generation was successful
        """
        try:
            # Process wikis and get related notes
            wiki_note_pairs = self.process_wikis_and_notes()

            if not wiki_note_pairs:
                logger.error("No valid wiki-note pairs found for generation")
                return None

            results = []
            # Process wiki-note pairs in parallel

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_pair = {executor.submit(self.process_wiki_note_pair, pair): pair for pair in wiki_note_pairs}

                for future in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(wiki_note_pairs),
                                   desc="Processing wiki-note pairs"):
                    pair = future_to_pair[future]
                    try:
                        pair_results = future.result()
                        results.extend(pair_results)
                    except Exception as e:
                        logger.error(f"Error processing wiki-note pair {pair[0][:50]}...: {e}")

            # Save results to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully generated {len(results)} synthetic data items and saved to {output_file}")
            return results

        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return None

    def generate_notes_answers(self, generated_questions_list: List[Dict], output_file: str) -> bool:
        """
        Generate synthetic data by processing wikis, selecting scenarios, and generating content.
        Uses parallel processing with max_workers=5.
        """

        def process_question(question):
            notes_content = question["notes"]
            system_prompt = ANSWER_WITH_NOTES_SYSTEM_PROMPT_ZH.format(
                notes_content=notes_content,
                bio=self.bio
            )
            user_prompt = question["generated_questions"]
            if user_prompt is None:
                return question
            answer = self.generate_with_user_prompt(system_prompt, user_prompt)
            question["answer"] = answer
            return question

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks and track them with tqdm
            future_to_question = {executor.submit(process_question, question): question for question in
                                  generated_questions_list}
            processed_questions = []

            for future in tqdm(concurrent.futures.as_completed(future_to_question), total=len(generated_questions_list),
                               desc="Generating answers"):
                try:
                    result = future.result()
                    processed_questions.append(result)
                except Exception as e:
                    question = future_to_question[future]
                    logger.error(f"Error processing question: {e}")
                    # Add the original question without answer if processing fails
                    processed_questions.append(question)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_questions, f, ensure_ascii=False, indent=2)

        return True


if __name__ == '__main__':
    # Example Usage (assuming a local model or API key is set as an environment variable)
    try:

        generator = SyntheticDataGenerator(
            language='zh',
        )

        os.makedirs("resources/data/stage3", exist_ok=True)
        synthetic_data_output_path = "resources/data/stage3/synthetic_data_with_notes_questions.json"

        generated_json = generator.generate_synthetic_data(output_file=synthetic_data_output_path)

        notes_answer_path = "resources/data/stage3/synthetic_data_with_notes_answers.json"

        success = generator.generate_notes_answers(generated_json, output_file=notes_answer_path)

        if success:
            logger.info("Synthetic data generation completed successfully!")
        else:
            logger.info("Synthetic data generation failed. Check logs for details.")

    except ValueError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(
            "Please make sure you have set the OPENAI_API_KEY environment variable")
        logger.error("or provide the api_key directly when instantiating the SyntheticDataGenerator.")
        logger.error(traceback.format_exc())
