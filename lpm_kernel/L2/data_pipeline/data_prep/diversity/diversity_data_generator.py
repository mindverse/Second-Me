from concurrent.futures import ThreadPoolExecutor
import json
import os
import logging
import random
import re
import traceback

import openai
import pandas as pd
from tqdm import tqdm
from enum import Enum
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from lpm_kernel.L2.data_pipeline.data_prep.diversity.utils import remove_similar_dicts
import lpm_kernel.L2.data_pipeline.data_prep.diversity.template_diversity as template_diversity

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class DataSynthesisMode(Enum):
    LOW = {"large_aug_para":1, "tiny_aug_para":1, "mini_aug_para":1}
    MEDIUM = {"large_aug_para":2, "tiny_aug_para":2, "mini_aug_para":2}
    HIGH = {"large_aug_para":4, "tiny_aug_para":3, "mini_aug_para":2}


class TqdmLoggingHandler:
    def __init__(self):
        pass
    
    def write(self, msg):
        logger.info(msg.strip())
    
    def flush(self):
        pass
    
tqdm_handler = TqdmLoggingHandler()


class DiversityDataGenerator:
    """Generates diversity data for training language models.
    
    This class is responsible for creating diverse training data based on user notes,
    entities, and configurations. It leverages LLMs to generate questions and answers.
    """
    
    def __init__(self, preference_language: str, is_cot: bool = True):
        """Initialize the diversity data generator.
        
        Args:
            preference_language: The language to use for generating data.
        """
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
        self.preference_language = preference_language
        self.max_workers = os.environ.get("concurrency_threads", 2)
        self.data_synthesis_mode = os.environ.get("DATA_SYNTHESIS_MODE", "low")
        self.is_cot = is_cot
        if self.is_cot:
            logger.info("generate diversity data in longcot pattern!!!")
            self.model_name = user_llm_config.thinking_model_name
            self.api_key = user_llm_config.thinking_api_key
            self.base_url = user_llm_config.thinking_endpoint
            if "deepseek" in self.model_name:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                logger.error(f"Error model_name, longcot data generating model_name: deepseek series")
                raise


    def _preprocess(self, entities_path: str, note_list: list, config_path: str, graph_path: str, user_name: str):
        """Preprocess the input data for diversity generation.
        
        Args:
            entities_path: Path to entities data file.
            note_list: List of note objects.
            config_path: Path to configuration file.
            graph_path: Path to graph data file.
            user_name: Name of the user.
            
        Returns:
            Tuple containing entity descriptions, entity types, and QA configuration.
        """
        entity_df = pd.read_parquet(graph_path)
        entity2type = {
            item["title"]: item["type"] for item in entity_df.to_dict(orient="records")
        }

        # read entity2desc
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                entities = json.load(f)
                entity2desc = {
                    item["entity_name"]: {
                        key: value for key, value in item.items() if key != "entity_name"
                    }
                    for item in entities
                }
        except Exception as e:
            return None, None, None
        
        # read note data
        id2note = {
            item.id: {
                key: value for key, value in item.to_json().items() if key != "id"
            }
            for item in note_list
        }

        for entity, entity_info in entity2desc.copy().items():
            doc_ids = entity_info["doc_id"]
            tmp = []
            for doc_id in doc_ids:
                if isinstance(doc_id, str):
                    continue
                else:
                    note_desc = id2note.get(doc_id, "")
                    if note_desc:
                        tmp.append(note_desc)
            entity2desc[entity]["note"] = tmp

        entity2desc.pop(f"{user_name}", None)
        entity2desc.pop(f"{user_name.upper()}", None)

        # exclude keys with time format
        time_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        filtered_data = {
            k: v for k, v in entity2desc.items() if not re.match(time_pattern, k)
        }
        entity2desc = filtered_data

        # clean note level data
        for entity, entity_info in entity2desc.copy().items():
            clusters = entity_info["note"]
            unique_dicts, cnt = remove_similar_dicts(clusters, similarity_threshold=0.9)
            entity2desc[entity]["note"] = unique_dicts

        # read config file
        with open(config_path, "r", encoding="utf-8") as f:
            QA_config = json.load(f)

        return entity2desc, entity2type, QA_config


    def _get_A_input(self, cluster: dict, question: str, user_name: str) -> str:
        """Generate the input for answer generation.
        
        Args:
            cluster: The data cluster containing entity information.
            question: The question to be answered.
            user_name: Name of the user.
            
        Returns:
            A string containing the formatted input for the answer generation model.
        """
        entity = cluster["entity_name"]
        entity_desc = cluster["entity_description"]
        entity_desc = f"Entity'{entity}',Relevant Info：'{entity_desc}'"

        tmpl = f"""I am {user_name}. Regarding {entity_desc}, here is some information I previously mentioned:\n\n"""

        chunk_tmpl = ""
        for ind, entity_dict in enumerate(cluster["note"]):
            if "processed" in entity_dict:
                content = entity_dict["processed"]
            else:
                content = entity_dict["content"]
                title = entity_dict["title"]
                insight = entity_dict["insight"]
                content = f"Title: {title}\nContent: {content}\nAI Insight: {insight}"

            tmp = f"___________________\n{content}\n"
            chunk_tmpl += tmp

        tmpl = (
            tmpl
            + chunk_tmpl
            + f"Based on the information I have previously recorded, please answer '{question}'. Note that you need to ensure the perspective is consistent, meaning that all instances of {user_name} should be replaced with the second person 'you'."
        )

        return tmpl


    def _get_Q_input(self, cluster: dict, user_name: str) -> str:
        """Generate the input for question generation.
        
        Args:
            cluster: The data cluster containing entity information.
            user_name: Name of the user.
            
        Returns:
            A string containing the formatted input for the question generation model.
        """
        entity = cluster["entity_name"]
        entity_desc = cluster["entity_description"]
        entity_desc = f"Entity'{entity}'：{entity_desc}"
        tmpl = f""""For {entity_desc}, here is the relevant content from my interactions with the AI robot:\n"""
        chunk_tmpl = ""
        for ind, entity_dict in enumerate(cluster["note"]):
            content = entity_dict["content"]
            title = entity_dict["title"]
            insight = entity_dict["insight"]
            content = f"Title: {title}\nContent: {content}\nAI Insight: {insight}"

            tmp = f"# Content {ind+1} #\n{content}\n"
            chunk_tmpl += tmp
        tmpl = (
            tmpl
            + chunk_tmpl
            + f"Please help me generate questions; note that you need to phrase them from my perspective, meaning all expressions of {user_name} should be replaced with the first person 'I'."
        )

        return tmpl


    def generate_data(self, entities_path: str, note_list: list, config_path: str, 
                     graph_path: str, user_name: str, global_bio: str, output_path: str):
        """Generate diversity data based on user notes and entities.
        
        Args:
            entities_path: Path to entities data file.
            note_list: List of note objects.
            config_path: Path to configuration file.
            graph_path: Path to graph data file.
            user_name: Name of the user.
            global_bio: User biography text.
            output_path: Path to save the generated data.
        """
        language_desc = f"Keep your response in {self.preference_language}"

        entity2desc, entity2type, QA_config = self._preprocess(
            entities_path, note_list, config_path, graph_path, user_name
        )

        if entity2desc is None:
            return 

        tmp = QA_config["query"]

        q_dict = {item["type"]: {k: item[k] for k in item if k != "type"} for item in tmp}

        tmp = QA_config["answer"]
        a_dict = {item["type"]: {k: item[k] for k in item if k != "type"} for item in tmp}

        templater = template_diversity.templater(
            q_dict, a_dict, user_name, global_bio, self.is_cot
        )

        entity2desc_list = [{**{"entity_name": k}, **v} for k, v in entity2desc.items()]

        # global questions, only process clusters with more than 8 notes, and split very large clusters
        large_clusters = [item for item in entity2desc_list if len(item["note"]) >= 8]
        logger.info(f"Large clusters: {len(large_clusters)}")

        exploded_clusters = []
        # split
        for sub_dict in large_clusters:
            for i in range(0, len(sub_dict["note"]), 4):
                tmp_dict = sub_dict.copy()

                tmp_dict["note"] = sub_dict["note"][i : i + 4]
                tmp_dict["doc_id"] = sub_dict["doc_id"][i : i + 4]
                exploded_clusters.append(tmp_dict)

            # ensure global effect, add some large global data
            notes_and_ids = list(zip(sub_dict["note"], sub_dict["doc_id"]))
            for _ in range(len(sub_dict["note"]) // 10 + 1):
                tmp_dict = sub_dict.copy()
                sampled_notes_and_ids = random.sample(
                    notes_and_ids, min(10, len(notes_and_ids))
                )
                tmp_dict["note"], tmp_dict["doc_id"] = zip(
                    *sampled_notes_and_ids
                )  # Unpack into two lists
                exploded_clusters.append(tmp_dict)

        # process small clusters
        mini_clusters = [
            item
            for item in entity2desc_list
            if len(item["note"]) < 8 and len(item["note"]) > 1
        ]

        logger.info(f"Mini clusters: {len(mini_clusters)}")

        # process other clusters
        tiny_clusters = [item for item in entity2desc_list if len(item["note"]) <= 1]

        logger.info(f"Tiny clusters: {len(tiny_clusters)}")

        filtered_tiny_clusters = [
            d
            for d in tiny_clusters
            if entity2type.get(d["entity_name"], "")
            in ["PERSON", "人", "组织", "ORGANIZATION", "人物"]
        ]

        logger.info(f"Filtered tiny clusters: {len(filtered_tiny_clusters)}")

        if len(exploded_clusters) > 0:
            logger.info("Execute large cluster generation")
            data_large = self._pipline(exploded_clusters, DataSynthesisMode[self.data_synthesis_mode.upper()].value["large_aug_para"], 
                                       q_dict, templater, language_desc, user_name)
        else:
            logger.info("Large cluster number is 0")
            data_large = []

        if len(mini_clusters) > 0:
            logger.info("Execute small cluster generation")
            data_mini = self._pipline(mini_clusters, DataSynthesisMode[self.data_synthesis_mode.upper()].value["mini_aug_para"], 
                                      q_dict, templater, language_desc, user_name)
        else:
            logger.info("Small cluster number is 0")
            data_mini = []

        if len(filtered_tiny_clusters) > 0:
            logger.info("Execute single entity cluster generation")
            q_dict.pop("unanswerable")
            q_dict.pop("global")
            data_tiny = self._pipline(filtered_tiny_clusters, DataSynthesisMode[self.data_synthesis_mode.upper()].value["tiny_aug_para"], 
                                      q_dict, templater, language_desc, user_name)
        else:
            logger.info("Single entity cluster number is 0")
            data_tiny = []

        combined_list = data_large + data_mini + data_tiny
        # calculate total entries
        total_entries = len(combined_list)
        logger.info(f"Total entries: {total_entries}")
        # store data
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined_list, f, ensure_ascii=False, indent=4)

        logger.info(f"Data has been stored to {output_path}")


    def _pipline(self, clusters: list, aug_para: int, q_dict: dict, 
                templater, language_desc: str, user_name: str) -> list:
        """Execute the pipeline for data generation.
        
        Args:
            clusters: List of data clusters.
            aug_para: Data augmentation coefficient.
            q_dict: Dictionary of question types.
            templater: Template handler object.
            language_desc: Language description string.
            user_name: Name of the user.
            
        Returns:
            List of generated QA data.
        """
        explode_clusters = []
        explode_questions_types = []
        for item in clusters:
            # add elements multiple times based on aug_para
            explode_clusters.extend([item] * aug_para)
            # randomly select different types based on weights
            weights = [v["weight"] for v in q_dict.values()]
            random_types = random.choices(list(q_dict.keys()), weights, k=aug_para)
            explode_questions_types.extend(random_types)

        logger.info("Start generating data")
        logger.info(f"Explode clusters: {len(explode_clusters)}")
        logger.info(f"Explode questions types: {len(explode_questions_types)}")

        questions, answers, answer_types, flat_question_types, flat_clusters = self._generate(
            explode_clusters, explode_questions_types, templater, q_dict, language_desc, user_name
        )

        # store data
        data = []
        for cluster, question, answer, question_type, answer_type in zip(
            flat_clusters, questions, answers, flat_question_types, answer_types
        ):
            if len(question) == 0 or len(answer) == 0:
                continue
            data.append(
                {
                    "user": question,
                    "assistant": answer,
                    "entity_name": cluster["entity_name"],
                    "question_type": question_type,
                    "answer_type": answer_type,
                    "doc_id": cluster["doc_id"],
                }
            )
        return data


    def _generate(self, explode_clusters: list, explode_questions_types: list, 
                 templater, q_dict: dict, language_desc: str, user_name: str) -> tuple:
        """Generate questions and answers using ThreadPoolExecutor.
        
        Args:
            explode_clusters: List of expanded data clusters.
            explode_questions_types: List of question types to generate.
            templater: Template handler object.
            q_dict: Dictionary of question types.
            language_desc: Language description string.
            user_name: Name of the user.
            
        Returns:
            Tuple of (questions, answers, answer_types, flat_question_types, flat_clusters).
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._Q_generate, cluster, question_type, templater, q_dict, language_desc, user_name)
                for cluster, question_type in zip(
                    explode_clusters, explode_questions_types
                )
            ]
            questions = []
            flat_clusters = []
            flat_question_types = []
            cnt = 0
            for future, cluster, question_type in zip(
                tqdm(futures, total=len(futures), desc="Q_generate", file=tqdm_handler),
                explode_clusters,
                explode_questions_types,
            ):
                try:
                    result = future.result()
                    cnt += 1 if result else 0
                    # Assuming result is a list of questions
                    questions.extend(result)
                    # Extend clusters and question types to match the number of questions
                    flat_clusters.extend([cluster] * len(result))
                    flat_question_types.extend([question_type] * len(result))
                except Exception as e:
                    logger.error(traceback.format_exc())

        # safety check
        logger.info(f"Count: {cnt}, len(explode_clusters): {len(explode_clusters)}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._A_generate, cluster, question, question_type, templater, language_desc, user_name)
                for cluster, question, question_type in zip(
                    flat_clusters, questions, flat_question_types
                )
            ]

            answers = []
            answer_types = []

            for future in tqdm(futures, total=len(futures), desc="A_generate", file=tqdm_handler):
                try:
                    result, answer_type = future.result()
                    answers.append(result)
                    answer_types.append(answer_type)
                except Exception as e:
                    logger.error(traceback.format_exc())

        return questions, answers, answer_types, flat_question_types, flat_clusters


    def _Q_generate(self, cluster: dict, question_type: str, templater, 
                   q_dict: dict, language_desc: str, user_name: str) -> list:
        """Generate questions based on the given cluster and type.
        
        Args:
            cluster: The data cluster containing entity information.
            question_type: Type of questions to generate.
            templater: Template handler object.
            q_dict: Dictionary of question types.
            language_desc: Language description string.
            user_name: Name of the user.
            
        Returns:
            List of generated questions.
        """
        user_input = self._get_Q_input(cluster, user_name)

        messages = [
            {
                "role": "system",
                "content": templater.get_Q_template(
                    question_type_prompt=q_dict[question_type]["prompt"]
                ),
            },
            {"role": "user", "content": user_input + language_desc},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            res = response.choices[0].message.content
        except Exception as e:
            logging.error(traceback.format_exc())
        
        # post-processing
        try:
            pattern = r"Question\s*\d+\s*:\s*(.*?)\|\|"
            questions = re.findall(pattern, res + "||")
        except Exception as e:
            logger.error(traceback.format_exc())
            questions = []
            return questions

        # safety check
        if questions:
            if "|" in questions[0] and len(questions) == 0:
                questions = questions[0].split("|")

        return questions


    def _A_generate(self, cluster: dict, question: str, question_type: str, 
                   templater, language_desc: str, user_name: str) -> tuple:
        """Generate answers based on questions and clusters.
        
        Args:
            cluster: The data cluster containing entity information.
            question: The question to answer.
            question_type: Type of question.
            templater: Template handler object.
            language_desc: Language description string.
            user_name: Name of the user.
            
        Returns:
            Tuple of (answer_text, answer_type).
        """
        user_input = self._get_A_input(cluster, question, user_name)
        system_prompt, answer_type = templater.get_A_template(question_type)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input + language_desc},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            result = response.choices[0].message
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
        except Exception as e:
            logging.error(traceback.format_exc())
            
        return final_result, answer_type