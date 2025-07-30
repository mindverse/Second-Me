import json

from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()
import re
import statistics
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from .prompt import Prompts
from .utils import Note, Entity, Timeline, EntityWiki, Conversation, TimelineType, EntityType, MonthlyTimeline, \
    MONTH_TIME_FORMAT

load_dotenv(override=True)

user_llm_config_service = UserLLMConfigService()
user_llm_config = user_llm_config_service.get_available_llm()
if user_llm_config is None:
    client = None
    MODEL_NAME = None
else:
    client = OpenAI(
        api_key=user_llm_config.chat_api_key,
        base_url=user_llm_config.chat_endpoint,
        timeout=45.0,  # Set global timeout
    )
    MODEL_NAME = user_llm_config.chat_model_name

from .utils import (build_clusters,
                    find_neibor_entities,
                    parse_daily_timelines, group_timelines_by_month, group_timelines_by_day)

PHRASES_LENGTH_THRESHOLD = 3
LONG_PHRASES_DISTANCE_THRESHOLD = 3
SHORT_PHRASES_DISTANCE_THRESHOLD = 1
RATE_THRESHOLD = 0.5
NEIBOR_ENTITY_N = 5
ENTITY_MERGE_BATCH_SIZE = CONCEPT_MERGE_BATCH_SIZE = 3

ENTITY_EXTRACTOR_DEFAULT_PROMPT = {
    "entity_extractor": Prompts.ENTITY_EXTRACT_SYSTEM_PROMPT_simplfied,
    "duplicate_entities": Prompts.DUPLICATE_ENTITY_SYSTEM_PROMPT,
    "merge_entities": Prompts.MERGE_ENTITY_SYSTEM_PROMPT,
    "extract_filter": Prompts.ENTITY_EXTRACT_FILTER_SYSTEM_PROMPT,
    "generate_timeline_by_note": Prompts.GENERATE_TIMELINE_BY_NOTE_SYSTEM_PROMPT
}

PERSONAL_WIKI_DEFAULT_PROMPT = {
    "personal_wiki_entity": Prompts.PERSONAL_WIKI_ENTITY_SYSTEM_PROMPT_gt_v0,
    "personal_wiki_person": Prompts.PERSONAL_WIKI_PERSON_SYSTEM_PROMPT_gt_v0,
    "personal_wiki_location": Prompts.PERSONAL_WIKI_LOCATION_SYSTEM_PROMPT_gt_v0,
    "personal_wiki_concept": Prompts.PERSONAL_WIKI_CONCEPT_SYSTEM_PROMPT,
    "timeline_generate": Prompts.TIMELINE_GENERATE_SYSTEM_PROMPT_gt_v0,
    "monthly_timeline_title_entity": Prompts.MONTHLY_TIMELINE_TITLE_ENTITY_SYSTEM_PROMPT,
    "monthly_timeline_title_person": Prompts.MONTHLY_TIMELINE_TITLE_PERSON_SYSTEM_PROMPT,
    "monthly_timeline_title_location": Prompts.MONTHLY_TIMELINE_TITLE_LOCATION_SYSTEM_PROMPT,
    "monthly_timeline_title_concept": Prompts.MONTHLY_TIMELINE_TITLE_CONCEPT_SYSTEM_PROMPT,
}


class EntityScorer:
    def __init__(self, langfuse_dict: Dict[str, Any]):
        self.llm = client
        self.langfuse_dict = langfuse_dict

    def score_entities(self, entities: List[Entity], notes: List[Note], conversations: List[Conversation],
                       global_bio: str, user_name: str) -> List[Entity]:
        max_workers = max(1, min(10, len(entities)))  # 确保 max_workers 至少为 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._calculate_entity_score, entity, notes, conversations, global_bio, user_name)
                for entity in entities]
            scored_entities = [future.result() for future in futures]

        return scored_entities

    def update_entity_scores(self, entities: List[Entity], notes: List[Note], global_bio: str, user_name: str) -> List[
        Entity]:
        max_workers = max(1, min(10, len(entities)))  # 确保 max_workers 至少为 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._recalculate_entity_score, entity, notes, global_bio, user_name)
                       for entity in entities]
            updated_entities = [future.result() for future in futures]

        return updated_entities

    def _calculate_entity_score(self, entity: Entity, notes: List[Note], conversations: List[Conversation],
                                global_bio: str, user_name: str) -> Entity:
        # 从实体时间线中提取笔记ID
        note_ids = [int(timeline.note_id) for timeline in entity.timelines]
        # 根据笔记ID从笔记列表中筛选出对应的笔记
        entity_notes = [note for note in notes if note.id in note_ids]
        # 更新entity_note的content字段
        for entity_note in entity_notes:
            entity_note.content = next((note.content for note in notes if note.id == entity_note.id), "")

        # 构建提示消息
        message = Prompts.return_introspection_extracted_entities_prompt(
            entity_name=entity.name,  # 实体名称
            input_notes=entity_notes,  # 输入的笔记列表
            user_name=user_name,  # 用户名
            global_bio=global_bio,  # 全局生物信息
            system_prompt=self.langfuse_dict['extract_filter']['system_prompt']  # 系统提示
        )
        answer = self.llm.chat.completions.create(
            model=MODEL_NAME,
            messages=message,
        )
        # 获取对话生成的内容
        content = answer.choices[0].message.content
        # 解析并更新实体分数
        self._parse_and_update_score(entity, content)
        return entity

    def _recalculate_entity_score(self, entity: Entity, notes: List[Note], global_bio: str, user_name: str) -> Entity:
        # 从实体时间线中提取内容
        timeline_contents = [timeline.content for timeline in entity.timelines]
        # 将现有的notes内容与timeline内容合并
        combined_contents = timeline_contents + [note.content for note in notes]

        # 确保传递的是Note对象而不是字符串
        entity_notes = [note for note in notes if note.content in combined_contents]

        # 构建提示消息
        message = Prompts.return_introspection_extracted_entities_prompt(
            entity_name=entity.name,  # 实体名称
            input_notes=entity_notes,  # 确保是Note对象列表
            user_name=user_name,  # 用户名
            global_bio=global_bio,  # 全局生物信息
            system_prompt=self.langfuse_dict['extract_filter']['system_prompt']  # 系统提示
        )
        # 与LLM模型进行对话生成
        # answer = self.llm.chat_generate(message)
        answer = self.llm.chat.completions.create(
            model=MODEL_NAME,
            messages=message,
        )
        # 获取对话生成的内容
        content = answer.choices[0].message.content
        # 解析并更新实体分数
        self._parse_and_update_score(entity, content)
        return entity

    def _parse_and_update_score(self, entity: Entity, content: str):
        try:
            # 提取JSON部分
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if not json_match:
                # 如果没有json标记，尝试直接解析整个内容
                score_data = json.loads(content)
            else:
                score_data = json.loads(json_match.group(1))

            # 更新实体的分数和原因
            entity.personal_uniqueness_reason = score_data['personal_uniqueness']['reason']
            personal_uniqueness_score = score_data['personal_uniqueness']['score']
            entity.contextual_relevance_reason = score_data['contextual_relevance']['reason']
            contextual_relevance_score = score_data['contextual_relevance']['score']
            entity.personal_significance_reason = score_data['personal_significance']['reason']
            personal_significance_score = score_data['personal_significance']['score']
            entity.rarity_reason = score_data['rarity']['reason']
            rarity_score = score_data['rarity']['score']
            entity.time_relevance_reason = score_data['time_relevance']['reason']
            time_relevance_score = score_data['time_relevance']['score']
            entity.frequency_reason = score_data['frequency']['reason']
            frequency_score = score_data['frequency']['score']
            entity.emotional_connection_reason = score_data['emotional_connection']['reason']
            emotional_connection_score = score_data['emotional_connection']['score']

            # 计算平均分数
            scores = [
                personal_uniqueness_score,
                contextual_relevance_score,
                personal_significance_score,
                rarity_score,
                time_relevance_score,
                emotional_connection_score,
                frequency_score
            ]

            average_score = statistics.mean(scores)
            # entity.score = average_score
            # 仅在新得分高于旧得分且新得分大于零时更新
            if average_score > entity.score and average_score > 0:
                entity.score = average_score

            # 根据分数决定是否生成wiki.lpm部分对实体召回率有要求，把生成wiki的门槛降低
            if average_score > 0.4:
                entity.gen_wiki = True
            elif average_score > 0.2 and entity.freq >= 2:
                entity.gen_wiki = True
            else:
                entity.gen_wiki = False

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error parsing entity score response: {str(e)}\nContent: {content}")
            entity.score = 0
            entity.gen_wiki = False
            entity.personal_uniqueness_reason = ""
            entity.contextual_relevance_reason = ""
            entity.personal_significance_reason = ""
            entity.rarity_reason = ""
            entity.time_relevance_reason = ""
            entity.frequency_reason = ""
            entity.emotional_connection_reason = ""


class EntityExtractor:
    _input_keys: List = ["userName", "aboutMe", "notes", "entities", "preferredLanguage"]
    _output_keys: List = ["entities"]
    _must_keys: List = ["userName", "notes", "entities"]

    model_params = {
        "temperature": 0,
        "max_tokens": 8000,
        "top_p": 0,
        "frequency_penalty": 0,
        "seed": 42,
        "presence_penalty": 0,
        # "request_timeout": 60,
        # "max_retries": 1
    }

    def __init__(self, **kwargs):
        self.max_threads = 10
        self.model_name = "gpt-4o-mini"  # 存在不稳定性
        # self.model_name = "gpt-4o"
        self.class_name = self.__class__.__name__
        self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.model_params.update(**kwargs)
        self.langfuse_dict = {
            k: self._get_langfuse_prompt(k, v) for k, v in ENTITY_EXTRACTOR_DEFAULT_PROMPT.items()
        }
        # 将self.llms设置为一个字典，其中包含不同的OpenAI客户端
        self.llms = {
            "entity_extractor": client,
            "duplicate_entities": client,
            "merge_entities": client
        }

        # self.redis_message = MessageRedis()

    def _get_langfuse_prompt(self, prompt_key, default_prompt) -> Dict[str, Any]:
        # try:
        #     # lf_prompt = self.get_prompt(name=prompt_key, label=self.prompt_label)
        #     # system_prompt = lf_prompt.prompt[0].get('content', default_prompt)
        #     # logging.info(f"Get prompt success: {prompt_key}")
        # except Exception as e:
        #     logging.error(f"Failed to get prompt: {traceback.format_exc()}")
        #     lf_prompt = None
        #     system_prompt = default_prompt
        system_prompt = default_prompt
        return {"system_prompt": system_prompt}

    def _call_(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用测试函数。
        Args:
            inputs (Dict[str, Any]): 包含用户信息的字典。
        Returns:
            Dict[str, Any]: 包含提取出的实体列表的字典。
        """

        # 获取输入参数
        entity_extract_input = inputs
        user_name = entity_extract_input.get("userName", "")
        notes = entity_extract_input.get("notes", [])
        conversations = entity_extract_input.get("conversations", [])
        entities = entity_extract_input.get("entities", [])
        user_self_intro = entity_extract_input.get("aboutMe", "")
        preferred_language = entity_extract_input.get("preferredLanguage", "English/English")
        global_bio = entity_extract_input.get("globalBio", "")
        if global_bio:
            conclusion_match = re.search(r'Conclusion ###\s*([\s\S]*?)$', global_bio)
            if conclusion_match:
                global_bio = conclusion_match.group(1).strip()
        original_entities = entity_extract_input.get("entities", [])

        notes = [Note(**note, userName=user_name) for note in notes]
        # 将 conversations 列表中的每个元素转换为 Conversation 对象
        conversations = [Conversation(**conversation, userName=user_name) for conversation in conversations]
        # 将 entities 列表中的每个元素转换为 Entity 对象
        old_entities = [Entity(**entity) for entity in entities]

        # 提取实体
        raw_entities = self.extract_entities(notes, conversations, preferred_language)
        # 创建实体评分器
        entity_scorer = EntityScorer(self.langfuse_dict)
        updated_entities = []

        # 如果用户没有历史实体列表，则初始化实体列表
        if not old_entities:
            logger.info(f"The user has no historical entity list, starting the entity list initial process!")
            entities = self.initialize_entity_list(raw_entities, preferred_language, notes, conversations, global_bio,
                                                   user_name)
            entities = entity_scorer.score_entities(entities, notes, conversations, global_bio, user_name)

            updated_entities = entities
            logger.info(
                f"The entity list initial process is completed! Successfully extracted {len(entities)} entities!")
        # 如果用户有历史实体列表，则更新实体列表
        else:
            logger.info(f"The user has historical entity list, starting the entity list update process!")
            new_entities = self.initialize_entity_list(raw_entities, preferred_language, notes, conversations,
                                                       global_bio, user_name)
            logger.info(f"After merge duplicate entity, Entity nums: {len(raw_entities)} -> {len(new_entities)}")

            # 更新实体列表并获取需要重新打分的实体
            entities, entities_to_score = self.update_entity_list(old_entities, new_entities)
            # 对需要重新打分的实体进行打分
            scored_entities = entity_scorer.update_entity_scores(entities_to_score, notes, global_bio, user_name)

            # 更新实体列表中的分数
            entity_dict = {entity.name: entity for entity in entities}
            for scored_entity in scored_entities:
                if scored_entity.name not in entity_dict:
                    logger.warning(f"Entity name '{scored_entity.name}' not found in entity_dict. Adding it now.")
                    entity_dict[scored_entity.name] = scored_entity
                elif scored_entity.score > entity_dict[scored_entity.name].score:
                    entity_dict[scored_entity.name] = scored_entity
            entities = list(entity_dict.values())

            logger.info(
                f"The entity list update process is completed! Successfully extracted {len(entities)} entities!")
            updated_entities = self.compare_and_keep_max_scores([], scored_entities)

        final_entities = self.compare_and_keep_max_scores(original_entities, entities)

        # 返回包含提取出的实体列表的字典
        return {
            "entities": [entity.to_dict() for entity in final_entities if entity.timelines],
            "updated_entities": [entity.to_dict() for entity in updated_entities] if updated_entities else []
        }

    def compare_and_keep_max_scores(self, original_entities: List[dict], updated_entities: List[Entity]) -> List[
        Entity]:
        # 创建字典存储最大分数实体
        entity_map = {entity.name: entity for entity in updated_entities}

        # 处理原始实体字典
        for entity_dict in original_entities:
            entity_name = entity_dict['name']
            original_score = float(entity_dict['score'])

            # 如果实体已存在且原始分数更高
            if entity_name in entity_map:
                if original_score > entity_map[entity_name].score:
                    # 保留原始分数并更新其他属性
                    entity_map[entity_name].score = original_score
                    entity_map[entity_name].freq = entity_dict.get('freq', entity_map[entity_name].freq)
            else:
                # 创建新Entity对象并保留
                new_entity = Entity(
                    name=entity_name,
                    entityType=EntityType(entity_dict['entity_type']),
                    score=original_score,
                    freq=entity_dict['freq'],
                    synonyms=entity_dict.get('synonyms', []),
                    timelines=[Timeline(**t) for t in entity_dict.get('timelines', [])]
                )
                entity_map[entity_name] = new_entity

        return list(entity_map.values())

    def extract_entities(self, notes: List[Note], conversations: List[Conversation], prefer_lang: str) -> List[
        Dict[str, Any]]:
        """
        从给定的笔记和对话中提取实体。

        Args:
            notes (List[Note]): 包含所有笔记的列表。
            conversations (List[Conversation]): 包含所有对话的列表。
            prefer_lang (str): 用户偏好的语言。

        Returns:
            List[Dict[str, Any]]: 包含提取出的实体的列表，每个实体表示为一个字典。
        """

        # 定义处理单个项目的函数
        def process_item(item: Union[Note, Conversation]) -> list[dict[str, Any]]:
            # 调用提取实体的函数
            return self.extract_entities_by_notes([item], prefer_lang)

        # 将笔记和对话合并为一个列表
        all_items = notes + conversations
        # 初始化结果列表
        results = []

        # 使用ThreadPoolExecutor创建线程池并执行处理函数
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # 使用tqdm显示进度条
            futures = [executor.submit(process_item, item) for item in tqdm(all_items, desc="处理实体")]
            # 遍历Future对象并获取处理结果
            for future in futures:
                try:
                    # 将处理结果添加到结果列表中
                    results.extend(future.result())
                except Exception as e:
                    # 捕获异常并记录错误日志
                    logger.error(f"Error processing item: {e}")

        # 返回结果列表
        return results

    def extract_entities_by_notes(self, entity_input: List[Union[Note, Conversation]], prefer_lang: str) -> List[
        Dict[str, Any]]:
        message = Prompts.return_entity_extract_prompt(prefer_lang=prefer_lang,
                                                       entity_input=entity_input,
                                                       user_name=entity_input[0].user_name,
                                                       user_self_intro="",
                                                       global_bio="",
                                                       system_prompt=self.langfuse_dict['entity_extractor'][
                                                           'system_prompt'])
        try:

            answer = self.llms["entity_extractor"].chat.completions.create(messages=message, model=MODEL_NAME,
                                                                           **self.model_params)
            content = answer.choices[0].message.content
            return self.extract_entity_postprocess_new(content)
        except Exception as e:
            logger.error(f"Error in extract_entities_by_notes: {e}")
            return []

    def extract_entity_postprocess_new(self, raw_result: str) -> List[Dict[str, Any]]:
        logger.debug(raw_result)
        json_data_str = re.search(r'```json\n([\s\S]+?)\n```', raw_result).group(1)
        json_data = json.loads(json_data_str)
        final_entities = []
        for entity_data in json_data:
            entity_name = self.clean_entity_name(entity_data['entity_name'])
            entity_type = entity_data['entity_type']
            entity_desc = entity_data['entity_description']
            entity_notes = entity_data['notes']
            final_entities.append({
                "name": entity_name,
                "entity_type": entity_type,
                "description": entity_desc,
                "notes": entity_notes
            })
        return final_entities

    def clean_entity_name(self, entity_name: str) -> str:
        # 清理实体名称，处理大小写和特殊字符
        entity_name = entity_name.strip()
        entity_name = re.sub(r'\s+', ' ', entity_name)
        entity_name = re.sub(r'(\w+)\.(\w+)', r'\1\2', entity_name)
        # candidate_symbols = ["'", '"']
        # while entity_name and entity_name[0] in candidate_symbols:
        #     entity_name = entity_name[1:]
        # while entity_name and entity_name[-1] in candidate_symbols:
        #     entity_name = entity_name[:-1]
        return entity_name

    def initialize_entity_list(self, raw_entities: List[Dict[str, Any]], preferred_language, notes: List[Note],
                               conversations: List[Conversation], global_bio: str, user_name: str):
        """
        初始化实体列表。

        Args:
            raw_entities (List[Dict[str, Any]]): 原始实体列表，每个实体是一个字典，包含实体的名称和其他信息。
            preferred_language (str): 用户偏好的语言。
            notes (List[Note]): 笔记列表，每个笔记是一个包含笔记内容的对象。
            conversations (List[Conversation]): 对话列表，每个对话是一个包含对话内容的对象。
            global_bio (str): 全局生物信息，是一个字符串。
            user_name (str): 用户名，是一个字符串。

        Returns:
            List[Entity]: 实体列表，每个实体是一个包含实体名称、类型、频率、同义词和时间线的对象。

        """
        # 统计每个实体名称出现的频率
        entity_freqs = Counter(entity["name"] for entity in raw_entities)

        # 获取长度超过阈值的实体名称列表
        long_phrase_entities = list(set(entity["name"] for entity in raw_entities
                                        if len(entity["name"]) > PHRASES_LENGTH_THRESHOLD))

        # 获取长度不超过阈值的实体名称列表
        short_phrase_entities = list(set(entity["name"] for entity in raw_entities
                                         if len(entity["name"]) <= PHRASES_LENGTH_THRESHOLD))

        # 根据长短语阈值构建实体名称的聚类
        long_phrase_clusters = build_clusters(long_phrase_entities, LONG_PHRASES_DISTANCE_THRESHOLD)
        short_phrase_clusters = build_clusters(short_phrase_entities, SHORT_PHRASES_DISTANCE_THRESHOLD)

        # 将长短语聚类结果合并，并统计每个实体的频率，生成原始实体列表
        raw_entity_list = self.clustered_entities(long_phrase_clusters + short_phrase_clusters, entity_freqs)

        # 构建同义词映射表
        synonym_map = {
            synonym: entity_name
            for entity_name, entity_info in raw_entity_list.items()
            for synonym in entity_info["synonyms"]
        }
        # 创建笔记ID到笔记对象的映射
        note_dict = {note.id: note for note in notes}

        # 处理原始实体列表中的每个实体
        for entity in raw_entities:
            # 如果实体名称不在同义词映射表中，则将其自身作为同义词加入映射表，并初始化实体信息
            if entity["name"] not in synonym_map:
                synonym_map[entity["name"]] = entity["name"]
                raw_entity_list[entity["name"]] = {
                    "synonyms": [entity["name"]],
                    "timelines": []
                }
            # 遍历实体中的每个笔记
            for entity_note in entity.get("notes", []):
                try:
                    note_id = int(entity_note.get("note_id", ""))
                    origin_note = note_dict.get(note_id)
                    if not origin_note:
                        logger.warning(f"Note ID {note_id} not found")
                        continue

                    # 获取实际笔记内容
                    content = origin_note.content
                    # 根据笔记生成时间线
                    entity_timeline = self.gen_timeline_by_note(user_name, preferred_language, entity, entity_note,
                                                                notes,
                                                                debug=False)  # 目前来看不需要为每个实体生成raw timeline
                    # 将生成的时间线添加到实体的时间线列表中
                    raw_entity_list[synonym_map[entity["name"]]]["timelines"].append(
                        Timeline(createTime=entity_note.get("create_time", ""),
                                 noteId=str(note_id),
                                 content=content,
                                 description=entity_timeline,
                                 timelineType=TimelineType(entity["entity_type"]))  # utils.py内容to_dict修改
                    )
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Skip invalid note for entity {entity['name']}: {str(e)}")
                    continue

        # 为每个实体设置实体类型
        for entity_name, entity_info in raw_entity_list.items():
            raw_entity_list[entity_name]["entity_type"] = self.judge_entity_type(entity_info["timelines"])

        # 打印处理后的实体数量
        logger.info(
            f"After duplicated entities processing, Entity nums: {len(raw_entities)} -> {len(raw_entity_list)}")

        # 根据原始实体列表生成最终的实体列表
        entities = [
            Entity(name=entity_name,
                   entityType=entity_info["entity_type"],
                   freq=len(entity_info["timelines"]),
                   synonyms=entity_info["synonyms"],
                   timelines=entity_info["timelines"])
            for entity_name, entity_info in raw_entity_list.items()
        ]
        return entities

    def gen_timeline_by_note(self, user_name, preferred_language, entity, entity_note, notes,
                             debug: bool = False) -> str:

        # 如果开启了调试模式，直接返回空字符串
        if debug == True:
            return ""

        origin_note = None
        for note in notes:
            # 找到与entity_note中的note_id匹配的note
            if note.id == int(entity_note["note_id"]):
                origin_note = note

        # 如果没有找到匹配的note，返回空字符串
        if origin_note is None:
            return ""
        else:
            # 生成基于note的提示信息
            message = Prompts.generate_timeline_by_notes(user_name, preferred_language, entity, origin_note,
                                                         system_prompt=self.langfuse_dict['generate_timeline_by_note'][
                                                             'system_prompt'])
            # 使用实体提取器生成回答
            answer = self.llms["entity_extractor"].chat.completions.create(messages=message, model=MODEL_NAME,
                                                                           **self.model_params)
            content = answer.choices[0].message.content

            # 使用正则表达式匹配"### Entity Timeline"之后的内容
            content_matches = re.search(r"### Entity Timeline(.*?)(###|$)", content, re.DOTALL)

            # 如果没有找到匹配的内容，记录警告并返回空字符串
            if not content_matches:
                logger.warning(f"无法从回答中提取时间线内容: {content[:100]}...")
                return ""

            # 获取匹配的内容
            raw_result = content_matches.group(1)

            # 替换换行符并去除首尾的空白字符
            content_section = raw_result.replace("\n", " ").strip()

            # 分割字符串，获取": "之后的内容作为时间线,增加安全检查，确保字符串中包含": "
            if ": " in content_section:
                timeline = content_section.split(": ", 1)[1]
            else:
                timeline = content_section

            # 更新content字段
            entity_note["content"] = origin_note.content

            return timeline

    def clustered_entities(self, clusters: List[List[str]], entity_freqs: Counter):
        """
        根据实体簇和实体频率统计信息，将实体簇中的实体映射到其同义词集合，并统计每个实体的频率和同义词。

        Args:
            clusters (List[List[str]]): 实体簇列表，每个子列表表示一个实体簇，包含该簇中的所有实体名称。
            entity_freqs (Counter): 实体频率统计信息，表示每个实体名称出现的频率。

        Returns:
            dict: 包含每个实体名称及其同义词集合、频率的字典。字典的键为实体名称，值为包含"freq"（频率）、"synonyms"（同义词列表）、"timelines"（时间线列表，本函数未使用）的字典。

        """
        entity_cluster_dict = {}

        for cluster in clusters:
            # 如果簇中实体数量小于2，直接创建映射关系
            if len(cluster) < 2:
                entity_maps = {e: [e] for e in cluster}
            else:
                # 调用内部方法获取原始实体映射关系
                raw_entity_maps = self.deep_duplicated_entities(cluster)
                # 根据实体频率统计信息，将原始实体映射关系转换为优化后的映射关系
                entity_maps = {
                    # 选择频率最高的实体作为代表，并将同义词集合映射到该实体
                    max(s, key=lambda x: entity_freqs[x]): s
                    for s in raw_entity_maps.values()
                }

            # 遍历优化后的实体映射关系
            for entity_name, synonyms in entity_maps.items():
                # 如果实体名称不在结果字典中，则初始化该实体名称的字典项
                if entity_name not in entity_cluster_dict:
                    entity_cluster_dict[entity_name] = {
                        "freq": 0,
                        "synonyms": [],
                        "timelines": []
                    }
                # 遍历同义词集合
                for synonym in synonyms:
                    # 累加同义词的频率
                    entity_cluster_dict[entity_name]["freq"] += entity_freqs[synonym]
                    # 将同义词添加到同义词列表中
                    entity_cluster_dict[entity_name]["synonyms"].append(synonym)

        return entity_cluster_dict

    def deep_duplicated_entities(self, entity_list: List[str]) -> Dict[str, List[str]]:
        """
        深度检测并处理实体重复问题。

        Args:
            entity_list (List[str]): 待检测的实体列表。

        Returns:
            Dict[str, List[str]]: 包含处理后的实体映射关系的字典。

        """
        # 创建初始的实体映射关系字典，将每个实体映射到它自身
        default_entity_maps = {
            e: [e] for e in entity_list
        }

        # 定义重复实体的正则表达式模式
        duplicate_entity_pattern = r'\{.*\}'

        # 生成系统提示信息
        message = Prompts.return_duplicate_entity_prompt(entity_list=entity_list,
                                                         system_prompt=self.langfuse_dict["duplicate_entities"][
                                                             "system_prompt"])

        # 与LLM进行对话，生成处理重复实体的回答
        answer = self.llms["duplicate_entities"].chat.completions.create(messages=message, model=MODEL_NAME,**self.model_params)

        # 从回答中提取内容
        content = answer.choices[0].message.content

        # 解析JSON响应，更新实体映射关系字典
        entity_maps = self.parse_json_response(content, default_entity_maps, duplicate_entity_pattern)

        return entity_maps

    def update_entity_list(self, old_entities: List[Entity], new_entities: List[Entity]) -> Tuple[
        List[Entity], List[Entity]]:
        """
        更新已有的实体列表，合并新提取的实体。
        Args:
            old_entities (List[Entity]): 已有的实体列表。
            new_entities (List[Entity]): 新提取的实体列表。
        Returns:
            Tuple[List[Entity], List[Entity]]: 更新后的实体列表和需要重新打分的实体列表。
        """
        ## 更新已有的实体列表，合并新提取的实体。
        # 创建一个字典，将已有的实体按照名称作为键存储
        old_entity_dict = {entity.name: entity for entity in old_entities}

        # 创建一个同义词字典，将每个实体的同义词映射到实体名称
        synonym_dict = {
            synonym: entity.name for entity in old_entities for synonym in entity.synonyms
        }
        # 将所有同义词存储在集合中
        old_entity_synonym_names = set(synonym_dict.keys())

        # 初始化未合并的实体列表和需要重新打分的实体列表
        unmerged_entities: List[Entity] = []
        entities_to_score: List[Entity] = []  # 需要重新打分的实体列表

        for entity in new_entities:
            # 检查新实体是否有同义词与已有实体的同义词匹配
            if union_name := set(entity.synonyms) & old_entity_synonym_names:
                union_entity_dict = {}

                for name in union_name:
                    match_old_entity = old_entity_dict[synonym_dict[name]]
                    union_entity_dict[match_old_entity.name] = union_entity_dict.get(match_old_entity.name,
                                                                                     0) + match_old_entity.freq

                # 找到频率最高的实体名称作为合并后的实体名称
                max_entity_name = max(union_entity_dict, key=union_entity_dict.get)
                old_entity = old_entity_dict[max_entity_name]

                # 修复 bug：在合并实体前，先处理 timelines 字段，避免重复挂载
                # 创建一个集合来存储已有的 timeline noteId，用于去重
                existing_note_ids = {timeline.note_id for timeline in old_entity.timelines}

                # 过滤掉新实体中已存在于旧实体的 timelines
                new_timelines = []
                for timeline in entity.timelines:
                    if timeline.note_id not in existing_note_ids:
                        new_timelines.append(timeline)
                        existing_note_ids.add(timeline.note_id)

                # 更新实体的 timelines，只保留不重复的
                entity.timelines = new_timelines

                # 现在可以安全地合并实体
                old_entity.merge_entity(entity)
                entities_to_score.append(old_entity)  # 合并后的实体需要重新打分
            else:
                unmerged_entities.append(entity)
                entities_to_score.append(entity)  # 新实体需要打分

        # 将未合并的实体转换为字典存储
        unmered_entity_dict = {entity.name: entity for entity in unmerged_entities}
        logger.info(f"After merge into old entities, Entity nums: {len(new_entities)} -> {len(unmerged_entities)}")
        old_entities = list(old_entity_dict.values())

        ## merge same object entity into old entities from new entities
        # 如果存在未合并的实体，则继续处理
        if unmerged_entities:
            # 找到与已有实体相邻的未合并实体
            neibor_entity_dict = find_neibor_entities(unmerged_entities, old_entities, NEIBOR_ENTITY_N)
            # 对未合并的实体进行预处理
            entities_batch_list = self.batch_merge_preprocess(neibor_entity_dict, old_entities, unmered_entity_dict)

            # 使用线程池执行批量合并操作
            with ThreadPoolExecutor(max_workers=min(self.max_threads, len(entities_batch_list))) as executor:
                futures = [executor.submit(self.batch_merge_entity, entities_batch) for entities_batch in
                           entities_batch_list]
                results = [merge_res for future in futures for merge_res in future.result()]

            logger.info(f"unmered_entity_dict.keys: {list(unmered_entity_dict.keys())}")
            logger.info(f"new_entity_name_list: {[res['new_entity_name'] for res in results]}")
            logger.info(f"Merge Result Num: {len(results)}")

            # 遍历合并结果，根据合并情况进行处理
            for merge_res in results:
                new_entity_name = merge_res["new_entity_name"]
                if new_entity_name not in unmered_entity_dict:
                    logger.warning(f"Entity {new_entity_name} not in new entity dict!, merge model performance issue!")
                    continue

                # 如果合并成功且目标实体在已有实体字典中
                if merge_res["merged"] and merge_res["merge_target"] in old_entity_dict:
                    old_entity = old_entity_dict[merge_res["merge_target"]]
                    old_entity.merge_entity(unmered_entity_dict[new_entity_name])
                    old_entity.entity_type = self.judge_entity_type(old_entity.timelines)
                    entities_to_score.append(old_entity)  # 合并后的实体需要重新打分
                else:
                    # 如果合并失败或目标实体不在已有实体字典中，将新实体添加到已有实体列表中
                    new_entity = unmered_entity_dict[new_entity_name]
                    old_entities.append(new_entity)
                    entities_to_score.append(new_entity)  # 新实体需要打分

        return old_entities, entities_to_score

    def batch_merge_preprocess(self, neibor_entity_dict: Dict[str, Entity], old_entities: List[Entity],
                               new_entities: Dict[str, Entity]) -> List[Dict[str, Any]]:
        """
        批量合并实体，将新提取的实体与旧实体进行合并，并返回合并结果。

        Args:
            neibor_entity_dict (Dict[str, Entity]): 邻接实体字典，键为实体名称，值为实体对象。
            old_entities (List[Entity]): 旧实体列表。
            new_entities (Dict[str, Entity]): 新提取的实体字典，键为实体名称，值为实体对象。

        Returns:
            List[Dict[str, Any]]: 返回合并后的实体信息列表，每个元素为包含合并结果的字典。

        """
        # 将旧实体列表转换为字典
        old_entities_dict = {entity.name: entity for entity in old_entities}

        # 构建邻接实体列表
        neibor_entity_list = [
            # 为每个新实体构建字典
            {
                # 新实体名称及其描述
                "new_entity_name": {
                    "name": new_entity_name,
                    # 如果新实体名称存在于新实体字典中，则获取其描述，否则为空字符串
                    "description": new_entities[new_entity_name].description if new_entity_name in new_entities else ""
                },
                # 候选实体名称列表
                "candidate_entity_names": [
                    # 为每个候选实体名称构建字典
                    {
                        "name": candidate_entity_name,
                        "similar_names": old_entities_dict[candidate_entity_name].synonyms,
                        "description": old_entities_dict[candidate_entity_name].description
                    } for candidate_entity_name in candidate_entity_names if candidate_entity_name in old_entities_dict
                ]
            } for new_entity_name, candidate_entity_names in neibor_entity_dict.items()
        ]

        # 将邻接实体列表按批次大小分割
        entities_batch_list = [
            neibor_entity_list[i: i + ENTITY_MERGE_BATCH_SIZE]
            for i in range(0, len(neibor_entity_list), ENTITY_MERGE_BATCH_SIZE)
        ]

        input_states_list = []

        # 遍历每个批次
        for entities_batch in entities_batch_list:
            input_statements = ""
            for i, entity in enumerate(entities_batch):
                # 添加新实体信息
                input_statements += f"Group {i + 1}:\n"
                input_statements += f"  New Entity Name: {entity['new_entity_name']['name']}\n"
                input_statements += f"  New Entity Description: {entity['new_entity_name']['description']}\n\n"
                # 遍历候选实体名称
                for j, candidate_entity in enumerate(entity["candidate_entity_names"]):
                    # 添加候选实体信息
                    input_statements += f"Candidate Entity {j + 1}:\n"
                    input_statements += f"  Name: {candidate_entity['name']}\n"
                    input_statements += f"  Similar Names: {candidate_entity['similar_names']}\n"
                    input_statements += f"  Description: \n  {candidate_entity['description']}\n"
                input_statements += "\n"
            # 将当前批次的输入语句添加到列表中
            input_states_list.append(input_statements)

        # 返回输入状态列表
        return input_states_list

    def batch_merge_entity(self, entities_batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        ## 批量合并实体，使用LLM判断是否需要合并。
        default_res = [{
            "merged": False,
            "merge_target": None
        } for _ in entities_batch]

        merge_entity_pattern = r'\[.*\]'
        messages = Prompts.return_merge_entity_prompt(merge_entity_json=entities_batch,
                                                      system_prompt=self.langfuse_dict["merge_entities"][
                                                          "system_prompt"])

        answer = self.llms["merge_entities"].chat.completions.create(messages=messages, model=MODEL_NAME,
                                                                     **self.model_params)
        content = answer.choices[0].message.content

        merge_res_list = self.parse_json_response(content, default_res, merge_entity_pattern)

        # 如果没有结果，直接返回空列表
        if not merge_res_list:
            return []

        return [
            {
                "merged": merge_res.get("merged", False),
                "merge_target": merge_res.get("merge_target") if merge_res.get("merged", False) else merge_res.get(
                    "new_entity_name", ""),
                "new_entity_name": merge_res.get("new_entity_name", "")
            } for merge_res in merge_res_list if isinstance(merge_res, dict)
        ]

    def judge_entity_type(self, entity_timelines: List[Timeline]) -> EntityType:
        if not entity_timelines:
            # 如果时间线为空，返回默认实体类型
            return EntityType.NORMAL_ENTITY
        person_rate = sum(timeline.timeline_type in [TimelineType.PERSON]
                          for timeline in entity_timelines) / len(entity_timelines)
        location_rate = sum(timeline.timeline_type in [TimelineType.GEO]
                            for timeline in entity_timelines) / len(entity_timelines)

        if person_rate > RATE_THRESHOLD:
            return EntityType.PERSON

        if location_rate > RATE_THRESHOLD:
            return EntityType.LOCATION

        return EntityType.NORMAL_ENTITY

    def parse_json_response(self, response: str, default_res: Optional[Union[Dict, List]], pattern: str) -> Dict[
        str, Any]:
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            logger.error(f"No Json Found: {response}")
            return default_res
        try:
            json_res = json.loads(matches[0])
            # 确保 'new_entity_name' 键存在
            # for item in json_res:
            #     if 'new_entity_name' not in item:
            #         item['new_entity_name'] = None
            # 确保 'json_res' 是一个列表，并且每个元素是字典
            if isinstance(json_res, list):
                for item in json_res:
                    if isinstance(item, dict) and 'new_entity_name' not in item:
                        item['new_entity_name'] = None
        except Exception as e:
            logger.error(f"Json Parse Error: {traceback.format_exc()}-{response}")
            return default_res
        return json_res


class PersonalWiki:
    _input_keys: List = ["userName", "oldWiki", "wikiType", "entityName", "timelines", "preferredLanguage"]
    _output_keys: List = ["entityWiki"]
    _must_keys: List = ["userName", "oldWiki", "wikiType", "entityName", "timelines", "preferredLanguage"]

    model_params = {
        "temperature": 0,
        "max_tokens": 3000,
        "top_p": 0,
        # "frequency_penalty": 0,
        # "seed": 42,
        # "presence_penalty": 0,
        # "request_timeout": 45,
        # "max_retries": 1,
    }

    def __init__(self, **kwargs):
        self.model_name = MODEL_NAME
        # self.model_name = "anthropic/claude-3-5-sonnet-20241022"
        self.max_threads = 5
        self.class_name = self.__class__.__name__
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.model_params.update(**kwargs)
        self.langfuse_dict = {
            k: self._get_langfuse_prompt(k, v) for k, v in PERSONAL_WIKI_DEFAULT_PROMPT.items()
        }
        self.llms = client

    def _get_langfuse_prompt(self, prompt_key, default_prompt) -> Dict[str, Any]:
        try:
            system_prompt = default_prompt
            model = MODEL_NAME
            logger.info(f"Get prompt success: {prompt_key}")
        except Exception as e:
            logger.error(f"Failed to get prompt [{prompt_key}]: {traceback.format_exc()}")
            system_prompt = default_prompt
            model = MODEL_NAME
        return {"system_prompt": system_prompt, "model": model}

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_name = inputs.get("userName", "")
        old_entity_wiki = inputs.get("oldEntityWiki", {})
        wiki_type = EntityType(inputs.get("wikiType", "NORMAL_ENTITY"))
        entity_name = inputs.get("entityName", "")
        user_self_intro = inputs.get("aboutMe", "")
        timelines = [Timeline(**timeline) for timeline in inputs.get("timelines", [])]

        preferred_language = inputs.get("preferredLanguage", "English/English")

        # 修改此行，确保提供 monthlyTimelines 参数
        old_entity_wiki = EntityWiki(**old_entity_wiki) if old_entity_wiki else None
        # old_entity_wiki = EntityWiki(**old_entity_wiki, monthlyTimelines=old_entity_wiki.get("monthlyTimelines", [])) if old_entity_wiki else None

        if not user_name:
            raise Exception(f"The user name is empty! Please check the user name!")

        if not entity_name or not timelines:
            raise Exception(f"The entity name or timelines is empty! Please check the entity name or timelines!")

        if wiki_type == EntityType.PERSON:
            entity_type = "person"
        elif wiki_type == EntityType.LOCATION:
            entity_type = "location"
        elif wiki_type == EntityType.CONCEPT:
            entity_type = "concept"
        else:
            entity_type = "entity"

        new_entity_wiki = self.update_entity_wiki_text(user_name, user_self_intro, entity_name, entity_type, timelines,
                                                       old_entity_wiki, preferred_language)
        # Do not need timeline for now
        # new_entity_wiki = self.update_entity_wiki_timelines(user_name, entity_name, entity_type, user_self_intro,
        #                                                     timelines, new_entity_wiki, preferred_language)

        return {
            "entityWiki": new_entity_wiki.to_dict()
        }

    def update_entity_wiki_text(self,
                                user_name: str,
                                user_self_intro: str,
                                entity_name: str,
                                entity_type: str,
                                timelines: List[Timeline],
                                old_entity_wiki: Optional[EntityWiki],
                                preferred_language: str):
        prompt_type = f"personal_wiki_{entity_type}"
        desc_type = "Concept Name" if entity_type == "concept" else "Entity Name"

        system_prompt = self.langfuse_dict[prompt_type]["system_prompt"].format(
            user_name=user_name,
            entity_name=entity_name,
            preview_version_wiki=old_entity_wiki.wiki_text if old_entity_wiki else "",
            prefer_lang=preferred_language,
            self_intro=user_self_intro
        )

        description_list = "\n".join([timeline._desc_() for timeline in timelines])

        user_prompt = f"""{desc_type}: {entity_name}

        # Impression Flow:
        # {description_list}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        llm = client
        answer = llm.chat.completions.create(messages=messages, model=self.langfuse_dict[prompt_type]["model"],
                                             **self.model_params)

        wiki_text = answer.choices[0].message.content

        logger.info(f"Generated wiki: {wiki_text}")

        if old_entity_wiki:
            old_entity_wiki.wiki_text = wiki_text
            return old_entity_wiki

        return EntityWiki(wikiText=wiki_text, monthlyTimelines=[])

    def update_entity_wiki_timelines(self,
                                     user_name: str,
                                     entity_name: str,
                                     entity_type: str,
                                     user_self_intro: str,
                                     timelines: List[Timeline],
                                     old_entity_wiki: Optional[EntityWiki],
                                     preferred_language: str):

        # Generate monthly timelines month by month, w/o cold start to avoid performance gap
        monthly_timeline_dict = group_timelines_by_month(timelines)

        system_prompt = self.langfuse_dict["timeline_generate"]["system_prompt"].format(
            user_self_intro=user_self_intro,
            user_name=user_name,
            prefer_lang=preferred_language,
            entity_name=entity_name
        )

        entity_wiki_month_date = {
            timeline.month_date: timeline.id for timeline in old_entity_wiki.monthly_timelines
        }

        max_month_idx = old_entity_wiki.max_month_idx if old_entity_wiki else 0

        missing_month_dates = set(monthly_timeline_dict.keys()) - set(entity_wiki_month_date.keys())

        for month_date in missing_month_dates:
            max_month_idx += 1
            entity_wiki_month_date[month_date] = max_month_idx

        with ThreadPoolExecutor(max_workers=min(self.max_threads, len(monthly_timeline_dict))) as executor:
            futures = [
                executor.submit(self.generate_monthly_timeline_byDay, entity_wiki_month_date[month_date], month_date,
                                monthly_timelines, system_prompt)
                for month_date, monthly_timelines in monthly_timeline_dict.items()]
            new_wiki_timelines = [future.result() for future in futures]

        # 把没有daily timeline的月份筛掉
        new_wiki_timelines = [timeline for timeline in new_wiki_timelines if timeline.daily_timelines]

        new_month_date = [timeline.month_date for timeline in new_wiki_timelines if timeline.month_date]
        origin_monthly_timelines = [timeline for timeline in old_entity_wiki.monthly_timelines if
                                    timeline.month_date not in new_month_date]
        origin_monthly_timelines.extend(new_wiki_timelines)
        monthly_timelines = sorted(origin_monthly_timelines,
                                   key=lambda x: datetime.strptime(x.month_date, MONTH_TIME_FORMAT))

        for month_idx, monthly_timeline in enumerate(monthly_timelines):
            if monthly_timeline.title:
                continue
            history_monthly_timelines = monthly_timelines[:month_idx]
            monthly_timelines[month_idx] = self.generate_monthly_timeline_title(history_monthly_timelines,
                                                                                monthly_timeline,
                                                                                entity_name,
                                                                                entity_type,
                                                                                preferred_language)

        old_entity_wiki.monthly_timelines = monthly_timelines
        return old_entity_wiki

    def generate_monthly_timeline_byDay(self,
                                        month_idx: int,
                                        month_date: str,
                                        timelines: List[Timeline],
                                        system_prompt: str,
                                        ) -> MonthlyTimeline:

        # 按天归并
        daily_timeline_dict = group_timelines_by_day(timelines)

        def generate_daily_timeline(day: str, daily_timelines: List[Timeline], day_idx: int) -> Dict[str, Any]:
            user_prompt = "\n".join([timeline._desc_(with_note_id=True) for timeline in daily_timelines])

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            llm = client
            answer = llm.chat.completions.create(messages=messages,
                                                 model=self.langfuse_dict["timeline_generate"]["model"],
                                                 **self.model_params)
            new_timeline_str = answer.choices[0].message.content
            logger.debug(f"Before processing:\n{new_timeline_str}")
            # 使用 parse_daily_timelines 函数解析每天的 timeline
            daily_timeline = parse_daily_timelines(new_timeline_str, day)
            daily_timeline["id"] = day_idx  # 设置 day_idx
            return daily_timeline

        with ThreadPoolExecutor(max_workers=min(self.max_threads, len(daily_timeline_dict))) as executor:
            futures = [executor.submit(generate_daily_timeline, day, daily_timelines, idx)
                       for idx, (day, daily_timelines) in enumerate(daily_timeline_dict.items())]
            daily_timelines = [future.result() for future in futures]
        daily_timelines = [timeline for timeline in daily_timelines if timeline['dateTime']]
        logger.debug(f"After processing and merging:\n{daily_timelines}")
        # 拼装成 MonthlyTimeline 对象
        monthly_timeline = MonthlyTimeline(
            id=month_idx,
            monthDate=month_date,
            title="",
            dailyTimelines=daily_timelines
        )

        return monthly_timeline

    def generate_monthly_timeline_title(self,
                                        history_monthly_timelines: List[MonthlyTimeline],
                                        monthly_timeline: MonthlyTimeline,
                                        entity_name: str,
                                        entity_type: str,
                                        prefer_lang: str):
        prompt_type = f"monthly_timeline_title_{entity_type}"
        system_prompt = self.langfuse_dict[prompt_type]["system_prompt"].format(
            prefer_lang=prefer_lang
        )

        history_monthly_timelines_str = "\n".join([timeline.title for timeline in history_monthly_timelines])

        if entity_type == "entity":
            prefix_prompt = ""
        elif entity_type == "person":
            prefix_prompt = f"Here are some of my experiences with {entity_name}."
        elif entity_type == "location":
            prefix_prompt = f"Current Location is {entity_name}."
        elif entity_type == "concept":
            prefix_prompt = f"Current Concept is {entity_name}."
        else:
            raise Exception(f"Invalid entity type: {entity_type}")

        user_prompt = f"""{prefix_prompt}
        Historical development stages and events:
        {history_monthly_timelines_str}

        Current events:
        {monthly_timeline._desc_()}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        llm = client
        answer = llm.chat.completions.create(messages=messages, model=self.langfuse_dict[prompt_type]["model"],
                                             **self.model_params)
        raw_monthly_timeline_title = answer.choices[0].message.content

        logger.info(f"Generated monthly timeline title:\n{raw_monthly_timeline_title}")

        date_str, title_str = self.parse_monthly_timeline_title(raw_monthly_timeline_title)
        if date_str != monthly_timeline.month_date:
            logger.warning(f"Date not match: {date_str} != {monthly_timeline.month_date}")

        monthly_timeline.title = title_str
        return monthly_timeline

    def parse_monthly_timeline_title(self, raw_monthly_timeline_title: str):
        date_pattern = r"Date:\s*([0-9]{4}-[0-9]{2})"
        title_pattern = r"Title:\s*([\s\S]*?)\s*(?:\n|$)"
        date_match = re.search(date_pattern, raw_monthly_timeline_title)
        title_match = re.search(title_pattern, raw_monthly_timeline_title)
        date_str = date_match.group(1) if date_match else ""
        title_str = title_match.group(1) if title_match else ""
        return date_str, title_str
