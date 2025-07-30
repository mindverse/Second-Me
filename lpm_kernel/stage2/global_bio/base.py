import json
import os
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.base.database_operate import store_bio, store_version
from lpm_kernel.configs.logging import get_train_process_logger
from lpm_kernel.models.l1 import L1Version
from .prompts import Prompts
from .utils import Bio, ConfidenceLevel

logger = get_train_process_logger()
from lpm_kernel.common.repository.database_session import DatabaseSession

# 加载环境变量
load_dotenv(override=True)

GLOBAL_BIO_V2_DEFAULT_PROMPT = {
    "global_bio_v2": Prompts.GLOBAL_BIO_INITIAL_SYSTEM_PROMPT,
    "common_perspective_shift": Prompts.COMMON_PERSPECTIVE_SHIFT_SYSTEM_PROMPT
}

IMPORTANCE_TO_CONFIDENCE = {
    1: ConfidenceLevel.VERY_LOW,
    2: ConfidenceLevel.LOW,
    3: ConfidenceLevel.MEDIUM,
    4: ConfidenceLevel.HIGH,
    5: ConfidenceLevel.VERY_HIGH
}


class GlobalBioV2():
    _input_keys: List = ["oldGlobalBio", "preferredLanguage"]
    _output_keys: List = ["globalBio"]
    _must_keys: List = ["oldGlobalBio"]

    model_params = {
        "temperature": 0,
        "max_tokens": 5000,
        "top_p": 0,
        "frequency_penalty": 0,
        "seed": 42,
        "presence_penalty": 0,
        "request_timeout": 45,
        "max_retries": 1
    }

    @property
    def _api_type(self):
        return "global_bio_v2"

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
        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()
        self.llms: Dict[str, openai.OpenAI] = {
            k: openai.OpenAI(
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
            for k, _ in GLOBAL_BIO_V2_DEFAULT_PROMPT.items()
        }
        self.model_name = self.user_llm_config.chat_model_name
        self.class_name = self.__class__.__name__

        self.model_params.update(**kwargs)
        self.langfuse_dict = {
            k: self._get_langfuse_prompt(k, v) for k, v in GLOBAL_BIO_V2_DEFAULT_PROMPT.items()
        }

    # 调试代码 使用本地的prompt
    def _get_langfuse_prompt(self, prompt_key, default_prompt) -> Dict[str, Any]:
        # 直接使用默认提示词，不再尝试从 langfuse 获取
        system_prompt = default_prompt
        logger.info(f"Using local default prompt for {prompt_key}")
        return {
            "lf_prompt": None,
            "system_prompt": system_prompt
        }

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prefer_language = inputs.get("preferredLanguage", "English/English")
        old_global_bio = inputs.get("oldGlobalBio", {})

        global_bio = Bio(**old_global_bio)
        global_bio = self.global_bio_generate(global_bio, prefer_language)

        global_bio = {
            "globalBio": global_bio.to_json()
        }
        result = {}
        result["globalBio"] = global_bio.get("globalBio").get("content")
        return result

    def global_bio_generate(self, raw_bio: Bio, prefer_language: str = None) -> Bio:
        user_prompt = raw_bio.to_str()
        _langfuse_prompt = self.langfuse_dict["global_bio_v2"]
        global_bio_message = self.build_message(_langfuse_prompt["system_prompt"],
                                                user_prompt,
                                                language=prefer_language)
        llm = self.llms["global_bio_v2"]
        answer = llm.chat.completions.create(
            model=self.model_name,
            messages=global_bio_message,
        )
        third_perspective_result = answer.choices[0].message.content
        raw_bio.summary_third_view = third_perspective_result
        raw_bio.content_third_view = raw_bio.complete_content()
        raw_bio = self.shift_perspective(raw_bio, prefer_language)
        raw_bio = self.assign_confidence_level(raw_bio)
        return raw_bio

    def shift_perspective(self, bio: Bio, prefer_language: str = None) -> Bio:
        system_prompt = Prompts.COMMON_PERSPECTIVE_SHIFT_SYSTEM_PROMPT
        user_prompt = bio.summary_third_view

        shift_perspective_message = self.build_message(system_prompt, user_prompt, language=prefer_language)
        llm = self.llms["common_perspective_shift"]
        answer = llm.chat.completions.create(
            model=self.model_name,
            messages=shift_perspective_message,
        )
        second_perspective_result = answer.choices[0].message.content

        bio.summary_second_view = second_perspective_result
        bio.content_second_view = bio.complete_content(second_view=True)
        return bio

    def assign_confidence_level(self, raw_bio: Bio) -> Bio:
        level_n, interest_n = len(IMPORTANCE_TO_CONFIDENCE), len(raw_bio.shades_list)
        level_list = [IMPORTANCE_TO_CONFIDENCE[level_n - int(i / interest_n * level_n)] for i in range(interest_n)]
        for shade, level in zip(raw_bio.shades_list, level_list):
            shade.confidence_level = level
        return raw_bio

    def build_message(self, system_prompt, user_prompt, language=None):
        raw_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if language:
            raw_message.append(
                {"role": "system", "content": Prompts.PREFER_LANGUAGE_SYSTEM_PROMPT.format(language=language)})
        return raw_message


if __name__ == "__main__":
    global_bio_v2 = GlobalBioV2()
    with open("resources/data/stage2/shades/shades_content.json", "r") as f:
        shades = json.load(f)
    shades = shades["shades"]

    inputs = {
        "oldGlobalBio": {
            "content": "",
            "summary": "",
            "shadesList": shades
        },
        "preferredLanguage": "简体中文/Simplified Chinese"
    }
    result = global_bio_v2._call(inputs)
    with DatabaseSession.session() as session:
        new_version = session.query(L1Version).order_by(
            L1Version.version.desc()).first().version + 1 if session.query(L1Version).order_by(
            L1Version.version.desc()).first() else 1
        store_version(session, new_version)
        store_bio(session, new_version, result["globalBio"])
    os.makedirs("resources/data/stage2/global_bio", exist_ok=True)
    with open("resources/data/stage2/global_bio/global_bio.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
