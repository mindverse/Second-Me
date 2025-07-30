import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from lpm_kernel.configs.logging import get_train_process_logger
from tqdm import tqdm

from lpm_kernel.base.data import BaseData
from lpm_kernel.base.database_operate import get_latest_global_bio
from lpm_kernel.base.stage2_prompt import QUERY_GEN, ANSWER_GEN, QUERY_GEN4RAWBIO

load_dotenv(override=True)

logger = get_train_process_logger()


class BioQAData(BaseData):
    def __init__(self, stage0_path: str = "resources/data/stage2/stage0_qa.json",
                 stage1_path: str = "resources/data/stage2/stage1_qa.json",
                 stage2_path: str = "resources/data/stage2/processed/stage2_qa.json",
                 max_workers: int = 10, is_cot: bool = True, ):
        super().__init__(is_cot=is_cot, max_workers=max_workers)
        self.stage0_qa_path = stage0_path
        self.stage1_qa_path = stage1_path
        self.stage2_qa_path = stage2_path
        self.bio = ""
        self.bio_data = self._load_data()

    def _load_data(self):
        global_bio = get_latest_global_bio()
        if global_bio is None:
            logger.warning("No global biography found in database. Using empty bio.")
            self.bio = ""
            return []

        self.bio = global_bio.content_third_view
        logger.info("Raw bio content:")
        logger.info(self.bio[:500] + "..." if len(self.bio) > 500 else self.bio)  # 打印前500个字符，以便查看格式

        pattern = r'##\s+([^\n]+)\n(.*?)(?=\n##\s+|\Z)'
        matches = re.findall(pattern, self.bio, re.DOTALL)

        logger.info(f"Found {len(matches)} sections using regex")

        data = []
        for section_title, section_content in matches:
            section_content = section_content.strip()
            data.append({
                "section": section_title.strip(),
                "content": section_content
            })

        return data

    def build_stage1queries_messages(self):
        messages_list = []
        for idx, item in enumerate(self.bio_data):
            message = [
                {"role": "system", "content": QUERY_GEN},
                {"role": "user",
                 "content": f"请根据以下内容提出多个问题：\n\n章节标题：{item['section']}\n内容：{item['content']}"}
            ]
            messages_list.append(message)
        return messages_list

    def build_stage0queries_messages(self):
        messages_list = []

        sections = "\n".join([item['section'] for item in self.bio_data])
        logger.info("sections:" + sections)
        message = [
            {"role": "system", "content": QUERY_GEN4RAWBIO},
            {"role": "user", "content": f"请根据以下章节内容提出多个问题：\n\n{sections}"}
        ]
        messages_list.append(message)
        return messages_list

    def build_stage0_messages(self, queries):
        messages_list = []
        usr_prompt = "问题：{question}\n记忆：{reference}"
        for query in queries:
            formatted_prompt = usr_prompt.format(question=query, reference=self.bio)
            messages = [
                {"role": "system", "content": ANSWER_GEN},
                {"role": "user", "content": formatted_prompt}
            ]
            messages_list.append(messages)
        return messages_list

    def build_stage1_messages(self):
        messages_list = []

        usr_prompt = "问题：{question}\n记忆：{reference}"

        for b in self.bio_data:
            questions = b["qa"]
            section = b["section"]
            content = b['content']
            reference = f"{section}\n{content}"
            for q in questions:
                formatted_prompt = usr_prompt.format(question=q["question"], reference=reference)
                messages = [
                    {"role": "system", "content": ANSWER_GEN},
                    {"role": "user", "content": formatted_prompt}
                ]
                messages_list.append({"messages": messages, "query": q["question"]})

        return messages_list

    def build_responsesByQwen(self, messages_list):
        def process_request(index, messages):
            try:
                response = self.reasoning_client.chat.completions.create(
                    model=self.reasoning_model_name,
                    messages=messages,
                    extra_body={"enable_thinking": True, "thinking_budget": 300},
                    stream=True
                )
                reasoning_content = ""
                answer_content = ""
                is_answering = False

                for chunk in response:
                    if not chunk.choices:
                        logger.debug("\nUsage:")
                        logger.debug(chunk.usage)
                        continue

                    delta = chunk.choices[0].delta

                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        reasoning_content += delta.reasoning_content
                    if hasattr(delta, "content") and delta.content:
                        if not is_answering:
                            is_answering = True

                        answer_content += delta.content
                return index, answer_content, reasoning_content
            except Exception as e:
                # logger.error(f"Raise ERROR: {e} WHEN GENERATE RESPONSE")
                logger.info(f"Error in request {index}: {str(e)}", exc_info=True)
                return index, "", ""

        results = [None] * len(messages_list)  # 初始化结果列表以保持顺序
        resultsWithReason = [None] * len(messages_list)  # 初始化结果列表以保持顺序

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(messages_list))) as executor:
            futures = [executor.submit(process_request, idx, messages) for idx, messages in enumerate(messages_list)]

        for future in tqdm(futures, total=len(messages_list), desc="Generating responses"):
            index, res, reason = future.result()

            match = re.search(r"<answer>(.*?)</answer>", res, re.DOTALL)
            final_answer = match.group(1).strip() if match else res.strip()
            results[index] = final_answer
            answerWithReason = "<think>" + reason + "</think>\n\n<answer>" + final_answer + "</answer>"
            resultsWithReason[index] = answerWithReason
            logger.info(final_answer)

        return results, resultsWithReason

    def postprocess4quries(self, queries):
        new_queries = []
        for q in queries:

            start_idx = q.find('{')
            end_idx = q.rfind('}')

            if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                logger.warning("No valid JSON found in the text.")
                logger.warning(q)
                new_queries.append([])
                continue

            json_str = q[start_idx:end_idx + 1]
            try:

                question_json = json.loads(json_str)

                qs = question_json.get("queries", [])

                new_queries.append(qs)

            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode Error: {e}")
                new_queries.append([])

        return new_queries

    def run(self):

        logger.info(f"开始针对完整的bio构建qa对")

        stage0_queries_messages = self.build_stage0queries_messages()
        logger.info(stage0_queries_messages)

        if not stage0_queries_messages:
            logger.warning("No biography data available. Exiting without generating QA pairs.")
            return

        queries0, _ = self.build_responsesByQwen(stage0_queries_messages)
        queries0 = self.postprocess4quries(queries0)[0]

        stage0_messages = self.build_stage0_messages(queries0)
        _, resWithReason0 = self.build_responsesByQwen(stage0_messages)
        stage0_qa = []
        for q, r in zip(queries0, resWithReason0):
            qa = {
                "query": q,
                "answer": r
            }
            stage0_qa.append(qa)

        os.makedirs(os.path.dirname(self.stage0_qa_path), exist_ok=True)
        with open(self.stage0_qa_path, "w", encoding="utf-8") as f:
            json.dump(stage0_qa, f, ensure_ascii=False, indent=4)

        logger.info(f"✅ 针对完整BIO已成功生成 {len(stage0_qa)} 条数据，保存至 {self.stage0_qa_path}")

        logger.info(f"开始针对bio的每一个章节构建qa对")
        stage1_queries_messages = self.build_stage1queries_messages()

        queries, _ = self.build_responsesByQwen(stage1_queries_messages)
        queries = self.postprocess4quries(queries)

        updated_bio_data = []
        for item, qus in zip(self.bio_data, queries):
            updated_bio_data.append({
                "section": item["section"],
                "content": item["content"],
                "qa": [{"question": q, "answer": ""} for q in qus]
            })

        self.bio_data = updated_bio_data
        logger.info(f"✅ query 生成已完成,开始构造qa对")

        stage1_messages = self.build_stage1_messages()
        stage1_mess = []
        for m in stage1_messages:
            stage1_mess.append(m["messages"])

        _, resWithReason = self.build_responsesByQwen(stage1_mess)
        stage1_qa = []
        for mess, r in zip(stage1_messages, resWithReason):
            qa = {
                "query": mess["query"],
                "answer": r
            }
            stage1_qa.append(qa)

        os.makedirs(os.path.dirname(self.stage1_qa_path), exist_ok=True)

        with open(self.stage1_qa_path, "w", encoding="utf-8") as f:
            json.dump(stage1_qa, f, ensure_ascii=False, indent=4)

        logger.info(f"✅ 针对bio各章节的QA对已成功生成 {len(stage1_qa)} 条数据，保存至 {self.stage1_qa_path}")

        stage2_qa = stage0_qa + stage1_qa

        os.makedirs(os.path.dirname(self.stage2_qa_path), exist_ok=True)
        with open(self.stage2_qa_path, "w", encoding="utf-8") as f:
            json.dump(stage2_qa, f, ensure_ascii=False, indent=4)
        logger.info(f"已生成{len(stage2_qa)}条数据，保存至 {self.stage2_qa_path}")


if __name__ == "__main__":
    bio_qa = BioQAData()
    bio_qa.run()
