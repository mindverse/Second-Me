import os
import sys
import json
import random

import concurrent.futures
import openai
from tqdm import tqdm
from prompt import JUDGE_COT_PROMPT, JUDGE_PROMPT, MEMORY_COT_PROMPT, MEMORY_PROMPT, CONTEXT_COT_PROMPT, CONTEXT_PROMPT, CONTEXT_ENHANCE_EVAL_SYS, JUDGE_EVAL_SYS, MEMORY_EVAL_SYS, USR

from utils import OPENAI_API_KEY,OPENAI_BASE_URL,Global_Bio

from openai import OpenAI
from pydantic import BaseModel
from collections import defaultdict

# COT mode
IS_COT = False
# USER NAME SETTING
USER_NAME = "Felix Tao"
# prefered language
preference_language = "English"

class Rate(BaseModel):
    comparison: str
    detailed_analysis: str

class DPOData:
    """Generates DPO data for training language models.
    
    This class is responsible for creating diverse training data based on user notes,
    entities, and configurations. It leverages LLMs to generate questions and answers.
    """
    
    def __init__(self, input_path, output_dir,preference_language: str):
        """Initialize the DPO data generator.
        
        Args:
            input_path: Path to the input JSON file.
            output_dir: Directory to save the output JSON files.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        
        # Use the API key and base URL from utils.py
        self.model_name = "gpt-4o"  # Set your model name here
        if OPENAI_BASE_URL:
            self.client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
            )
        else:
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
            ) 
        self.preference_language = preference_language

    def load_and_sample_data(self, sample_fraction=0.1):
        """
        Load data from a JSON file and sample a fraction of it.

        :param sample_fraction: Fraction of data to sample.
        :return: Sampled data.
        """
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        sampled_data = random.sample(data, int(len(data) * sample_fraction))
        
        chat_messages = self.create_chat_data(sampled_data)
        
        return chat_messages
    
    # build messages in chat format
    def create_chat_data(self,data):
        def preprocess(sample, is_cot=False):
            if sample.get('assistant') is None and sample.get('enhanced_request') is not None:
                user_message = f"{USER_NAME}'s request is " + sample['user_request']
                infer_prompt = CONTEXT_COT_PROMPT.format(user_name=USER_NAME) if is_cot else CONTEXT_PROMPT.format(user_name=USER_NAME)
                messages = [
                    {"role": "system", "content": infer_prompt},
                    {"role": "user", "content": user_message},
                    # {"role": "assistant", "content": sample['enhanced_request'].strip('\n')},
                ]
                return [{"messages": messages,"user":user_message,"label":sample['enhanced_request'].strip('\n'),"eval_prompt":CONTEXT_ENHANCE_EVAL_SYS,"infer_prompt":infer_prompt}]
            if sample.get('assistant') is None and sample.get('user_feedback') is not None:
                user_message = f"{USER_NAME}'s request is " + sample['user_request'] + "\n" + "The response of expert is " + sample['expert_response']
                infer_prompt = JUDGE_COT_PROMPT.format(user_name=USER_NAME) if is_cot else JUDGE_PROMPT.format(user_name=USER_NAME)
                messages = [
                    {"role": "system", "content": infer_prompt},
                    {"role": "user", "content": user_message},
                    # {"role": "assistant", "content": sample['user_feedback'].strip('\n')},
                ]
                global_bio = Global_Bio
                return [{"messages": messages,"user":user_message,"label":sample['user_feedback'].strip('\n'),"eval_prompt":JUDGE_EVAL_SYS.format(global_bio=global_bio),"infer_prompt":infer_prompt}]
            sample['assistant'] = sample['assistant'].strip('\n')
            if sample.get('timestamp') is not None and sample.get('is_timeqa', None) is None:
                # messages1 = [
                #     {"role": "system", "content": "You are a helpful assistant.\n\nThe current date is " + sample['timestamp'][:10]},
                #     {"role": "user", "content": "<|ME|>" + sample['user']},
                #     {"role": "assistant", "content": sample['assistant']},
                # ]
                messages2 = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "<|ME|>" + sample['user']},
                    {"role": "assistant", "content": sample['assistant']},
                ]
                if 'None' in sample['assistant']:
                    return []
                # return [{"content": tokenizer.apply_chat_template(messages1, tokenize=False)}, 
                #         {"content": tokenizer.apply_chat_template(messages2, tokenize=False)}]
                return [{"messages": messages2}]
            elif sample.get('is_timeqa', None) is not None:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant.\n\nToday’s date is " + sample['timestamp']},
                    {"role": "user", "content": "<|ME|>" + sample['user']},
                    {"role": "assistant", "content": sample['assistant']},
                ]
                if 'None' in sample['assistant']:
                    return []
                return {"messages": messages}
            elif sample.get('exact_day', None) is not None:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "<|ME|>" + sample['user']},
                    {"role": "assistant", "content": sample['assistant']},
                ]
                return [{"messages": messages}]
            else:
                infer_prompt = MEMORY_COT_PROMPT.format(user_name=USER_NAME) if is_cot else MEMORY_PROMPT.format(user_name=USER_NAME)
                messages = [
                    {"role": "system", "content": infer_prompt},
                    {"role": "user", "content": sample['user']},
                    # {"role": "assistant", "content": sample['assistant']},
                ]
                if 'None' in sample['assistant']:
                    return []
                return [{"messages": messages,"user":sample['user'],"label":sample['assistant'],"eval_prompt":MEMORY_EVAL_SYS,"infer_prompt":infer_prompt}]

        res_dataset = []

        for case in data:
            res_dataset.extend(preprocess(case, IS_COT))

        # res = Dataset.from_list(res_dataset)
        # print(f"**************Dataset contains {res.num_rows} elements.**************")
        print(f"**************Dataset contains {len(res_dataset)} elements.**************")
        # print(res_dataset[:2])
        return res_dataset

    def generate_all_traces(self, processed_data):
        """
        Generate traces for all processed data.

        :param processed_data: Preprocessed data.
        :return: All generated traces.
        """
        all_traces=[]
        for instance  in tqdm(processed_data, desc=f"Generating traces"):
                
                message = instance.get("messages",[])
                # generate trace for each message
                traces = self.generate_traces(message,3)
                
                # attach traces to each instance
                instance_with_traces = {
                    "user": instance["user"],
                    "label": instance["label"],
                    "traces": traces,
                    "eval_prompt": instance["eval_prompt"],
                    "infer_prompt":instance["infer_prompt"]
                }
                print(instance_with_traces)
                all_traces.append(instance_with_traces)
                
        return all_traces
    def generate_traces(self,messages, nums_traces=3):
        """
        Generate traces using the OpenAI API.
        llama.cpp can serve as http server
        so we can use it as a openai compatible endpoint.

        :param messages: List of messages to send to the API.
        :param nums_traces: The number of traces to generate.
        :return: List of traces.
        """
        traces = []
        client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="key")
        
        for _ in range(nums_traces):
            response = client.chat.completions.create(
                model="",
                messages=messages,
                stream=False,
                temperature=0.7,
                max_tokens=2048,
                top_p=1.0  # Adjust top_p as needed
            )
            traces.append(response.choices[0].message.content)
        
        return traces
    def compare_eval(self, instances):
        """
        Compare evaluations and determine chosen and rejected responses.

        :param instances: Instances with traces.
        :return: Instances with chosen and rejected responses.
        """
        all_eval_messages = []
        
        # compare traces
        for ins in instances:
            traces = ins["traces"]
            if len(traces) < 3:
                raise ValueError("Each instance must have exactly 3 traces.")
            
            # build messages
            eval_messages = [
                [{
                    "role": "system",
                    "content": ins["eval_prompt"]
                }, {
                    "role": "user",
                    "content": USR.format(
                        user_input=ins["user"], 
                        model_answer_1=traces[0],  # trace1 vs trace2
                        model_answer_2=traces[1],
                        reference_info=ins["label"]
                    )
                }],
                [{
                    "role": "system",
                    "content": ins["eval_prompt"]
                }, {
                    "role": "user",
                    "content": USR.format(
                        user_input=ins["user"], 
                        model_answer_1=traces[0],  # trace1 vs trace3
                        model_answer_2=traces[2],
                        reference_info=ins["label"]
                    )
                }],
                [{
                    "role": "system",
                    "content": ins["eval_prompt"]
                }, {
                    "role": "user",
                    "content": USR.format(
                        user_input=ins["user"], 
                        model_answer_1=traces[1],  # trace2 vs trace3
                        model_answer_2=traces[2],
                        reference_info=ins["label"]
                    )
                }]
            ]
            all_eval_messages.extend(eval_messages)

        # access eval rs
        trying_limit = len(all_eval_messages)
        eval_results = self.multi_process_request(all_eval_messages[:trying_limit], 10, self.process_request_structered, Rate)

        # group results
        for ins_idx, ins in enumerate(instances):
            start_idx = ins_idx * 3
            end_idx = start_idx + 3
            instance_eval_results = eval_results[start_idx:end_idx]
            
            print(instance_eval_results)

            # get rejected responses and chosen responses
            tmp_comparisons = []
            for result in instance_eval_results:
                if type(result) == Rate:
                    tmp_comparisons.append(result.comparison)
                else:
                    tmp_comparisons.append('tie')
            
            chosen_response, rejected_response, detailed_analysis = self.compare_traces(
                traces=ins["traces"],
                eval_results=tmp_comparisons
            )

            # attach results to each instance
            ins["chosen_response"] = chosen_response
            ins["rejected_response"] = rejected_response
            ins["detailed_analysis"] = detailed_analysis

        # print the results
        print(f"choose_response: {chosen_response}")
        print(f"rejected_response: {rejected_response}")

        
        return instances
    
    def compare_traces(self,traces, eval_results):
        """
        Compare three traces to determine the best and worst trace.
        :param traces: A list of three traces [trace1, trace2, trace3].
        :param eval_results: The results of pairwise comparisons, formatted as [{"comparison": "first win"/"tie"/"second win", "detailed_analysis": "..."}, ...].
        :return: chosen_response, rejected_response, detailed_analysis
        """
        # initialization
        win_loss = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
        
        # comparison res
        comparisons = [
            (0, 1, eval_results[0]),  # trace1 vs trace2
            (0, 2, eval_results[1]),  # trace1 vs trace3
            (1, 2, eval_results[2]),  # trace2 vs trace3
        ]
        
        # calculate wins, losses, and ties
        for i, j, result in comparisons:
            if result == "first win":
                win_loss[traces[i]]["wins"] += 1
                win_loss[traces[j]]["losses"] += 1
            elif result == "second win":
                win_loss[traces[j]]["wins"] += 1
                win_loss[traces[i]]["losses"] += 1
            elif result == "tie":
                win_loss[traces[i]]["ties"] += 1
                win_loss[traces[j]]["ties"] += 1
            else:
                raise ValueError(f"Invalid comparison result: {result}")
        
        # calculate win rate for each trace
        def calculate_win_rate(trace):
            total = win_loss[trace]["wins"] + win_loss[trace]["losses"] + win_loss[trace]["ties"]
            if total == 0:
                return 0
            return win_loss[trace]["wins"] / total
        
        # select chosen_response and rejected_response
        sorted_traces = sorted(traces, key=lambda x: calculate_win_rate(x), reverse=True)
        chosen_response = sorted_traces[0]
        rejected_response = sorted_traces[-1]
        
        # get detailed analysis
        detailed_analysis = f"Chosen Response: {chosen_response} (Win Rate: {calculate_win_rate(chosen_response):.2f})\n"
        detailed_analysis += f"Rejected Response: {rejected_response} (Win Rate: {calculate_win_rate(rejected_response):.2f})\n"
        detailed_analysis += "Comparison Details:\n"
        for trace in traces:
            detailed_analysis += f"{trace}: Wins={win_loss[trace]['wins']}, Losses={win_loss[trace]['losses']}, Ties={win_loss[trace]['ties']}\n"
        
        return chosen_response, rejected_response, detailed_analysis
    
    def process_request_structered(self,messages, format_class):
        try:
            model = self.model_name
            completion = self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=format_class,
            )
            message = completion.choices[0].message
            if message.parsed:
                print(f"model answer:{message.parsed}")
                return message.parsed
            else:
                return message.refusal
        except Exception as e:
            return f"Error occurred: {str(e)}"
    
    def multi_process_request(self,all_messages, max_workers, func, structure=None):
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(all_messages))) as executor:
            futures = [(i, executor.submit(func, messages, structure)) if structure is not None else (i, executor.submit(func, messages)) for i, messages in enumerate(all_messages)]
            results = [None] * len(all_messages) 

            for i, future in tqdm(futures):
                try:
                    result = future.result()
                    results[i] = result
                except Exception as e:
                    results[i] = f"Raise ERROR: {e} WHEN GENERATE RESPONSE"

        return results

    def prepare_dpo_datasets(self,sampled_data):
        """
        Prepare full and direct training versions of the DPO dataset.

        :param sampled_data: Sampled data from the input JSON file.
        :return: Full version and direct training version of the dataset.
        """
        full_version = []
        direct_training_version = []

        for item in sampled_data:
            full_version.append(item)
            direct_training_version.append({
                'prompt': {"system":item['infer_prompt'],"user":item['user']},
                'chosen': item['chosen_response'],
                'rejected': item['rejected_response']
            })

        return full_version, direct_training_version

    def save_datasets(self,output_dir, full_version, direct_training_version):
        """
        Save the full and direct training versions of the DPO dataset to JSON files.

        :param output_dir: Directory to save the output JSON files.
        :param full_version: Full version of the dataset.
        :param direct_training_version: Direct training version of the dataset.
        """
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'dpo_full.json'), 'w', encoding='utf-8') as file:
            json.dump(full_version, file, ensure_ascii=False, indent=4)

        with open(os.path.join(output_dir, 'dpo_direct.json'), 'w', encoding='utf-8') as file:
            json.dump(direct_training_version, file, ensure_ascii=False, indent=4)

    def run(self):
        """
        Main function to orchestrate the workflow.
        """
        # Load and sample data, combine system prompt for each task.
        sampled_data = self.load_and_sample_data()
        
        # Generate traces for all cases
        all_traces = self.generate_all_traces(sampled_data)
        
        # Compare eval -> get chosen and rejected responses
        compare_res = self.compare_eval(all_traces)

        # Prepare DPO datasets
        full_version, direct_training_version = self.prepare_dpo_datasets(compare_res)

        # Save datasets
        self.save_datasets(self.output_dir,full_version, direct_training_version)

        print(f"Sampled data saved to {self.output_dir}")

# Example usage
if __name__ == "__main__":
    input_path = 'resources/data/merged.json'
    output_dir = 'resources/L2/data/dpo/'
    dpo_data = DPOData(input_path, output_dir,preference_language)
    dpo_data.run()