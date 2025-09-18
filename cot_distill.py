import requests
import os
import json
import time
import logging
import re
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from parser import get_args
from sts import calculate_text_similarity
from extraction_util import (
    extract_code,
    extract_algorithm,
)   
from generate_prompt import get_algorithm_template

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import openai
from openai import OpenAI

load_dotenv()

class DistillRunner():
    def __init__(self, args):
        self.client_kwargs = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "frequency_penalty": 1,
            "presence_penalty": 1,
            "n": 1,  # Number of answers to generate
            "timeout": args.openai_timeout,
        }
        self.client = None
        self.max_workers = getattr(args, 'max_workers', 40)  # 并行
        self.batch_size = getattr(args, 'batch_size', 100)   # 批处理
        self.processed_results = []

    def query_deepseek(self, prompt, max_retries=3, retry_delay=30):
        """
        Call DeepSeek API and merge reasoning process with results
    
        """
        assert isinstance(prompt, list), "prompt must be a list"
        
        # Get API key from environment
        #api_key =  "use_yourownapikey"
        #api_key = os.getenv("DEEPSEEK_API_KEY") 
        api_key = "yourownapikey"
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        
        if self.client is None:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        
        retries = 0
        while retries <= max_retries:
            try:
                response = self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                
                reasoning_content = response.choices[0].message.reasoning_content
                content = response.choices[0].message.content
                combined = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
            
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                return {
                    "is_api": "True",
                    "generation": combined,
                    "finish_reason": response.choices[0].finish_reason,
                    "api_metadata": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "prompt_tokens_details": None,
                        "total_tokens": total_tokens
                    }
                }
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                retries += 1
                logger.warning(f"API call failed (attempt {retries}/{max_retries}): {repr(e)}")
                if retries <= max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    sleep(retry_delay)
                else:
                    logger.error(f"Max retries reached, API call failed")
                    raise
            except Exception as e:
                logger.error(f"Model execution failed, prompt: {prompt}")
                logger.error(f"Exception: {repr(e)}")
                raise

    def get_cot_dataset(self):
        try:
            from datasets import load_dataset
            ds = load_dataset("open-r1/codeforces-cots", "solutions_py", split="train")
            logger.info(f"Dataset loaded successfully, contains {len(ds)} records")
            return ds
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def build_chat_messages(self, prompt):
        """
        prompt
        """

        chat_messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        return chat_messages

    def create_message_format(self, prompt, api_generation):
        messages = [
            {
                "content": prompt,
                "role": "user"
            },
            {
                "content": api_generation,
                "role": "assistant"
            }
        ]
        return messages

    def save_file(self, results, output_dir="./test/data"):

        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"test_results_{timestamp}.json")
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4) 

        logger.info(f"File saved to: {output_path}")

    def process_single_sample(self, data_item):
        """
        Process a single data sample
        """
        try:
            result_item = {k: v for k, v in data_item.items()}
            # Get API generation
            # build chat prompt difficulty
            #print(data_item["api_metadata"])
            format_prompt = self.build_chat_messages(data_item["prompt"])
            api_result = self.query_deepseek(format_prompt)

            '''
            # 再次调用api生成题目所用算法类型
            algorithm_prompt = self.build_chat_messages(get_algorithm_template(data_item["description"]))
            algorithm_result = self.query_deepseek(algorithm_prompt)
            data_item["algorithm"] = extract_algorithm(algorithm_result["generation"])
            api_result["algorithm"] = extract_algorithm(algorithm_result["generation"])
            '''
            
            
            result_item.update(api_result)
            
            result_item["messages"] = self.create_message_format(
                data_item["prompt"], 
                api_result["generation"]
            )

            data_item["is_api"] = "False"
            result_item["is_api"] = api_result["is_api"]

            data_item["generation_code"] = extract_code(data_item["generation"])
            result_item["generation_code"] = extract_code(api_result["generation"])
            
            '''
            # Calculate similarity score
            #api_key = "" 
            # api_key from aliyunbailian
            api_key = os.getenv("DASHSCOPE_API_KEY")
            score = calculate_text_similarity(
                data_item["generation"],
                api_result["generation"],
                api_key=api_key
            )
            score_details = {"dataset_generation": data_item["generation"], "api_generation": api_result["generation"], "similarity_score": score}
            result_item["similarity_score"] = score_details
            '''
            score_details = {"dataset_generation": data_item["generation"], "api_generation": api_result["generation"]}
            result_item["similarity_score"] = score_details
            #logger.info(f"Successfully processed sample ID: {data_item.get('id', 'unknown')}")
            
            return (data_item, result_item)
        except Exception as e:
            logger.error(f"Error processing sample: {repr(e)}")
            return None

    def process_batch(self, batch_items):
        results = []
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_single_sample, item) 
                      for item in batch_items]
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        data_item, result_item = result
                        results.append(data_item)
                        results.append(result_item)
                except Exception as e:
                    logger.error(f"Error getting result: {repr(e)}")
        
        return results

    '''
    def process_dataset_parallel(self, num_samples=None):

        dataset = self.get_cot_dataset()
        total_samples = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        
        logger.info(f"Starting parallel processing of {total_samples} samples, max workers: {self.max_workers}, batch size: {self.batch_size}")
        
        self.processed_results = []
        
        # Batch processing
        for start_idx in range(0, total_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_samples)
            logger.info(f"Processing batch {start_idx//self.batch_size + 1}, items {start_idx}-{end_idx-1}")
            batch_items = [dataset[i] for i in range(start_idx, end_idx)]
            # Process current batch
            batch_results = self.process_batch(batch_items)
            self.processed_results.extend(batch_results)
            
            logger.info(f"Processed {len(self.processed_results)/2}/{total_samples} samples")
            

            if len(self.processed_results) % (self.batch_size*4) == 0:
                self.save_file(self.processed_results, output_dir="./test/data/intermediate")
                logger.info(f"Intermediate results saved, progress: {len(self.processed_results)/2}/{total_samples}")

        self.save_file(self.processed_results)
        
        logger.info(f"Processing complete, total processed samples: {len(self.processed_results)/2}")
        return self.processed_results
    '''

    def process_dataset_parallel(self, start_idx=0, end_idx=None):
        """
        处理数据集中指定区间的样本
        
        Args:
            start_idx: 起始索引（包含）
            end_idx: 结束索引（不包含），如果为None则处理到数据集末尾
        """
        dataset = self.get_cot_dataset()
        total_samples = len(dataset)
        
        # 设置默认结束位置
        if end_idx is None:
            end_idx = total_samples
        else:
            end_idx = min(end_idx, total_samples)
        
        # 验证索引有效性
        if start_idx < 0 or start_idx >= end_idx:
            raise ValueError(f"Invalid index range: start_idx={start_idx}, end_idx={end_idx}")
        
        process_range = end_idx - start_idx
        logger.info(f"Starting parallel processing of samples {start_idx}-{end_idx-1} (total {process_range} samples), "
                f"max workers: {self.max_workers}, batch size: {self.batch_size}")
        
        self.processed_results = []
        
        # Batch processing
        for batch_start in range(start_idx, end_idx, self.batch_size):
            batch_end = min(batch_start + self.batch_size, end_idx)
            logger.info(f"Processing batch {(batch_start-start_idx)//self.batch_size + 1}, "
                    f"items {batch_start}-{batch_end-1}")
            
            batch_items = [dataset[i] for i in range(batch_start, batch_end)]
            batch_results = self.process_batch(batch_items)
            self.processed_results.extend(batch_results)
            
            logger.info(f"Processed {len(self.processed_results)/2}/{process_range} samples")
            
            if len(self.processed_results) % (self.batch_size*4) == 0:
                self.save_file(self.processed_results, output_dir=f"./test/data/intermediate")
                logger.info(f"Intermediate results saved, progress: {len(self.processed_results)/2}/{process_range}")

        self.save_file(self.processed_results, output_dir=f"./test/data_{start_idx}-{end_idx}")
        
        logger.info(f"Processing complete, total processed samples: {len(self.processed_results)/2}")
        return self.processed_results

if __name__ == "__main__":
    args = get_args()
    # e.g., python script.py --max_workers 10 --batch_size 20 --num_samples 100
    runner = DistillRunner(args)
    
    # Process specified number of samples
    #num_samples = getattr(args, 'num_samples', 5)
    results = runner.process_dataset_parallel(start_idx=7000, end_idx=9000)



    
