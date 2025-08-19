import os
import time
import numpy as np
from openai import OpenAI
from typing import List, Optional, Union, Dict, Tuple, Any, Callable
import functools
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from extraction_util import extract_code

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('text_similarity')

ENV_API_KEY = "DASHSCOPE_API_KEY"

DEFAULT_MODEL = "text-embedding-v3"
DEFAULT_DIMENSIONS = 1024
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 充实次数
DEFAULT_RETRY_ATTEMPTS = 3

def get_api_key(api_key: Optional[str] = None) -> str:
    if api_key:
        return api_key

    env_api_key = os.getenv(ENV_API_KEY)
    if env_api_key:
        return env_api_key
    raise ValueError(f"No API key is provided, and the {ENV_API_KEY} environment variable is not set.")

def create_client(api_key: str, base_url: str = DEFAULT_BASE_URL) -> OpenAI:

    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )

@retry(stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS), 
       wait=wait_exponential(multiplier=1, min=2, max=10))
def get_embedding(client: OpenAI, text: str, model: str, dimensions: int) -> List[float]:

    try:
        logger.debug(f"Requesting the embedding API, text length: {len(text)}")
        response = client.embeddings.create(
            model=model,
            input=text,
            dimensions=dimensions,
            encoding_format="float"
        )
        
        # 从响应中提取嵌入向量
        embedding = response.data[0].embedding
        
        return embedding
    except Exception as e:
        logger.error(f"Failed to obtain the embedding vector: {str(e)}")
        raise


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    "计算两个向量的余弦相似"
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0 if norm1 != norm2 else 1.0
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    similarity = max(-1.0, min(1.0, similarity))
    
    return float(similarity)

def calculate_text_similarity(text1: str, text2: str, api_key: Optional[str] = None, 
                             model: str = DEFAULT_MODEL, dimensions: int = DEFAULT_DIMENSIONS) -> float:
    "计算两个文本之间的相似性，使用阿里云百炼API生成文本嵌入向量"
    try:
        api_key = get_api_key(api_key)
        client = create_client(api_key)
        embedding1 = get_embedding(client, text1, model, dimensions)
        embedding2 = get_embedding(client, text2, model, dimensions)
        
        # 计算余弦相似度
        similarity = cosine_similarity(embedding1, embedding2)
        
        return similarity
    except Exception as e:
        logger.error(f"Failed to calculate text similarity: {str(e)}")
        return 0.0


def sts_score(text1, text2, api_key=None):

    try:
        similarity = calculate_text_similarity(text1, text2, api_key)
        #logger.info(f"文本相似度分数: {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"error: {e}")
        return 0.0

# def extract_code(text):
#     import re
#     code_blocks = re.findall(r'```(?:python)?\n(.+?)\n```', text, re.DOTALL)
#     if code_blocks:
#         return '\n'.join(code_blocks)
#     return text

# def code_sts_score(text1, text2, api_key=None):
#     """计算两段代码之间的语义相似度分数
    
#     Args:
#         text1 (str): 第一段代码文本
#         text2 (str): 第二段代码文本
#         api_key (str, optional): API密钥，默认为None，将从环境变量获取
        
#     Returns:
#         float: 相似度分数，范围为0到1
#     """

#     code1 = extract_code(text1)
#     code2 = extract_code(text2)
    
#     # 计算相似度
#     return sts_score(code1, code2, api_key)

if __name__ == "__main__":
    # 示例文本
    text1 = "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买"
    text2 = "中国科学院计算所前瞻实验室"
    text3 = "sshih1ihishi1"

    api_key = "sk-155ff93dfcc141efb5e41dc416ea7d47" 
    
 
    similarity = calculate_text_similarity(text1, text2, api_key=api_key)
    print(f"两个文本之间的相似度: {similarity:.4f}")