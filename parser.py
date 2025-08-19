import os
import torch
import argparse
import multiprocessing

#与deepseek交互的各个参数
def get_args():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-reasoner",
        help="Name of the model to use matching `cot_distill.py`",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="trust_remote_code option used in huggingface models",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1, help="Temperature for sampling"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument(
        "--max_tokens", type=int, default=8192, help="Max tokens for sampling"
    )
    parser.add_argument(
        "--openai_timeout", type=int, default=90, help="Timeout for requests to OpenAI"
    )
    parser.add_argument(
        "--stop",
        default="```",
        type=str,
        help="Stop token (use `,` to separate multiple tokens)",
    )
    
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=10, 
        help="要处理的样本数量"
    )
    args = parser.parse_args()
    return args


def test():
    args = get_args()
    print(args)


if __name__ == "__main__":
    test()