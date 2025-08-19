import json


def get_algorithm_template(question):
    '''
    prompt = "You have a solid theoretical foundation in data structures and algorithms, and are proficient in the principles and implementation details of various algorithms including dynamic programming, greedy algorithms, divide and conquer algorithms, backtracking algorithms, depth-first search, and breadth-first search.You will be given a question (problem specification) and need to accurately judge the type of algorithm and output.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt +=f"Put your answer within a single block:\n"
    prompt += f"```algorithm\n# YOUR ALGORITHM HERE\n```\n\n"
    '''
    prompt = (
        "You are an expert in algorithms and can quickly identify all possible algorithmic approaches for a given problem. "
        "Your task is to analyze the problem and return **all potentially applicable algorithm types** in the format below, without any additional explanation.\n\n"
        f"Question: {question}\n\n"
        "Output format (strictly follow this, no extra text):\n"
        "```algorithm\n<algorithm_type1>、<algorithm_type2>、<algorithm_type3>\n```\n\n"
        "Possible algorithm types include (but are not limited to):\n"
        "- Dynamic programming\n"
        "- Greedy algorithms\n"
        "- Divide and conquer\n"
        "- Backtracking\n"
        "- Depth-first search (DFS)\n"
        "- Breadth-first search (BFS)\n"
        "- Binary search\n"
        "- Two pointers\n"
        "- Sliding window\n"
        "- Union-Find (Disjoint Set Union)\n"
        "- Topological sort\n"
        "- Dijkstra's algorithm\n"
        "- Bellman-Ford\n"
        "- Floyd-Warshall\n"
        "- Minimum spanning tree (Prim/Kruskal)\n"
        "- Knapsack problem\n"
        "- Bit manipulation\n"
        "- Math/number theory\n"
        "- Simulation/brute force\n"
        "- Other (specify if none of the above fit)\n\n"
        "Example 1:\n"
        "Question: Find the shortest path in an unweighted graph.\n"
        "```algorithm\nBreadth-first search (BFS)\n```\n\n"
        "Example 2:\n"
        "Question: Find all subsets of a set.\n"
        "```algorithm\nBacktracking、Bit manipulation\n```\n\n"
        "Example 3:\n"
        "Question: Find the kth largest element in an array.\n"
        "```algorithm\nQuickselect、Heap、Sorting\n```"
    )
    return prompt

def generate_prompt(problem_data):
    prompt = (
        "You will be given a competitive programming problem.\n"
        "Analyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. Please reason step by step about your solution approach, then provide a complete implementation in Python 3 that is thoroughly optimized for both speed and memory usage.\n\n"
        "Your solution must read input from standard input (input()), write output to standard output (print()).\n"
        "Do not include any debug prints or additional output.\n\n"
        "Put your final solution within a single code block:\n"
        "```python\n<your code here>\n```\n\n"
        f"# Problem\n\n{problem_data['description']}\n\n"
        f"## Constraints\nTime limit per test: {problem_data['time_limit']} seconds\n"
        f"Memory limit per test: {problem_data['memory_limit']} megabytes\n\n"
        f"## Input Format\n{problem_data['input_format']}\n\n"
        f"## Output Format\n{problem_data['output_format']}\n\n"
        "## Examples\n"
    )
    
    for example in problem_data['examples']:
        prompt += f'```input\n{example["input"]}\n```\n```output\n{example["output"]}\n```\n-----\n'
    
    if 'note' in problem_data and problem_data['note']:
        prompt += f"\n## Note\n{problem_data['note']}\n"
    
    return prompt


