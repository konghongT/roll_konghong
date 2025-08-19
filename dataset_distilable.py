from distilabel.llms import OpenAILLM
#from distilabel.llms import MistralAILLM
from distilabel.steps.tasks import GenerateInstruction
from test.parser import get_args

args = get_args()
seepseek_api = OpenAILLM(
    api_key = os.getenv("DEEPSEEK_API"),
    base_url = "https://api.deepseek.com",
    **args
)
# 构建增强管道
with Pipeline().ray(num_cpus=8) as pipe:
    # 加载种子数据
    load_seeds = LoadDataFromHub(
        repo_id="open-r1/codeforces-cots",
        split="train",
        columns=["prompt"]
    )
    '''
    # 指令生成
    inst_gen = GenerateInstruction(
        llm=MistralAILLM(model="mistral-large-latest"),
        num_instructions=3, # 每个种子生成3个变体
        input_mappings={"prompt":"seed_text"},
        diversity=0.8   # 多样性控制参数
    )
    '''
    # 响应生成
    resp_gen = GenerateText(
        llm=OpenAILLM(),
        input_mappings={"instruction":"prompt"}
    )

    #load_seeds >> inst_gen >> resp_gen
    load_seeds >> resp_gen
# 运行并保存
dataset = pipe.run(
    parameters={
        "LoadDataFromHub": {"limit": 2},
        "GenerateInstruction": {
            "llm": {"max_tokens": 1024}
        }
    }
)
for data in dataset:
    print(data)
#dataset.push_to_hub("my-organization/enhanced-instructions")
