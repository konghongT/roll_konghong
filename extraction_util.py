
#把代码从文本中分离出来
def extract_code(model_output: str):
    outputlines = model_output.split("\n")
    # first try to extract ```python if not then try ```
    indexlines = [
        i
        for i, line in enumerate(outputlines)
        if "```python" in line or "```Python" in line
    ]
    if indexlines:
        start_index = indexlines[0]
    else:
        start_index = None
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if start_index is not None:
        indexlines = [i for i in indexlines if i > start_index]
        indexlines = [start_index] + indexlines
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])

def extract_algorithm(model_output: str):
    outputlines = model_output.split("\n")
    # first try to extract ```python if not then try ```
    indexlines = [
        i
        for i, line in enumerate(outputlines)
        if "```algorithm" in line or "```Algorithm" in line
    ]
    if indexlines:
        start_index = indexlines[0]
    else:
        start_index = None
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if start_index is not None:
        indexlines = [i for i in indexlines if i > start_index]
        indexlines = [start_index] + indexlines
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])