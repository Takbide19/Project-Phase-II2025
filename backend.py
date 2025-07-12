backend.py

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import os
#
# app = FastAPI()
#
# MODEL_PATH = r"F:/shruti/saved_codellema"
#
# if not os.path.exists(MODEL_PATH):
#     raise RuntimeError(f"Model path not found: {MODEL_PATH}. Please run download_and_save_model.py first.")
#
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
# model.to("cpu")  # or "cuda" if available
#
# class CodeRequest(BaseModel):
#     code_snippet: str
#
# def detect_and_fix_bug(code_snippet: str) -> str:
#     prompt = f"""
# ### Buggy Code:
# {code_snippet}
#
# ### Task:
# 1. Identify the exact line where the error occurs.
# 2. Explain what the issue is.
# 3. Provide a corrected version of the code.
#
# ### Expected Output:
# - Line of error: (Mention the exact line)
# - Explanation: (What is wrong?)
# - Fixed Code:
# """
#     inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
#
#     try:
#         output = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             temperature=0.2,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id,
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Model generation error: {str(e)}")
#
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#
#     # Extract fixed code only
#     fixed_code_lines = []
#     extract = False
#     for line in response.splitlines():
#         if "Fixed Code:" in line:
#             extract = True
#             continue
#         if extract:
#             fixed_code_lines.append(line)
#
#     fixed_code = "\n".join(fixed_code_lines).strip()
#     return fixed_code if fixed_code else response
#
# @app.post("/analyze")
# async def analyze_code(request: CodeRequest):
#     if not request.code_snippet.strip():
#         raise HTTPException(status_code=400, detail="Code snippet cannot be empty.")
#     fixed_code = detect_and_fix_bug(request.code_snippet)
#     return {"fixed_code": fixed_code}
#
#
#



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = FastAPI()

MODEL_PATH = r"F:/shruti/saved_codellema"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model path not found: {MODEL_PATH}. Please run download_and_save_model.py first.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
model.to("cpu")  # Use "cuda" if available and appropriate

class CodeRequest(BaseModel):
    code_snippet: str

def detect_and_fix_bugs(code_snippet: str) -> str:
    prompt = f"""
### Buggy Code:
{code_snippet}

### Task:
- Identify all bugs in the code.
- For each bug, provide:
    - Line number
    - Error type (e.g., SyntaxError, NameError)
    - Explanation
- Then provide the corrected version of the full code.

### Output Format:
Errors:
1. Line: <line number>
   Error Type: <type>
   Issue: <explanation>

2. Line: <line number>
   Error Type: <type>
   Issue: <explanation>

Fixed Code:
<corrected code here>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    try:
        output = model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model generation error: {str(e)}")

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

@app.post("/analyze")
async def analyze_code(request: CodeRequest):
    if not request.code_snippet.strip():
        raise HTTPException(status_code=400, detail="Code snippet cannot be empty.")
    result = detect_and_fix_bugs(request.code_snippet)
    return {"result": result}
