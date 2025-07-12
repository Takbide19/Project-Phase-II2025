app.py

# import torch
# import gradio as gr
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import os
#
# MODEL_PATH = r"F:/shruti/saved_codellema"
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# if not os.path.exists(MODEL_PATH):
#     raise RuntimeError(f"Model path not found: {MODEL_PATH}. Please run download_and_save_model.py first.")
#
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
# model.to(device)
#
# def detect_and_fix_bugs(code_snippet):
#     prompt = f"""
# ### Buggy Code:
# {code_snippet}
#
# ### Task:
# - Identify all bugs in the code.
# - For each bug, provide:
#     - Line number
#     - Error type (e.g., SyntaxError, NameError)
#     - Explanation
# - Then provide the corrected version of the full code.
#
# ### Output Format:
# Errors:
# 1. Line: <line number>
#    Error Type: <type>
#    Issue: <explanation>
#
# ...
#
# Fixed Code:
# <corrected code here>
# """
#
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     output = model.generate(
#         **inputs,
#         max_new_tokens=512,
#         temperature=0.2,
#         do_sample=False,
#         pad_token_id=tokenizer.eos_token_id,
#     )
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#
#     # Extract fixed code only
#     fixed_code_lines = []
#     extract = False
#     for line in response.splitlines():
#         if line.strip() == "Fixed Code:":
#             extract = True
#             continue
#         if extract:
#             fixed_code_lines.append(line)
#
#     fixed_code = "\n".join(fixed_code_lines).strip()
#     return fixed_code if fixed_code else response
#
# interface = gr.Interface(
#     fn=detect_and_fix_bugs,
#     inputs=gr.Textbox(
#         label="Enter your Python code",
#         lines=12,
#         placeholder="Paste your buggy Python code here...",
#     ),
#     outputs=gr.Textbox(label="Analysis & Fixed Code"),
#     title="Multi-Bug Python Fixer",
#     description="Detects multiple bugs in Python code and provides corrected code.",
# )
#
# if _name_ == "_main_":
#     interface.launch(share=False)
#
#
#
#
#
#



import torch
# import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODEL_PATH = r"F:/shruti/saved_codellema"
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model path not found: {MODEL_PATH}. Please run download_and_save_model.py first.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
model.to(device)

def detect_and_fix_bugs(code_snippet):
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

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=768,
        temperature=0.2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# interface = gr.Interface(
#     fn=detect_and_fix_bugs,
#     inputs=gr.Textbox(
#         label="Enter your Python code",
#         lines=12,
#         placeholder="Paste your buggy Python code here...",
#     ),
#     outputs=gr.Textbox(label="Errors & Fixed Code"),
#     title="Multi-Bug Python Fixer",
#     description="Detects multiple bugs in Python code and provides corrected code.",
# )
#
# if _name_ == "_main_":
#     interface.launch(share=False)
