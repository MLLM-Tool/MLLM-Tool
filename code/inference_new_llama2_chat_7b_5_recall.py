import os

def modify_and_run_script(original_script, run_number):
    # 替换 answer_file 的值
    modified_script = original_script.replace(
        "\'answer_file\': \'../data/eval_llama2_chat_7b_answer_all_5.json",
        f"\'answer_file\': \'../data/eval_llama2_chat_7b_answer_all_5_v{run_number}.json"
    )

    # 将修改后的脚本写入临时文件
    temp_script_path = f'temp_script_llama2_chat_7b_5_{run_number}.py'
    with open(temp_script_path, 'w') as file:
        file.write(modified_script)

    # 执行修改后的脚本
    os.system(f'python {temp_script_path}')

# 读取原始脚本内容
file_path = 'inference_new_llama2_chat_7b_5.py'
with open(file_path, 'r') as file:
    original_script = file.read()

# 循环执行脚本 9 次
for i in range(1, 10):
    modify_and_run_script(original_script, i)
