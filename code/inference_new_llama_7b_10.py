import os
from model.openllama2 import OpenLLAMAPEFTModel
import torch
import json
from config import *
# import matplotlib.pyplot as plt
# from diffusers.utils import export_to_video
import scipy
from tqdm import tqdm


def predict(
        input,
        image_path=None,
        audio_path=None,
        video_path=None,
        max_tgt_len=200,
        top_p=10.0,
        temperature=0.1,
        # history=None,
        modality_cache=None,
        stops_id=None,
):
    if image_path == "" and audio_path == "" and video_path == "":
        # return [(input, "There is no input data provided! Please upload your data and start the conversation.")]
        print('no image, audio, video, and thermal are input')
    else:
        print(
            f'[!] image path: {image_path}\n[!] audio path: {audio_path}\n[!] video path: {video_path})')

    # prepare the prompt
    prompt_text = ''
    prompt_text += '### Human: '

    image_base_path = "../data/IT_data_ins/T+X-T_data/mm_dataset/image"
    video_base_path = "../data/IT_data_ins/T+X-T_data/mm_dataset"
    audio_base_path = "../data/IT_data_ins/T+X-T_data/mm_dataset/audio"

    modality = 'Text'
    mm_path = ''

    if image_path != "":
        image_path = os.path.join(image_base_path, image_path)
        mm_path = image_path
        modality = 'Image'
    if audio_path != "":
        audio_path = os.path.join(audio_base_path,audio_path)
        mm_path = audio_path
        modality = 'Audio'
    if video_path != "":
        video_path = os.path.join(video_base_path,video_path)
        mm_path = video_path
        modality = 'Video'

    print('prompt_text: ', input)
    print('image_path: ', image_path)
    print('audio_path: ', audio_path)
    print('video_path: ', video_path)



    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_tgt_len,
        'modality_embeds': modality_cache,
        'stops_id': stops_id,
        'modality': modality,
        'mm_path' : mm_path,
        'query': f' {input}'

    })
    return response

def get_questions(question_file):
 
    # Load questions file
    question_jsons = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            question_jsons.append(line)

    return question_jsons

def run_eval(args, question_jsons):
    ans_jsons = []
    prompt_text = ''

    with open(question_jsons, "r", encoding="utf-8") as original_file:
        data_list = json.load(original_file)
    

    for ques_json in data_list:
        idx = ques_json["id"]
        image_path = ques_json['image_path']
        audio_path = ques_json['audio_path']
        video_path = ques_json['video_path']
        prompt = ques_json["conversations"][0]["value"]

        output = predict(input=prompt, image_path=image_path,
                     audio_path=audio_path, video_path=video_path,
                     max_tgt_len=max_tgt_length, top_p=top_p,
                     temperature=temperature, modality_cache=modality_cache,
                     stops_id=[[2277]]
                    #  generator=g_cuda,
                     )
        print(output)
        output = output.split("\n")[0]
        ans_jsons.append(
            {
                "question_id": idx,
                "questions": prompt,
                "response": output,
            }
        )

    # Write output to file
    with open(args['answer_file'], "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")

    return ans_jsons



if __name__ == '__main__':
    # init the model

    g_cuda = torch.Generator(device='cuda').manual_seed(1337)
    args = {'model': 'openllama_peft',
            'toollmm_ckpt_path': './ckpt/toollmm_llama_7b_v0_peft/TIVA_1_8_11141830/',
            'imagebind_ckpt_path': '../pretrained_checkpoint/imagebind_ckpt/',
            'llm_ckpt_path': '../pretrained_checkpoint/LLM_ckpt/',
            'max_tgt_len': 1024,
            'stage': 3,
            'root_dir': '../',
            'mode': 'validation',
            'question_file': '../data/IT_data_ins/T+X-T_data/combined_data.json',
            'answer_file': '../data/eval_llama_7b_answer_all.json'
            }
    args.update(load_config(args))

    model = OpenLLAMAPEFTModel(**args)
    delta_ckpt = torch.load(os.path.join(args['toollmm_ckpt_path'], 'pytorch_model_0.pt'), map_location=torch.device('cuda'))
    # print(delta_ckpt)
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().cuda()


    """Override Chatbot.postprocess"""
    max_tgt_length = 150
    top_p = 1.0
    temperature = 0.4
    modality_cache = None

    ans_jsons = []
 
    run_eval(args,args['question_file'])
