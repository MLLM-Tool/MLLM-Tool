from header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

import torch
from torch.nn.utils import rnn
import hashlib

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (torch.tensor(stop).cuda() == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            if turn['input_modality'] != 'text':
                text = '</Img> ' + turn['value'] + '\n### Assistant: '
            else:
                text = turn['value'] + '\n### Assistant: '
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'gpt':
                text = "{'api_name': " + turn['value'] + '}\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    """
    :param mode: the target modality
    :param num_tokens: the number of generated signal tokens for generation
    """
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

PROMPT_START = '### Human: <Img>'
class OpenLLAMAPEFTModel(nn.Module):
    '''LoRA for LLaMa model'''

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args

        imagebind_ckpt_path = args['imagebind_ckpt_path']
        llm_ckpt_path = args['llm_ckpt_path']
        max_tgt_len = args['max_tgt_len']
        self.max_tgt_len = max_tgt_len
        self.device = torch.cuda.current_device()
        stage = args['stage']
        epochs = args['epochs']

        print (f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        self.visual_encoder, self.visual_hidden_size = \
        imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print ('Visual encoder initialized.')

        print (f'Initializing language decoder from {llm_ckpt_path} ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        # import pdb;pdb.set_trace()
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_ckpt_path)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print ('Language decoder initialized.')

        print('Tokenizer initialized.')

        self.llama_proj = nn.Linear(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )
        self.embeddings_dir = "../data/embeddings"

    def load_transformed_video_pt(self,video_paths):
        video_outputs = []
        for video_path in video_paths:
            try:
                video_data = torch.load(video_path[:-4]+'.pt')
                video_data_squeezed = torch.squeeze(video_data['vision'], dim=0)
                video_outputs.append(video_data_squeezed)

            except KeyError:
                raise ValueError(f"The key 'vision' was not found in the file {video_path}")
        return torch.stack(video_outputs, dim=0).to(self.device)

    def save_embeddings(self, paths, embeddings):
        # Make sure the directory exists
        os.makedirs(self.embeddings_dir, exist_ok=True)

        for path, embed in zip(paths, embeddings):
            # Generate a unique filename for each video embedding
            unique_filename = self.generate_unique_filename(path)
            filename = os.path.join(self.embeddings_dir, unique_filename)
            # Save the tuple (path, embedding) to a file
            torch.save(embed, filename)
    
    def save_video_embeddings(self, paths, embeddings):
        # Make sure the directory exists
        os.makedirs(self.embeddings_dir, exist_ok=True)

        for path, embed in zip(paths, embeddings):
            # Generate a unique filename for each video embedding
            # unique_filename = self.generate_unique_filename(path)
            # filename = os.path.join(self.embeddings_dir, unique_filename)
            filename = os.path.join(self.embeddings_dir, os.path.basename(path) + '.pt')
            # Save the tuple (path, embedding) to a file
            torch.save(embed, filename)
    
    def generate_unique_filename(self, path):
        hash_object = hashlib.md5(path.encode())
        return hash_object.hexdigest() + '.pt'
    
    def get_video_embeddings(self, video_paths):
        embeddings_dict = {}
        paths_to_process = []

        for path in video_paths:
            # unique_filename = self.generate_unique_filename(path)
            # filename = os.path.join(self.embeddings_dir, unique_filename)
            filename = os.path.join(self.embeddings_dir, os.path.basename(path) + '.pt')
            if os.path.exists(filename):
                # If embedding already exists, load it
                embed = torch.load(filename,map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            else:
                # If embedding does not exist, add to the list to process
                paths_to_process.append(path)
                embed = None
            embeddings_dict[path] = embed
        
        if paths_to_process:
            # Process the videos that do not have saved embeddings
            inputs = {ModalityType.VISION: data.load_and_transform_video_data(paths_to_process, self.device)}
            inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)
                new_embeddings = embeddings[ModalityType.VISION]
            self.save_video_embeddings(paths_to_process, new_embeddings)
            for path, embed in zip(paths_to_process, new_embeddings):
                embeddings_dict[path] = embed

        # Return embeddings in the order of the original video_paths
        return torch.stack([embeddings_dict[path] for path in video_paths if embeddings_dict[path] is not None])

    def get_audio_embeddings(self, audio_paths):
        embeddings_dict = {}
        paths_to_process = []

        for path in audio_paths:
            unique_filename = self.generate_unique_filename(path)
            filename = os.path.join(self.embeddings_dir, unique_filename)
            if os.path.exists(filename):
                # If embedding already exists, load it
                embed = torch.load(filename,map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            else:
                # If embedding does not exist, add to the list to process
                paths_to_process.append(path)
                embed = None
            embeddings_dict[path] = embed
        
        if paths_to_process:
            # Process the videos that do not have saved embeddings
            inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(paths_to_process, self.device)}
            inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)
                new_embeddings = embeddings[ModalityType.AUDIO]
            self.save_embeddings(paths_to_process, new_embeddings)
            for path, embed in zip(paths_to_process, new_embeddings):
                embeddings_dict[path] = embed

        # Return embeddings in the order of the original video_paths
        return torch.stack([embeddings_dict[path] for path in audio_paths if embeddings_dict[path] is not None])

    def get_image_embeddings(self, image_paths):
        embeddings_dict = {}
        paths_to_process = []

        for path in image_paths:
            unique_filename = self.generate_unique_filename(path)
            filename = os.path.join(self.embeddings_dir, unique_filename)
            if os.path.exists(filename):
                # If embedding already exists, load it
                embed = torch.load(filename,map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            else:
                # If embedding does not exist, add to the list to process
                paths_to_process.append(path)
                embed = None
            embeddings_dict[path] = embed
        
        if paths_to_process:
            # Process the videos that do not have saved embeddings
            inputs = {ModalityType.VISION: data.load_and_transform_vision_data(paths_to_process, self.device)}
            inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
            with torch.no_grad():
                embeddings = self.visual_encoder(inputs)
                new_embeddings = embeddings['vision']
            self.save_embeddings(paths_to_process, new_embeddings)
            for path, embed in zip(paths_to_process, new_embeddings):
                embeddings_dict[path] = embed

        # Return embeddings in the order of the original video_paths
        return torch.stack([embeddings_dict[path] for path in image_paths if embeddings_dict[path] is not None])

    def encode_video(self, video_paths):
        video_embeds = self.get_video_embeddings(video_paths)
        inputs_llama = self.llama_proj(video_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_audio(self, audio_paths):
        # inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        # # convert into visual dtype
        # inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        # with torch.no_grad():
        #     embeddings = self.visual_encoder(inputs)
        #     audio_embeds = embeddings[ModalityType.AUDIO] # bsz x 1024
        audio_embeds = self.get_audio_embeddings(audio_paths)
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_image(self, image_paths):
        # inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # # convert into visual dtype
        # inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        # with torch.no_grad():
        #     embeddings = self.visual_encoder(inputs)
        #     image_embeds = embeddings['vision'] # bsz x 1024
        image_embeds = self.get_image_embeddings(image_paths)
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = input_ids.shape[0]

        bos = torch.ones([batch_size, 1],
                         dtype=input_ids.dtype,
                         device=input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1

        p_before = PROMPT_START
        
        # peft model need deeper call

        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        
        if img_embeds is not None:
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim

            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
        else:
            p_before = '### Human: '
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
                self.device)
            # peft model need deeper call
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(
                    batch_size, -1, -1)  # bsz x s1 x embed_dim
            
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, p_after_embeds], dim=1).to(
                self.device)  # bsz x (1+s1+s2) x embed_dim

            # create targets
            empty_targets = (
                torch.ones([batch_size, 1 + p_before_embeds.size()[1]],  # 1 (bos) + s1
                           dtype=torch.long).to(self.device).fill_(-100)
            )  # bsz x (1 + s1)
            targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)  # bsz x (1 + s1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(
                self.device)  # bsz x (1 + s1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
            assert attention_mask.size() == targets.size()  # bsz x (1 + s1 + s2)
        return inputs_embeds, targets, attention_mask 

    def _train_with_mode(self, texts, img_embeds=None, modality='text'):
        """
        :param num_gen_tokens: the number of generation tokens
        :param modality: mode can be 'image' / 'video' / 'audio' / 'text'
        :param text_hidden_fcs: alignment module
        :param gen_token_idx: List
        :param text_emb_layers: the layer index of LLM hidden states
        :param text_prompt_embeddins: the textual caption/prompt embeddings
        :param loss_scale: the scale on the mse loss for alignment
        :param stage: the training stage
        :param
        """
        
        input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, texts, self.max_tgt_len)
        
        inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        loss = outputs.loss
        # calculate the token accuarcy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask    # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    def forward(self, inputs):
        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'ImageToText':
            image_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_image(image_paths)
            loss, gen_acc = self._train_with_mode(inputs['output_texts'], mm_embeds, modality='text')
        elif dataset_type == 'VideoToText':
            video_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_video(video_paths)
            loss, gen_acc = self._train_with_mode(inputs['output_texts'], mm_embeds, modality='text')
        elif dataset_type == 'AudioToText':
            audio_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_audio(audio_paths)
            loss, gen_acc = self._train_with_mode(inputs['output_texts'], mm_embeds, modality='text')
        elif dataset_type == 'TextToText':
            loss, gen_acc = self._train_with_mode(inputs['output_texts'], None, modality='text')
        else:
            raise NotImplementedError

        return loss, gen_acc

    def extract_multimodal_feature(self, inputs):
        features = []
        if inputs['image_path']:
            image_embeds, _ = self.encode_image(inputs['image_path'])
            features.append(image_embeds)
        if inputs['audio_path']:
            audio_embeds, _ = self.encode_audio(inputs['audio_path'])
            features.append(audio_embeds)
        if inputs['video_path']:
            video_embeds, _ = self.encode_video(inputs['video_path'])
            features.append(video_embeds)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds

    
    def _prepare_image_embed(self, text, batch_size):
        pattern = r'Image>(.*?)<\/Image'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='pt').to(self.device)

        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                        -1)  # bsz x s1 x embed_dim
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                    -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('image path: ', m)
            _temp_embedding, _ = self.encode_image([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def _prepare_video_embed(self, text, batch_size):
        pattern = r'Video>(.*?)<\/Video'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='pt').to(self.device)

        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                        -1)  # bsz x s1 x embed_dim
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                    -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('Video path: ', m)
            _temp_embedding, _ = self.encode_video([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def _prepare_audio_embed(self, text, batch_size):
        pattern = r'Audio>(.*?)<\/Audio'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='pt').to(self.device)

        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                        -1)  # bsz x s1 x embed_dim
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                    -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('Audio path: ', m)
            _temp_embedding, _ = self.encode_audio([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)


    def prepare_generation_embedding(self, inputs):
        batch_size = 1
        features = []
        
        p_before = '### Human: '
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(
            self.device)
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(
                batch_size, -1, -1)  # bsz x s1 x embed_dim

        if inputs['modality']  == 'Text':
            text = inputs["query"] + '\n### Assistant: '
            text_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
            text_embeds = self.llama_model.model.model.embed_tokens(text_tokens.input_ids).expand(batch_size,
                                                                                                        -1, -1)
            bos = torch.ones([batch_size, 1],
                                 dtype=text_tokens.input_ids.dtype,
                                 device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
            bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
            input_embeds = torch.cat([bos_embeds, p_before_embeds, text_embeds], dim=1)

        else:
            p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
            p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='pt').to(self.device)
            p_beforetoken_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                        -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                    -1)  # bsz x s2 x embed_dim
            if inputs['modality'] == 'Image':
                _temp_embedding, _ = self.encode_image([inputs['mm_path']])
                features.append(_temp_embedding)
                feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
                image_embeds = torch.cat([p_beforetoken_embeds, feature_embeds, p_after_embeds], dim=1)
            elif inputs['modality'] == 'Video':
                _temp_embedding, _ = self.encode_video([inputs['mm_path']])
                features.append(_temp_embedding)
                feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
                image_embeds = torch.cat([p_beforetoken_embeds, feature_embeds, p_after_embeds], dim=1)
            elif inputs['modality'] == 'Audio':
                _temp_embedding, _ = self.encode_audio([inputs['mm_path']])
                features.append(_temp_embedding)
                feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
                image_embeds = torch.cat([p_beforetoken_embeds, feature_embeds, p_after_embeds], dim=1)
            
            text = inputs["query"] + '\n### Assistant: '
            text_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
            text_embeds = self.llama_model.model.model.embed_tokens(text_tokens.input_ids).expand(batch_size,
                                                                                                        -1, -1)
            bos = torch.ones([batch_size, 1],
                                 dtype=text_tokens.input_ids.dtype,
                                 device=text_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
            bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
            input_embeds = torch.cat([bos_embeds, p_before_embeds, image_embeds, text_embeds], dim=1)
        
        return input_embeds


    def generate_tokens_embeddings(self, inputs, input_embeds, temperature: float = 0.0, top_p: float = 1.0):
        """
        This function is used to generate the tokens 
        inputs: dict
        input_embeds: tensor
        return:
            out: the output tokens index
        """
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=inputs['stops_id'], encounters=1)])

        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            # repeat_pen,
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

        # out = outputs.sequences

        return outputs
    
    def generate(self, inputs):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature, used to modulate logit distribution.
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache,
                'modality': modality,
                'mm_path' : mm_path 
            }
        '''
        input_embeds = self.prepare_generation_embedding(inputs)
        outputs = self.generate_tokens_embeddings(inputs, input_embeds)
        output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return output_text

