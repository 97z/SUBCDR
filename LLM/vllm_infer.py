import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
from tqdm import tqdm
import os
import copy
import math
from typing import List, Union
from venv import logger
from vllm import LLM, SamplingParams
from datasets import load_dataset
#from template import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VLLMInference():
    def __init__(self,
                 model_name_or_path:str,
                 model_type:str,

                 dtype: str = 'float16',  
                 seed: int=0, 
                 trust_remote_code: bool = True, 
                 tensor_parallel_size: int = 4,   
                 gpu_memory_utilization: float = 0.9, 
                **kwargs
                ):
        
        self.model_name_or_path = model_name_or_path
        self.model_name_suffix = os.path.split(model_name_or_path)[-1]
        assert model_type in ['llama', 'qwen2'], f"model_type must be in ['llama', 'qwen2'], but got {model_type}"
        self.model_type = model_type

        self.model = LLM(model=model_name_or_path,
                         trust_remote_code=trust_remote_code,
                         tensor_parallel_size=tensor_parallel_size,
                         dtype=dtype,
                         gpu_memory_utilization=gpu_memory_utilization,
                         #max_model_len=8000,
                         #max_seq_len_to_capture=max_seq_len_to_capture,
                         **kwargs
                        )
        self.tokenizer = self.model.get_tokenizer()
        logger.info(f"vllm tokenizer: \n{self.tokenizer}")
        
 
    def _generate(self, 
                 text:Union[str,List[str]],
                 gen_kwargs:dict,
                 batch_size:int=10,  
                 dataset_name:str=None
                 ):
        print(dataset_name)
        gen_kwargs = copy.deepcopy(gen_kwargs)

        if self.model_type == 'llama':

            gen_kwargs['stop_token_ids'] = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")]
        elif self.model_type == 'qwen2':

            gen_kwargs['stop_token_ids'] = [151645,151643]
        else:
            raise ValueError(f"Only support llama and qwen2, model_type {self.model_type} is not supported")
        
 
        sampling_params = SamplingParams(**gen_kwargs)
        logger.warning(f"Now vllm running sampling_params \n{sampling_params}")
 
        
        if isinstance(text, str):
            text = [text]
        text = [i.strip() for i in text]
        system_prompt = ""
        
        prompt_file = f"/root/autodl-tmp/crossaug_data/{dataset_name}/{dataset_name}_prompt_v11.txt"
        with open(prompt_file, 'r') as f:
            for line in f.readlines():
                system_prompt += line
    
        text = [[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": i}
            ] for i in text]
        
        text = [self.tokenizer.apply_chat_template(i, 
                                                    add_generation_prompt=True, 
                                                    tokenize=False  
                                                ) for i in text]
        text = [self.tokenizer.encode(i, add_special_tokens=False) for i in text]
        logger.info(f"vllm one input:\n{self.tokenizer.decode(text[0])}")
        batch_num = math.ceil(len(text)/batch_size)
        logger.info(f"VLLM batch size: {batch_size}, length of datas num: {len(text)}, batch_num: {batch_num}")
 
    
        final_outputs = []
        batch_outputs = self.model.generate(prompt_token_ids = text, 
                                                sampling_params=sampling_params)
        if sampling_params.n == 1:
            for output in batch_outputs:
                final_outputs.append({"response":output.outputs[0].text})
        else:
            for outputss in batch_outputs:
                temp = []
                for output in outputss.outputs:
                    temp.append({"response":output.text}) 
                final_outputs.append(temp)
        return final_outputs