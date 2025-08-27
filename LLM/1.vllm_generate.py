import argparse
from vllm_infer import VLLMInference
import json
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., movie, music, video)")
args = parser.parse_args()


dataset_name = args.dataset 
input_file = f"/root/autodl-tmp/crossaug_data/{dataset_name}/crossaug_{dataset_name}_train_.json"
output_file = f"/root/autodl-tmp/crossaug_data/{dataset_name}/{dataset_name}_v11_summary.json"


gen_kwargs_vllm = {
    "max_tokens": 1150,
    "top_p": 0.9,
    "temperature": 0.6,
}
gen_kwargs_vllm_10 = {
    "max_tokens": 1150,
    "top_p": 0.95,
    "temperature": 1.2,
    "n": 10  
}


model_id = "/root/autodl-tmp/llama3_8b_instruct"
llama = VLLMInference(model_name_or_path=model_id, model_type="llama")


user_prompts = load_dataset("json", data_files=input_file)
print(dataset_name)

results = llama._generate(user_prompts["train"]["prompt"], gen_kwargs_vllm, dataset_name = dataset_name)


with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Processing complete. Results saved to: {output_file}")
