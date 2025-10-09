import sys
sys.path.append('../')
import os
import json
import pandas as pd
from vllm import LLM, SamplingParams
from utils import concat_csv, split_document_into_sections, create_prompt, create_user_input
from constant import ICD_DIR, GEN_DIR, SYNONYM_PATH
# Load model
# model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

save_dir = "llama32_1B"
if not os.path.isdir(os.path.join(GEN_DIR, save_dir)):
    os.mkdir(os.path.join(GEN_DIR, save_dir))

max_input_len = 6000
max_gen_len = 8000
chunk_size = 500 
num_gpus = 4
max_model_len = max_input_len + max_gen_len


sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_gen_len)
llm = LLM(model=model_name, tensor_parallel_size=num_gpus, max_model_len=max_model_len)
tokenizer = llm.get_tokenizer()

section_titles = json.load(open(SYNONYM_PATH))

for ds in ["mimic3-50"]:
    for split in ["train"]:
        save_path = os.path.join(GEN_DIR, save_dir, f"{ds}_{split}.csv")
        data_path = os.path.join(ICD_DIR, f"{ds}_{split}.json")

        with open(data_path, "r") as f:
            data = json.load(f)

        chunks = []

        # Split the data into chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            df_chunk = pd.json_normalize(chunk)
            chunks.append(df_chunk)
            print(df_chunk.shape[0])
        df_list = []
        for ci, df_chunk in enumerate(chunks):
            section_data = []
            for _, row in df_chunk.iterrows():
                sections = split_document_into_sections(row["TEXT"], section_titles)
                result = []
                for sec_id, (section_id, section_text) in enumerate(sections.items()):
                    result.append({"SUBJECT_ID": row["subject_id"],
                                    "HADM_ID": row["hadm_id"],
                                    "LABELS": row["LABELS"],
                                    "SEC_ID": sec_id, 
                                    "SEC_TEXT": section_text,
                                    "SEC_LENGTH": len(section_text.split())})        
                section_data.extend(result)

            section_df = pd.DataFrame(section_data)
            texts = section_df["SEC_TEXT"]
            print(len(texts))
            user_inputs = [create_user_input(tokenizer, t, max_input_len) for t in texts]
            prompts = [create_prompt(inp) for inp in user_inputs]
            outputs = llm.generate(prompts, sampling_params)
            generated_texts= [output.outputs[0].text for output in outputs]
            print(len(generated_texts))
            section_df["GEN_TEXT"] = generated_texts

            print(f"__Done Split:{split} Chunk:{ci}__")
            df_list.append(section_df)
        concatenated_df = pd.concat(df_list, ignore_index=True)
        concatenated_df.to_csv(save_path, index=False)
        print(f"Concatenated CSV saved as {save_path}")

