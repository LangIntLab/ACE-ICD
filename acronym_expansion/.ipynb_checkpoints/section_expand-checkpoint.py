import sys
import os
import json
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from utils import (
    concat_csv,
    split_document_into_sections,
    create_prompt,
    create_prompt_input
)
from constant import ICD_DIR, GEN_DIR, TITLE_PATH

# Add parent directory to path
sys.path.append('../')

def main(args):
    # Save directory setup
    save_path_dir = os.path.join(GEN_DIR, args.model_name.split("/")[-1])
    if not os.path.isdir(save_path_dir):
        os.mkdir(save_path_dir)

    # Model & generation parameters
    max_model_len = args.max_input_len + args.max_gen_len

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_gen_len
    )
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.num_gpus,
        max_model_len=max_model_len
    )
    tokenizer = llm.get_tokenizer()

    # Load section titles
    with open(TITLE_PATH, "r") as f:
        section_titles = json.load(f)

    # Process datasets
    for ds in args.datasets:
        for split in args.splits:
            save_path = os.path.join(save_path_dir, f"{ds}_{split}.csv")
            data_path = os.path.join(ICD_DIR, f"{ds}_{split}.json")

            # Load JSON data
            with open(data_path, "r") as f:
                data = json.load(f)

            # Split data into chunks
            chunks = []
            for i in range(0, len(data), args.chunk_size):
                chunk = data[i:i + args.chunk_size]
                df_chunk = pd.json_normalize(chunk)
                chunks.append(df_chunk)
                print(f"Chunk {len(chunks)}: {df_chunk.shape[0]} records")

            df_list = []

            # Process each chunk
            for ci, df_chunk in enumerate(chunks):
                section_data = []

                # Split each document into sections
                for _, row in df_chunk.iterrows():
                    sections = split_document_into_sections(row["TEXT"], section_titles)
                    for sec_id, (section_id, section_text) in enumerate(sections.items()):
                        section_data.append({
                            "SUBJECT_ID": row["subject_id"],
                            "HADM_ID": row["hadm_id"],
                            "LABELS": row["LABELS"],
                            "SEC_ID": sec_id,
                            "SEC_TEXT": section_text,
                            "SEC_LENGTH": len(section_text.split())
                        })

                section_df = pd.DataFrame(section_data)

                # Generate model outputs
                texts = section_df["SEC_TEXT"].tolist()
                print(f"Generating for {len(texts)} sections...")

                user_inputs = [create_prompt_input(tokenizer, t, args.max_input_len) for t in texts]
                prompts = [create_prompt(inp) for inp in user_inputs]

                outputs = llm.generate(prompts, sampling_params)
                generated_texts = [output.outputs[0].text for output in outputs]

                print(f"Generated {len(generated_texts)} outputs")
                section_df["GEN_TEXT"] = generated_texts

                print(f"__Done Split: {split} Chunk: {ci}__")
                df_list.append(section_df)

            # Concatenate all chunks and save
            concatenated_df = pd.concat(df_list, ignore_index=True)
            concatenated_df.to_csv(save_path, index=False)
            print(f"Concatenated CSV saved as {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datasets", nargs="+", default=["mimic3-50", "mimic3", "mimic3-50l"], help="List of datasets to process")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--chunk_size", type=int, default=500, help="Number of records per chunk")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--max_input_len", type=int, default=6000)
    parser.add_argument("--max_gen_len", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)

    args = parser.parse_args()
    main(args)