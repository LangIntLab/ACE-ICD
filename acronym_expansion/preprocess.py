import os
import sys
import json
import argparse
import pandas as pd
from numpy import save
from constant import GEN_DIR, TITLE_PATH
from utils import split_document_into_sections

sys.path.append("../")


def main(args):
    data_path = os.path.join(GEN_DIR, args.model_name.split("/")[-1], f"{args.dataset}_{args.split}.csv")
    save_dir = os.path.join(GEN_DIR, args.model_name.split("/")[-1])
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Load data
    df = pd.read_csv(data_path)
    df["NEW_TEXT"] = ""

    # Counters
    diff_count, serv_count, tag_count, nan_count = 0, 0, 0, 0

    for i, row in df.iterrows():
        gen_text = str(row["GEN_TEXT"]).lower()
        sec_text = str(row["SEC_TEXT"])

        if "<eot_id>" in gen_text:
            df.loc[i, "NEW_TEXT"] = sec_text
            tag_count += 1
            continue

        if gen_text == "nan":
            df.loc[i, "NEW_TEXT"] = sec_text
            nan_count += 1
            continue

        if "national aeronautics" in gen_text:
            df.loc[i, "NEW_TEXT"] = sec_text
            serv_count += 1
            continue

        if len(gen_text.split()) - len(sec_text.split()) > args.max_diff:
            df.loc[i, "NEW_TEXT"] = sec_text
            diff_count += 1
            continue

        df.loc[i, "NEW_TEXT"] = gen_text

    # Replace old text with cleaned text
    df["GEN_TEXT"] = df["NEW_TEXT"]
    df = df.drop(columns=["NEW_TEXT"])

    # Summary
    print(f"Total sections: {df.shape[0]}")
    print(f"# Generated sections with diff > {args.max_diff}: {diff_count}")
    print(f"# Generated sections with Service error (national aeronautics): {serv_count}")
    print(f"# Generated sections with system prompt tags: {tag_count}")
    print(f"# Generated sections with nan error: {nan_count}")
    print(f"Total discarded sections: {diff_count + serv_count + tag_count + nan_count}")

    # Sort by document identifiers
    df = df.sort_values(by=["HADM_ID", "SEC_ID"])

    # Group by 'HADM_ID' and merge the section texts in sorted order
    merged_df = (
        df.groupby("HADM_ID")
        .agg({
            "SUBJECT_ID": "first",
            "LABELS": "first",
            "GEN_TEXT": lambda texts: "\n\n".join(texts),
            "SEC_ID": lambda ids: ", ".join(map(str, ids)),
        })
        .reset_index()
    )

    # Remove "note:" sections at the end
    with open(TITLE_PATH, "r") as f:
        section_titles = json.load(f)
    section_titles["note"] = ["note:"]

    texts = merged_df["GEN_TEXT"]
    processed_texts = []
    note_count = 0

    for t in texts:
        sections = split_document_into_sections(t, section_titles)
        document = []

        for section, content in sections.items():
            if section.startswith("note"):
                note_count += 1
            else:
                document.append(content)

        document_text = "\n".join(document)
        processed_texts.append(document_text)

    print(f"# Generated sections with note: at the end: {note_count}")

    # Replace text columns
    merged_df["TEXT"] = processed_texts
    merged_df = merged_df.drop(columns=["GEN_TEXT", "SEC_ID"])
    merged_df = merged_df.rename(columns={"SUBJECT_ID": "subject_id", "HADM_ID": "hadm_id"})

    # Ensure string types and sorting
    merged_df["subject_id"] = merged_df["subject_id"].astype(str)
    merged_df["hadm_id"] = merged_df["hadm_id"].astype(str)
    merged_df = merged_df.sort_values(by=["subject_id", "hadm_id"])

    # Save output
    output_path = os.path.join(save_dir, f"{args.dataset}_{args.split}.json")
    merged_df.to_json(output_path, orient="records", indent=4)

    print(f"Saved cleaned JSON to: {output_path}")
    print(f"Saved document count: {merged_df.shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", type=str, default="mimic3-50")
    parser.add_argument("--max_diff", type=int, default=200, help="Max word count difference allowed between generated and original text")

    args = parser.parse_args()
    main(args)
