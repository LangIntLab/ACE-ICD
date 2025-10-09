import os
import re
import numpy as np
import pandas as pd


def create_section_regex(section_titles):
    """
    Create a regex pattern from the section titles dictionary.
    Each title and its variations are captured as possible section headings.
    """
    # List to hold all section title variations
    patterns = []
    
    for section, variations in section_titles.items():
        # For each section, create a pattern for its variations
        variations_pattern = "|".join(map(re.escape, variations))
        patterns.append(f"(?P<{"_".join(section.split())}>{variations_pattern})")
    
    # Combine all patterns
    return "|".join(patterns)
    

def split_document_into_sections(document, section_titles):
    """
    Split a document into sections based on the section titles dictionary.
    """
    # Create regex pattern for section headings
    section_regex = create_section_regex(section_titles)
    sections_ids = {k: 0 for k in section_titles.keys()}
    
    # Find all matches for section titles and their corresponding content
    sections = {}
    current_section = None
    current_content = []
    
    # Iterate over each line of the document
    for line in document.splitlines():
        # Check if the line matches any section title
        match = re.match(section_regex, line.strip(), re.IGNORECASE)
        if match:
            # If a section was already being built, save it
            if current_section:
                sections[current_section + "_" + str(sections_ids[current_section.replace("_", " ")])] = "\n".join(current_content)
                sections_ids[current_section.replace("_", " ")] += 1
            
            # Start a new section
            current_section = match.lastgroup
            current_content = [line]
        else:
            # If no match, continue adding to the current section
            if current_section:
                current_content.append(line)
    
    # Add the final section to the dictionary
    if current_section:
        sections[current_section + "_" + str(sections_ids[current_section.replace("_", " ")])] = "\n".join(current_content)
    
    return sections

def split_csv(split, file_path, chunk_size):
    """
    Splits a large CSV file into smaller CSV files, each with a maximum of chunk_size rows.
    """
    chunk_list = [] 

    dir_path = "/".join(file_path.split("/")[:-1])
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        chunk_file = os.path.join(dir_path, f"{split}_chunk_{i}.csv")
        chunk.to_csv(chunk_file, index=False)
        chunk_list.append(chunk_file)

    return chunk_list

def concat_csv(chunk_files, output_file):
    """
    Concatenates multiple CSV files into a single DataFrame and saves it as a CSV.
    """
    df_list = [pd.read_csv(chunk) for chunk in chunk_files]
    concatenated_df = pd.concat(df_list, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved as {output_file}")


def create_prompt(prompt_input):
    prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>"You are a helpful assistant."<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Here is the paragraph with all acronyms expanded to their full forms:'''    
    return prompt

def create_prompt_input(tokenizer, sample, max_length):
    sample_token_ids = tokenizer.encode(sample, return_tensors="pt")[0].numpy()
    
    if len(sample_token_ids) > max_length:
        sample_token_ids = sample_token_ids[:max_length]
        
    sample = tokenizer.decode(sample_token_ids)
    sample = f'''Expand all acronyms to their full forms while preserving all the details in the following paragraph, do not mention the acronyms again. Paragraph: {sample}'''
    return sample