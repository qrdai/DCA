#!/usr/bin/env python3
"""
Convert the 'conversations' field in UltraMedical JSONL dataset into 'messages' format
and create a subsample of 50K examples with type 'Exam'.
"""

import json
import random


def convert_file(input_path, output_path):
    """
    Reads a JSONL file, converts 'conversations' to 'messages', and writes to a new JSONL file.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            item = json.loads(line)
            conversations = item.get('conversations', [])
            messages = []
            for conv in conversations:
                if conv.get('from') == 'human':
                    role = 'user'
                elif conv.get('from') == 'gpt':
                    role = 'assistant'
                else:
                    # default mapping for unexpected roles
                    role = conv.get('from')
                content = conv.get('value', '')
                messages.append({'role': role, 'content': content})
            item['messages'] = messages
            # Remove the old field
            item.pop('conversations', None)
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')


def subsample_exam(input_path, output_path, sample_size, seed=42):
    """
    Filters entries with type 'Exam', randomly samples sample_size examples,
    and writes to a new JSONL file.
    """
    exam_items = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            item = json.loads(line)
            if item.get('type') == 'Exam':
                exam_items.append(item)
    if len(exam_items) < sample_size:
        print(f"Warning: only {len(exam_items)} 'Exam' entries found. Sampling all available.")
        sample_size = len(exam_items)
    random.seed(seed)
    subsample = random.sample(exam_items, sample_size)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in subsample:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    sample_size = 50000
    seed = 42
    input_path = '../data/UltraMedical/UltraMedical-train.jsonl'
    converted_path = '../data/UltraMedical/UltraMedical-train-converted.jsonl'
    subsample_path = f'../data/UltraMedical/UltraMedical-train-Exam-{sample_size // 1000}k.jsonl'
    # convert_file(input_path, converted_path)
    subsample_exam(converted_path, subsample_path, sample_size=sample_size, seed=seed)


if __name__ == '__main__':
    main()
