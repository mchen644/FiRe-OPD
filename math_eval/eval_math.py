import argparse
import os
import torch
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from math_verify import parse, verify
import copy

try:
    from math_eval.cod_prompt import build_cod_messages
except ImportError:
    from cod_prompt import build_cod_messages


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None
    
def apply_chat_template(toker, messages, chat_template=None, enable_thinking=False):
    if chat_template is None:
        input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
    else:
        input_prompt = chat_template.format(prompt=messages[0]["content"])
    return input_prompt


def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--begin_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--no_extra_prompt", action="store_true",
                        help="Do not append 'Please reason step by step...' instruction. Use this to match verl training-time val behavior.")
    parser.add_argument("--prompt_style", choices=["baseline", "cod"], default="baseline",
                        help="Prompt style for evaluation. 'cod' reuses the sibling Chain-of-Draft repo's GSM8K CoD YAML and adapts final answers to \\boxed{}.")
    parser.add_argument("--cod_repo_dir", type=str, default=None,
                        help="Path to the Chain-of-Draft repo. Defaults to ../chain-of-draft relative to this repository.")
    parser.add_argument("--cod_task", type=str, default="gsm8k",
                        help="Chain-of-Draft task config to load for CoD prompts. Default: gsm8k.")
    parser.add_argument("--cod_shot", "--shot", dest="cod_shot", type=int, default=0,
                        help="Number of Chain-of-Draft few-shot examples to include for --prompt_style cod. Use -1 for all examples.")
    parser.add_argument("--mistakes_file", type=str, default=None,
                        help="Optional JSONL file for examples with at least one incorrect response.")
    args = parser.parse_args()
    

    toker = AutoTokenizer.from_pretrained(args.model_path)
    args.model_name = os.path.basename(args.model_path)

    llm_kwargs = dict(
        model=args.model_path, tokenizer=args.model_path,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens, n=args.n, seed=args.seed)


    with open(args.input_file, "r", encoding="utf-8") as file:
        input_data = [json.loads(line) for line in file]

    if args.begin_idx >= 0 and args.end_idx >= 0:
        input_data = input_data[args.begin_idx: args.end_idx]


    prompt_messages = []
    for item in input_data:
        problem = item["problem"]
        answer = str(item["answer"])
        item["answer"] = answer
        if not args.no_extra_prompt:
            problem = problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        if args.prompt_style == "cod":
            messages = build_cod_messages(
                problem,
                cod_repo_dir=args.cod_repo_dir,
                shot=args.cod_shot,
                task=args.cod_task,
            )
        else:
            messages = [{"role": "user", "content": problem}]
        prompt_messages.append(messages)

    chat_template = None
    
    prompt_token_ids = [apply_chat_template(toker, messages, chat_template=chat_template, enable_thinking=args.enable_thinking)
                            for messages in prompt_messages]
    
    generations = llm.generate(prompt_token_ids, sampling_params=sampling_params)


    res_data = []
    mistake_records = []
    for i in range(len(input_data)):
        d = copy.deepcopy(input_data[i])
        # For each input, collect all responses and boxed answers
        responses = []
        boxed_answers = []
        response_lengths = []
        acc_list = []
        wrong_responses = []
        # There are args.n generations per input
        for j in range(len(generations[i].outputs)):
            response = generations[i].outputs[j].text.strip()
            responses.append(response)
            response_length = len(toker.encode(response, add_special_tokens=False))
            response_lengths.append(response_length)
            boxed_answer = remove_boxed(last_boxed_only_string(response))
            boxed_answers.append(boxed_answer)
            # Compare boxed_answer with d["answer"]
            if boxed_answer is None:
                acc = False
            else:
                try:
                    if len(boxed_answer) > 300:
                        boxed_answer = boxed_answer[:300]
                    acc = verify(parse("\\boxed{" + d["answer"] + "}"), parse("\\boxed{" + boxed_answer + "}"))
                except Exception:
                    acc = False
            acc_list.append(acc)
            if not acc:
                wrong_responses.append({
                    "sample_index": j,
                    "pred_answer": boxed_answers[-1],
                    "response": response,
                    "response_length": response_length,
                })
        d["pred_answers"] = boxed_answers
        d["responses"] = responses
        d["response_lengths"] = response_lengths
        d["acc_list"] = acc_list
        d["model"] = args.model_name
        d["prompt_style"] = args.prompt_style
        d["cod_shot"] = args.cod_shot if args.prompt_style == "cod" else 0
        res_data.append(d)
        if wrong_responses:
            mistake_records.append({
                "index": i,
                "model": args.model_name,
                "prompt_style": args.prompt_style,
                "cod_shot": args.cod_shot if args.prompt_style == "cod" else 0,
                "problem": d["problem"],
                "answer": d["answer"],
                "pred_answers": boxed_answers,
                "acc_list": acc_list,
                "wrong_responses": wrong_responses,
            })

    total_preds = 0
    correct_preds = 0
    pass_at_k = 0
    avg_length = 0
    for d in res_data:
        accs = d.get("acc_list", [])
        total_preds += len(accs)
        correct_preds += sum(1 for acc in accs if acc)
        if any(acc for acc in accs):
            pass_at_k += 1
        response_lengths = d.get("response_lengths", [])
        if response_lengths:
            avg_length += sum(response_lengths) / len(response_lengths)

    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    pass_at_k  = pass_at_k / len(res_data) if len(res_data) > 0 else 0.0
    avg_length = avg_length / len(res_data) if len(res_data) > 0 else 0.0
    print(f"dataset: {args.input_file}")
    print(f"model: {args.model_path}")
    print(f"prompt_style: {args.prompt_style}")
    if args.prompt_style == "cod":
        print(f"cod_repo_dir: {args.cod_repo_dir or '../chain-of-draft'}")
        print(f"cod_task: {args.cod_task}")
        print(f"cod_shot: {args.cod_shot}")
    print(f"Total predictions: {total_preds}")
    print(f"Accurate predictions: {correct_preds}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"pass@k: {pass_at_k:.4f}")
    print(f"avg_length: {avg_length:.4f}")
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as file:
        for d in res_data:
            file.write(json.dumps(d, ensure_ascii=False) + "\n")

    if args.mistakes_file is not None:
        mistakes_dir = os.path.dirname(args.mistakes_file)
        if mistakes_dir:
            os.makedirs(mistakes_dir, exist_ok=True)
        with open(args.mistakes_file, "w", encoding="utf-8") as file:
            for d in mistake_records:
                file.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"mistakes_file: {args.mistakes_file} ({len(mistake_records)} records)")


if __name__ == '__main__':
    main()
