import json
import openai
import numpy as np
import time
import re
from openai import OpenAI, AsyncOpenAI
from common.utils import read_txt, read_json
from common.math_equivalence import strip_string

# API configuration - please set your own API endpoint and key
API_URL = "YOUR_API_URL_HERE"
API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "gpt-4o-mini"
MODEL_TAG = "gpt-4o-mini"

client = OpenAI(base_url=API_URL, api_key=API_KEY)
async_client = AsyncOpenAI(base_url=API_URL, api_key=API_KEY)

class Meta:
    def __init__(self, question, gt, rounds):
        self.question = question
        self.gt = gt
        self.rounds = rounds

    def __str__(self):
        return f"gt: {self.gt}\nrounds: {self.rounds}"

    def __repr__(self):
        return f"gt: {self.gt}\nrounds: {self.rounds}"

class Round:
    def __init__(self, pred_answers, most_answer, is_correct):
        self.pred_answers = pred_answers
        self.most_answer = most_answer
        self.is_correct = is_correct

    def __str__(self):
        return f"pre: {self.pred_answers}: {self.most_answer}: {self.is_correct}"

    def __repr__(self):
        return f"pre: {self.pred_answers}: {self.most_answer}: {self.is_correct}"

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets

def parse_yes_no(string):
    if "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None

def solve_math_problems(input_str):
    pattern = r'\(([-]?\d+)\)'

    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]

    return None

def parse_YN(input_str):
    pattern = r'(\(Yes\)|\(No\))'
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is not None:
        solution = solution.replace("(", "")
        solution = solution.replace(")", "")
        if solution == "YES":
            solution = "Yes"
        if solution == "NO":
            solution = "No"

    return solution

def parse_answer(input_str):
    pattern = r'(\([A-Z]\))'
    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    return solution

def parse_math_anser(input_str):
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"

    matches = re.findall(pattern, input_str)

    solution = None

    for match_str in matches[::-1]:
        solution = match_str
        if solution == "":
            return None
        if solution[-1] == ".":
            solution = solution[:-1]
        if solution[0] == "$" and solution[-1] == "$":
            solution = solution[1:-1]
        solution = strip_string(solution).strip()
        solution = solution.replace("\(", "")
        solution = solution.replace("\)", "")

        if solution:
            break

    return solution

def compute_accuracy(gt, pred_solutions, log=False, idx=0, is_math=False):
    if is_math:
        if type(pred_solutions) == list:
            pred_answers = []

            for pred_solution in pred_solutions:
                pred_answer = parse_math_anser(pred_solution)
                if pred_answer is not None:
                    pred_answers.append(strip_string(pred_answer))
            if pred_answers == []:
                print("None")
                return 0, None
            pred_answer = most_frequent(pred_answers)
        else:
            pred_answer = parse_math_anser(pred_solutions)
        equal_res = "no"
        if strip_string(gt) == pred_answer:
            equal_res = "yes"
        if equal_res.lower() == "yes":
            return 1, pred_answer
        else:
            return 0, pred_answer
    else:
        if type(pred_solutions) == list:
            pred_answers = []

            for pred_solution in pred_solutions:
                pred_answer = parse_answer(pred_solution)
                if pred_answer is None:
                    pred_answer = solve_math_problems(pred_solution)
                if pred_answer is None:
                    pred_answer = parse_YN(pred_solution)
                if pred_answer is not None:
                    pred_answers.append(pred_answer)
            if pred_answers == []:
                return 0, None
            pred_answer = most_frequent(pred_answers)
        else:
            pred_answer = parse_answer(pred_solutions)
            if pred_answer is None:
                pred_answer = solve_math_problems(pred_solutions)
        if gt == pred_answer:
            return 1, pred_answer
        else:
            return 0, pred_answer

def compare_equal(str1, str2):
    fewshot_ost_prompt = read_txt("prompt/math_equal_prompt.txt")
    fewshot_content = fewshot_ost_prompt.format(
        expression1=str1,
        expression2=str2,
    )

    try:
        response = client.chat.completions.create(model=MODEL_TAG, messages=[{"role": "user", "content": fewshot_content}], max_tokens=4096, n=1)
        completion = json.loads(response.json())
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        print("retrying due to an error......")
        print(e)
        time.sleep(20)
        return compare_equal(str1, str2)["choices"][0]["message"]["content"]

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num

def eval_bbh(file_name, is_math=False):
    response_dict = json.load(open(result_path + "/" + file_name + ".json", "r"))
    questions = list(response_dict.keys())
    accs = []

    for round in range(len(response_dict[questions[0]][0][0])):
        if round % 2 == 1:
            continue
        if round == 0:
            continue
        accuracies = []
        results = []
        idx = 0
        for question in questions:
            responses, gt, question_idx = response_dict[question]
            pred_solutions = []
            for response in responses:
                pred_solution = response[round]['content']
                pred_solutions.append(pred_solution)
            accurate, pred_answer = compute_accuracy(gt, pred_solutions, idx == 47 and round == 4, is_math=is_math)
            if accurate is not None:
                accuracies.append(float(accurate))
                results.append({
                    "question_id": question,
                    "gt": gt,
                    "pred_answer": pred_answer,
                    "round": round
                })
            else:
                # Handle case where accuracy computation failed
                print(f"Warning: Failed to compute accuracy for question {question_idx}")
                print(f"Ground truth: {gt}")
            idx += 1
        print(f"Round{round},accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
        accs.append(np.mean(accuracies))
    print(accs)

def eval_single(file_name, is_math=False):
    response_dict = json.load(open(result_path + "/" + file_name + ".json", "r"))
    questions = list(response_dict.keys())
    accs = []

    for round in range(len(response_dict[questions[0]][0][0])):
        errcnt = 0
        corrcnt = 0
        if round % 2 == 1:
            continue
        if is_math and round == 0:
            continue
        accuracies = []
        results = []
        idx = 0
        for question in questions:
            responses, gt, question_idx = response_dict[question]
            pred_solutions = []
            pred_solution = responses[0][round]['content']
            pred_solutions.append(pred_solution)
            accurate, pred_answer = compute_accuracy(gt, pred_solutions, idx == 47 and round == 4, is_math=is_math)
            if accurate == 0:
                errcnt += 1
            else:
                corrcnt += 1
            if accurate is not None:
                accuracies.append(float(accurate))
                results.append({
                    "question_id": question,
                    "gt": gt,
                    "pred_answer": pred_answer,
                    "round": round
                })
            else:
                # Handle case where accuracy computation failed
                print(f"Warning: Failed to compute accuracy for question {question_idx}")
                print(f"Ground truth: {gt}")
            idx += 1
        print(f"Round{round},accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))
        accs.append(np.mean(accuracies))
        print(f"err_cnt:{errcnt},corrcnt:{corrcnt}")
        break
    print(accs)

def extract_bbh(file_name, is_math=False):
    hard_id = []
    metas = []
    response_dict = json.load(open(result_path + "/" + file_name + ".json", "r"))
    questions = list(response_dict.keys())
    results = {
        "questions_count": len(questions),
        "response_structure": [
            len(response_dict[questions[0]]),
            len(response_dict[questions[0]][0]),
            len(response_dict[questions[0]][0][0])
        ],
        "details": []
    }

    agent_count = len(response_dict[questions[0]][0])
    round_count = len(response_dict[questions[0]][0][0])
    logs = []
    idx = 0
    log_likelihood = 0
    entropy = 0
    for question in questions:
        question_detail = {
            "question": question,
            "gt": None,
            "rounds": []
        }
        response, gt, question_idx = response_dict[question]
        if is_math:
            gt = strip_string(gt)
        question_detail["gt"] = gt
        log_txt = f"question: {question}\n{gt}\n"
        all_correct = []
        rounds = []
        for round in range(1, round_count):
            if round % 2 == 1:
                continue
            if round == 0:
                continue

            round_detail = {
                "round": round,
                "pred_answers": [],
                "most_answer": None,
                "is_correct": None
            }

            pred_solutions = []
            for agent in range(agent_count):
                pred_solutions.append(response[agent][round]['content'])

            pred_answers = []

            if is_math:
                for pred_solution in pred_solutions:
                    pred_answer = parse_math_anser(pred_solution)
                    if pred_answer is not None:
                        pred_answers.append(strip_string(pred_answer))
            else:
                for pred_solution in pred_solutions:
                    pred_answer = parse_answer(pred_solution)

                    if pred_answer is None:
                        pred_answer = solve_math_problems(pred_solution)

                    if pred_answer is None:
                        pred_answer = parse_YN(pred_solution)

                    if pred_answer is not None:
                        pred_answers.append(pred_answer)
            if pred_answers == []:
                pred_answers = ["(None)"] * agent_count
            most_answer = most_frequent(pred_answers)
            is_correct = gt == most_answer

            round_detail["pred_answers"] = pred_answers
            round_detail["most_answer"] = most_answer
            round_detail["is_correct"] = is_correct
            all_correct.append(is_correct)
            log_txt += f"{question_idx}"
            log_txt += f"{pred_answers}"
            log_txt += f"{most_answer}"
            log_txt += f"{is_correct}\n"

            question_detail["rounds"].append(round_detail)
            round = Round(pred_answers, most_answer, is_correct)
            rounds.append(round)

            single_entropy = 0
            correct_prob = pred_answers.count(gt) / len(pred_answers)
            if correct_prob > 0:
                log_likelihood += np.log2(correct_prob)
            for answer in set(pred_answers):
                prob = pred_answers.count(answer) / len(pred_answers)
                single_entropy -= prob * np.log2(prob)
            entropy += single_entropy

        results["details"].append(question_detail)
        logs.append(log_txt)
        meta = Meta(question, gt, rounds)
        metas.append(meta)
        idx += 1
    entropy /= len(metas)
    print(f"Log Likelihood: {log_likelihood}")
    print(f"Entropy: {entropy}")

    count = 0
    for meta in metas:
        if meta.rounds[0].is_correct == False and meta.rounds[1].is_correct == True:
            count += 1
    print(f"correct2incorrect:{count}")

    count = 0
    for meta in metas:
        if meta.rounds[0].is_correct:
            count += 1
    print(f"first round correcct:{count}")

    acc = [0.0 for i in range(500 if is_math else 250)]
    count = 0
    for idx in range(len(metas)):
        meta = metas[idx]
        if meta.rounds[1].is_correct:
            acc[idx] = 1.0
            count += 1
    print(f"second round correct:{count}")

    count = 0
    for meta in metas:
        if meta.rounds[0].is_correct == True and meta.rounds[1].is_correct == False:
            count += 1
    print(f"Number of transitions from correct to incorrect: {count}")

    count = 0
    for meta in metas:
        if meta.rounds[0].pred_answers.count(meta.gt) / len(meta.rounds[0].pred_answers) <= meta.rounds[1].pred_answers.count(meta.gt) / len(meta.rounds[1].pred_answers):
            count += 1
    print(f"Number of cases where first round accuracy is less than or equal to second round accuracy: {count}, Total: {len(metas)}, Proportion: {count/len(metas)}")

    count = 0
    for meta in metas:
        if meta.rounds[0].pred_answers.count(meta.gt) / len(meta.rounds[0].pred_answers) < meta.rounds[1].pred_answers.count(meta.gt) / len(meta.rounds[1].pred_answers):
            count += 1
    print(f"Number of cases where first round accuracy is less than second round accuracy: {count}, Total: {len(metas)}, Proportion: {count/len(metas)}")

    count = 0
    for meta in metas:
        if meta.rounds[0].pred_answers.count(meta.gt) / len(meta.rounds[0].pred_answers) > meta.rounds[1].pred_answers.count(meta.gt) / len(meta.rounds[1].pred_answers):
            count += 1
    print(f"Number of cases where first round accuracy is greater than second round accuracy: {count}, Total: {len(metas)}, Proportion: {count/len(metas)}")

    for r in range(2):
        print(f"Distribution of accuracy in round {r}")
        count = [0 for i in range(11)]
        for meta in metas:
            count[int(meta.rounds[r].pred_answers.count(meta.gt) / len(meta.rounds[r].pred_answers) * 10)] += 1
        print(count)

    with open(f"extract_detail_{file_name}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    list_of_tasks = ["geometric_shapes_id", "logical_deduction_seven_objects_id", "math_500_id"]

    list_of_actions = [["expand"], ["expand", "exchange"], ["expand", "exchange", "exchange"], ["expand", "exchange", "exchange", "exchange"], ["expand", "exchange", "exchange", "exchange", "exchange"], ["expand", "exchange", "exchange", "exchange", "exchange", "exchange"]]
    types = ["exchange", "exchangeI4", "exchangeI6"]

    MODEL_NAME = "gpt-4o-mini"
    model_names = ["gpt-4o-mini", "qwen2.5-7b-instruct", "qwen2.5-3b-instruct", "glm-4-flashx", "glm-4-flash", "qwen-turbo", "qwen-plus"]

    task_name = "math_500_id"
    result_path = f"{MODEL_NAME}/results/debate/{task_name}"

    file_name = f"debate_{MODEL_NAME}_10_2_expand_exchange_agent_com0_False"
    print(file_name)
    eval_bbh(file_name, is_math=True)
    extract_bbh(file_name, is_math=True)
    file_name = f"debate_{MODEL_NAME}_10_2_expand_exchangeI4_agent_com0_False"
    print(file_name)
    eval_bbh(file_name, is_math=True)
    extract_bbh(file_name, is_math=True)
    file_name = f"debate_{MODEL_NAME}_10_2_expand_exchangeI6_agent_com0_False"
    print(file_name)
    eval_bbh(file_name, is_math=True)
    extract_bbh(file_name, is_math=True)
    file_name = f"debate_{MODEL_NAME}_10_2_expand_exchangeI2_agent_com0_False"
    print(file_name)
    eval_bbh(file_name, is_math=True)
    extract_bbh(file_name, is_math=True)
