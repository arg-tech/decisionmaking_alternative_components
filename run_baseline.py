import pandas as pd
import json, os, random
from tqdm import tqdm
import re
from alternatives_generation import LLMAlternativesGenerator
from llm_eval_metrics import CGMetricsCalculation, MATCHERS_EXAMPLES
from openai import OpenAI

from datasets import load_dataset

random.seed(2)


# Load files
HF_LINK = 'arg-tech/decisionmaking_alternative_components'

solution_df = load_dataset(HF_LINK, 'solutions')['train'].to_pandas()
replies_df = load_dataset(HF_LINK, 'replies')['train'].to_pandas()
posts_meta_json_df = load_dataset(HF_LINK, 'posts_meta')['train'].to_pandas()
posts_meta_json = posts_meta_json_df.to_dict('records')
posts_meta_json = {x['post_id']: x for x in posts_meta_json}
target_components_upvote_scores_meta_df = load_dataset(HF_LINK, 'scores')['train'].to_pandas()
target_components_upvote_scores_meta = target_components_upvote_scores_meta_df.to_dict('records')
target_components_upvote_scores_meta = {x['post_id']: x for x in target_components_upvote_scores_meta}


N_EX = [5, 10]
N_RUNS = 3

client = OpenAI(api_key="YOUR KEY", base_url="YOUR URL")

initial_client = client
support_client = client

MODELNAME = "gemma2"

MAX_CONT_RUNS = 20
SEED = 2


SAVE_DIR = f"{MODELNAME}--options_preds/"

# filter no solution comments
replies_df = replies_df[replies_df["solution_text"]!="NO_SOLUTION"]
solution_df = solution_df[solution_df["solution_text"]!="NO_SOLUTION"]

replies_df = replies_df[["solution_text", "post_id"]]
solution_df = solution_df[["solution_text", "post_id"]]
merged_solutions_df = pd.concat([replies_df, solution_df])

all_posts = posts_meta_json_df['post_id'].values.tolist()
all_posts = list(sorted(all_posts, key = lambda x: random.uniform(0,1)))

def get_train_test(n_ex, n_splits, all_posts):
    if not n_ex:
        return [
            {
                "train": [],
                "test": all_posts
            }
        ]
    

    splits = []
    
    for i in range(n_splits):

        random_order_posts = list(sorted(all_posts, key = lambda x: random.uniform(0,1)))
        train_posts, test_posts = random_order_posts[:n_ex], random_order_posts[n_ex:]

        splits.append(
            {
                "train": train_posts,
                "test": test_posts
            }
        )
    return splits

def build_examples(examples_posts, merged_solutions_df, posts_meta_json):
    examples = [
        {
            "problem": posts_meta_json[post_id]["body"],
            "title": posts_meta_json[post_id]["title"],
            "post_id": post_id,
            "options": merged_solutions_df[merged_solutions_df["post_id"] == post_id]["solution_text"].unique().tolist()
        } for post_id in examples_posts
    ]
    return examples

def generate_preds(
        save_dir,
        model_inputs_splits_dict,
        max_continue_nums,
        modelname,
        client,
        seed
):
    with open(os.path.join(save_dir, "splits.json"), "w") as f:
        json.dump(model_inputs_splits_dict, f)
    
    for n_ex in model_inputs_splits_dict:
        for i, split_dict in enumerate(model_inputs_splits_dict[n_ex]):

            
            preds_save_path = os.path.join(save_dir, f"NumberExamples{n_ex}_Run{i}")
            os.makedirs(preds_save_path, exist_ok=True)

            generator = LLMAlternativesGenerator(
                modelname=modelname,
                client=client,
                examples=split_dict["train"],
                seed=seed
            )

            verbose_out = f"""Modelname: {modelname}; Num examples: {n_ex}; Run: {i+1} (of {len(model_inputs_splits_dict[n_ex])})"""
            print(verbose_out)
            for test_sample_dict in tqdm(split_dict["test"], desc=verbose_out):

                sample_preds_save_path = os.path.join(preds_save_path, test_sample_dict["post_id"])
                os.makedirs(sample_preds_save_path, exist_ok=True)

                predicted_output = generator.predict(
                    problem=test_sample_dict['problem'], title=test_sample_dict['title']
                )
                with open(os.path.join(sample_preds_save_path, "0.txt"), 'w') as f:
                    f.write(predicted_output)

                num_follows = 1
                while ">>STOP<<" not in predicted_output and num_follows <= max_continue_nums:
                    predicted_output = generator.generate_more(
                        previous_output=predicted_output,
                        problem=test_sample_dict['problem'], 
                        title=test_sample_dict['title']
                        )
                    with open(os.path.join(sample_preds_save_path, f"{num_follows}.txt"), 'w') as f:
                        f.write(predicted_output)
                    num_follows += 1


model_inputs_splits_dict = {}

for n_ex in N_EX:
    model_inputs_splits_dict[n_ex] = []
    
    n_ex_splits = get_train_test(
        n_ex=n_ex, 
        n_splits=N_RUNS, 
        all_posts=all_posts
    )
    for split_dict in n_ex_splits:
        model_inputs_splits_dict[n_ex].append(
            {
                "train": build_examples(
                    examples_posts=split_dict["train"], 
                    merged_solutions_df=merged_solutions_df, 
                    posts_meta_json=posts_meta_json   
                ),
                "test": build_examples(
                    examples_posts=split_dict["test"], 
                    merged_solutions_df=merged_solutions_df, 
                    posts_meta_json=posts_meta_json   
                )
            }
        )



os.makedirs(SAVE_DIR, exist_ok=True)


# No examples run
all_examples_data = build_examples(
    examples_posts=all_posts, 
    merged_solutions_df=solution_df,
    posts_meta_json=posts_meta_json   
)

generate_preds(
    save_dir=SAVE_DIR,
    model_inputs_splits_dict={ 0: [
        {"train": [], "test": all_examples_data}
    ]},
    max_continue_nums=MAX_CONT_RUNS,
    modelname=MODELNAME,
     client=client,
     seed=SEED
)

generate_preds(
    save_dir=SAVE_DIR,
    model_inputs_splits_dict=model_inputs_splits_dict,
    max_continue_nums=MAX_CONT_RUNS,
    modelname=MODELNAME,
    client=client,
    seed=SEED
)


# Parse predistions

def parse_components(output):
    output = output.strip()

    if ">>STOP<<" in output:
        if output.endswith("```"):

            output = output[:output.index(">>STOP<<")].strip()
            output += "```"

        else:
            output = output[:output.index(">>STOP<<")].strip()

    if "```" in output and not output.startswith("```"):
        output = output[output.index("```"):]

    if output.count("```") > 2:
        splitter = "```"
    else:
        if all([line.strip().startswith("*") or re.match(r"[0-9]", line[0].strip()) for line in output.split("\n") if
                line.strip() and line.strip() != "```"]):
            splitter = "\n"
        elif all([line.strip().startswith("<OPTION TEXT>") for line in output.split("\n") if
                  line.strip() and line.strip() != "```"]):
            splitter = "<OPTION TEXT>"
        else:
            splitter = "\n\n"

    components = output.split(splitter)
    components = [x.replace("```", "") for x in components]
    components = [x.strip() for x in components if x.strip()]
    components = [x for x in components if not x.endswith(":")]
    components = [x for x in components if re.search(r"[a-zA-Z]", x)]
    components = [re.sub(r"<[^>]*>", "", x) for x in components]

    return components


def calc_metrics(model_dirs, num_examples,
                 merged_solutions_target_df, replies_df,
                 target_components_upvote_scores_meta,
                 metric_calc, model_parse_funcs):
    agg_metrics = {}

    all_accumulated_components = {}

    for modeldir in model_dirs:
        agg_metrics[modeldir] = {}

        for num_example in num_examples:
            print(modeldir, num_example)
            agg_metrics[modeldir][num_example] = {}

            if num_example not in all_accumulated_components:
                all_accumulated_components[num_example] = {}

            num_example_dirs = [
                os.path.join(modeldir, dirname)
                for dirname in os.listdir(modeldir)
                if dirname.startswith(f"NumberExamples{num_example}_Run")
            ]

            for pred_dir in num_example_dirs:
                agg_metrics[modeldir][num_example][pred_dir] = []

                if num_example not in all_accumulated_components[num_example]:
                    all_accumulated_components[num_example][pred_dir] = {}

                for post_id_dir in os.listdir(pred_dir):
                    if post_id_dir not in all_accumulated_components[num_example][pred_dir]:
                        all_accumulated_components[num_example][pred_dir][post_id_dir] = []

                    all_post_components = []

                    full_post_id_dir = os.path.join(pred_dir, post_id_dir)
                    if os.path.isdir(full_post_id_dir):
                        for filename in os.listdir(full_post_id_dir):

                            if filename.endswith(".txt"):
                                full_filename_path = os.path.join(full_post_id_dir, filename)
                                content = open(full_filename_path, 'r').read()
                                components = model_parse_funcs[modeldir](content)
                                all_post_components += components
                                all_accumulated_components[num_example][pred_dir][post_id_dir] += components

                    target_components = merged_solutions_target_df[
                        merged_solutions_target_df["post_id"] == post_id_dir
                        ]
                    target_components = target_components["solution_text"].unique().tolist()
                    target_components = [x for x in target_components if x != "NO_SOLUTION"]

                    target_did_components = replies_df[replies_df["post_id"] == post_id_dir]
                    target_did_components = target_did_components[target_did_components["author_did"] == 1]
                    target_did_components = target_did_components['solution_text'].unique()

                    target_components_upvote_scores = [
                        target_components_upvote_scores_meta[post_id_dir]["norm_scores"][
                            target_components_upvote_scores_meta[post_id_dir]["texts"].index(s_text)
                        ] for s_text in target_components
                    ]

                    metrics = metric_calc.calculate_metrics(
                        pred_components=all_post_components,
                        target_components=target_components,
                        target_did_components=target_did_components,
                        target_components_upvote_scores=target_components_upvote_scores
                    )
                    metrics["all_post_components"] = all_post_components
                    metrics["post_id"] = post_id_dir

                    agg_metrics[modeldir][num_example][pred_dir].append(metrics)
                    with open('Llama3.1--Allruns--backup--agg_metrics.json', 'w') as f:
                        json.dump(agg_metrics, f)

    return agg_metrics, all_accumulated_components


def calc_ensemble(all_accumulated_components,
                  merged_solutions_target_df,
                  replies_df,
                  target_components_upvote_scores_meta,
                  metric_calc
                  ):
    agg_metrics = {}

    for num_example in all_accumulated_components:
        agg_metrics[num_example] = {}

        for pred_dir in all_accumulated_components[num_example]:
            agg_metrics[num_example][pred_dir] = []

            for post_id_dir in all_accumulated_components[num_example][pred_dir]:
                target_components = merged_solutions_target_df[
                    merged_solutions_target_df["post_id"] == post_id_dir
                    ]
                target_components = target_components["solution_text"].unique().tolist()
                target_components = [x for x in target_components if x != "NO_SOLUTION"]

                target_did_components = replies_df[replies_df["post_id"] == post_id_dir]
                target_did_components = target_did_components[target_did_components["author_did"] == 1]
                target_did_components = target_did_components['solution_text'].unique()

                target_components_upvote_scores = [
                    target_components_upvote_scores_meta[post_id_dir]["norm_scores"][
                        target_components_upvote_scores_meta[post_id_dir]["texts"].index(s_text)
                    ] for s_text in target_components
                ]

                metrics = metric_calc.calculate_metrics(
                    pred_components=all_accumulated_components[num_example][pred_dir][post_id_dir],
                    target_components=target_components,
                    target_did_components=target_did_components,
                    target_components_upvote_scores=target_components_upvote_scores
                )

                agg_metrics[num_example][pred_dir].append(metrics)
    return agg_metrics





metric_calc = CGMetricsCalculation(
    initial_client=initial_client,
    support_client=support_client,
    initial_examples=MATCHERS_EXAMPLES,
    support_examples=MATCHERS_EXAMPLES,
    initial_llm="llama3",
    support_llm="mistral"
)

MODEL_DIRS = [
    SAVE_DIR
    #    "gemma2--options_preds",
    # "llama3--options_preds",
    # "llama3.1--options_preds",
    #        "mistral--options_preds"
]



def gemma_preprocess(text):
    if ">>STOP<<" in text:
        text = text[:text.index(">>STOP<<")]

    # Regex to match everything that starts with ** and ends with :**
    pattern = r'\*\*.*?:\*\*'

    # Replacing the matched patterns with an empty string
    cleaned_text = re.sub(pattern, '', text)

    cleaned_text = cleaned_text.replace("*", "")
    cleaned_text.replace("\n", "\n\n")

    output = parse_components(cleaned_text)

    return output


model_parse_funcs = {
    f"{MODELNAME}--options_preds": gemma_preprocess,
}

SAVE_NAME = f"{MODELNAME}--models-metrics.json"

agg_metrics, all_accumulated_components = calc_metrics(
    model_dirs=MODEL_DIRS,
    num_examples=N_EX,
    merged_solutions_target_df=solution_df,
    replies_df=replies_df,
    target_components_upvote_scores_meta=target_components_upvote_scores_meta,
    metric_calc=metric_calc,
    model_parse_funcs=model_parse_funcs
)

with open(SAVE_NAME, 'w') as f:
    json.dump(agg_metrics, f)

with open("all_accumulated_components--" + SAVE_NAME, 'w') as f:
    json.dump(all_accumulated_components, f)

