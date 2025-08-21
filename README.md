# Automating Alternative Generation in Decision-Making

This repo contains code and annotation guidelines for the paper:

```text
@inproceedings{
kostiukemnlp2025,
title={Automating Alternative Generation in Decision-Making},
author={Yevhen Kostiuk, Clara Seyfried, Chris Reed},
booktitle={EMNLP 2025},
year={2025}
}
```

To avoid dataset leak and ensure the fair benchmarking of the newer LLMs, the dataset is available by request. See [HF link](HF link).

## Running Code
The code requires OpenAI client compatible endpoint as an inference point.
The code for the paper's results is available in [run_baseline.py](run_baseline.py).

To run it, set the following variables in [run_baseline.py](run_baseline.py) and run the file:

```python
from openai import OpenAI
from datasets import load_dataset

# Datasets loading (will be provided upon request)

HF_LINK = 'arg-tech/decisionmaking_alternative_components'

solution_df = load_dataset(HF_LINK, 'solutions')['train'].to_pandas()
replies_df = load_dataset(HF_LINK, 'replies')['train'].to_pandas()
posts_meta_json_df = load_dataset(HF_LINK, 'posts_meta')['train'].to_pandas()
posts_meta_json = posts_meta_json_df.to_dict('records')
posts_meta_json = {x['post_id']: x for x in posts_meta_json}
target_components_upvote_scores_meta_df = load_dataset(HF_LINK, 'scores')['train'].to_pandas()
target_components_upvote_scores_meta = target_components_upvote_scores_meta_df.to_dict('records')
target_components_upvote_scores_meta = {x['post_id']: x for x in target_components_upvote_scores_meta}

# Number of examples for few shot inference (zero shot will be run separately)
N_EX = [5, 10]
# Number of experiments per few-shot setting (e.g. for each sample, N_RUNS times the inference with the same ranndom seed but different selected few-shot examples from the train dataset will be run for each n from N_EX)
N_RUNS = 3
    

# LLM endpoint
client = OpenAI(api_key="YOUR KEY", base_url="YOUR URL")

# clients for ensemble
initial_client = client
support_client = client

# model to run an experiment for
MODELNAME = "gemma2"

# max numbers of generations until the stopping token (">STOP<") will be generated
MAX_CONT_RUNS = 20

# random seed for ollama inference
SEED = 2

SAVE_DIR = f"{MODELNAME}--options_preds/"
```

### Alternative Generation
The code is available in [alternatives_generation.py](alternatives_generation.py).

How to run (example with gemma2):

```python
from alternatives_generation import LLMAlternativesGenerator
from llm_eval_metrics import MATCHERS_EXAMPLES
from openai import OpenAI

modelname = "gemma2" # make sure it is pulled for ollama via `ollama pull gemma2`

examples = MATCHERS_EXAMPLES # see in llm_eval_metrics.py for example of the format.
seed = 2 # random seed to pass to ollama

client = OpenAI(api_key="YOUR KEY", base_url="YOUR URL")

altern_generator = LLMAlternativesGenerator(
    modelname=modelname,
    client=client,
    examples=examples,
    seed=seed
)

# inputs 
# title is a title of the reddit post;
# problem is a content of the problem with details
title = "University Choice Dillema"
problem = "I got accepted in two universities. The first one is reqlly expensive one, but it is my dream one. The second one is chepaer, but far away. What should I do? Which one should I choose?"

# predict alternatives. Output is a str, unparsed result of the generation.
generated_alternatives = altern_generator.predict(
    problem=problem, title=title
)

# predict more, based on the previous outputs. Output is a str, unparsed result of the generation.
generated_alternatives_more = altern_generator.generate_more(
    problem=problem, title=title, previous_output=generated_alternatives
)
```