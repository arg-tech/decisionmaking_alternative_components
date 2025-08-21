from tqdm import tqdm
from itertools import combinations

MATCHERS_EXAMPLES = [
    {
    "pred_solution": "Open physical stores in new geographic regions",
    "target_solution": "If we comply with all the local regulations, open physical stores in new locations",
    "output": "NOT MATCH"
    },
    {
    "pred_solution": "Cook at home more frequently instead of dining out",
    "target_solution": "Prepare meals at home more often rather than eating out.",
    "output": "MATCH"
    },
    {
    "pred_solution": "Use carpooling or public transportation",
    "target_solution": "Buy a month ticket for a bus and use it",
    "output": "NOT MATCH"
    },
    {
    "pred_solution": "Use carpooling or public transportation",
    "target_solution": "Use bus or other communal transport",
    "output": "MATCH"
    },
    {
    "pred_solution": "Apply for the job at the first company and then tell your boss you did it",
    "target_solution": "Apply for the job at the first company and before that tell your boss you did it",
    "output": "NOT MATCH"
    },
    {
    "pred_solution": "Buy a green car",
    "target_solution": "Buy a red car",
    "output": "NOT MATCH"
    },
    {
    "pred_solution": "Buy a green sandwich",
    "target_solution": "Buy a green biscuit",
    "output": "NOT MATCH"
    },
    {
    "pred_solution": "Talk with your son about the dangers of playing in construction site, but be gentle about it. Make sure he understands. After that, give him back his ps4",
    "target_solution": "Have a calm conversation with your son about the risks of playing around a construction site and make sure he fully understands your point. Once you're done, give him back his Playstation four.",
    "output": "MATCH"
    }
]

class LLMSolutionMatcher:
    def __init__(self,
                 client,
                 examples,
                 model,
                 seed=2):
        self.client = client
        self.examples = examples
        self.model = model

        self.seed = seed

    def make_intro_prompt(self):
        prompt = """You are a semantic similarity system. You need to determine if the two provided solutions that were suggested for a problem are MATCH or NOT MATCH. What you need to do is to say, if the tow solutions are the same or not.
        When making a decision, follow the folowing criteria:
        1. If one solution has a condition and other does not, they are NOT MATCH.
        2. If solutions contain different content, they are NOT MATCH.
        3. If solutions suggest the opposite, they are NOT MATCH.
        4. Solutions suggest the same action, but tho whom/which it should be applied is different, they are NOT MATCH.
        5. If in the solutions the order of actions is different, they are NOT MATCH.

        Otherwise - solutions MATCH."""

        return {"role": "system", "content": prompt}

    def create_input_message_question(self, pred_solution, target_solution):
        prompt = f"""Solution 1:{pred_solution}\nSolution 2:{target_solution}"""
        return {"role": "user", "content": prompt}

    def create_chat_messages(self, pred_solution, target_solution):
        messages = []
        messages.append(self.make_intro_prompt())

        for example_dict in self.examples:
            messages.append(
                self.create_input_message_question(
                    pred_solution=example_dict["pred_solution"],
                    target_solution=example_dict["target_solution"],
                )
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": example_dict["output"]
                }
            )
        messages.append(
            self.create_input_message_question(
                pred_solution=pred_solution,
                target_solution=target_solution,
            )
        )
        return messages

    def is_match(self, pred_solution, target_solution):
        messages = self.create_chat_messages(pred_solution, target_solution)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            seed=self.seed,
            stream=False
        )

        completion_text_output = completion.choices[0].message.content

        return completion_text_output

class EnsembleLLMSolutionMatcher:
    def __init__(self,
                 initial_client,
                 support_client,
                 initial_examples=MATCHERS_EXAMPLES,
                 support_examples=MATCHERS_EXAMPLES,
                 initial_llm="llama3",
                 support_llm="mistral"
                 ):
        self.initial_sol_matcher = LLMSolutionMatcher(
            client=initial_client,
            examples=initial_examples,
            model=initial_llm
        )
        self.support_sol_matcher = LLMSolutionMatcher(
            client=support_client,
            examples=support_examples,
            model=support_llm
        )

    def parse_pred(self, pred_text):

        if "NOT MATCH" in pred_text:
            return "NOT MATCH"
        if "MATCH" in pred_text:
            return "MATCH"

        return "NO ANSWER"

    def predict(self, pred_solution, target_solution):

        init_pred = self.initial_sol_matcher.is_match(pred_solution, target_solution)
        parsed_init_pred = self.parse_pred(init_pred)

        if parsed_init_pred != "NO ANSWER":
            return parsed_init_pred

        supp_pred = self.support_sol_matcher.is_match(pred_solution, target_solution)
        parsed_supp_pred = self.parse_pred(supp_pred)

        if parsed_supp_pred != "NO ANSWER":
            return parsed_supp_pred

        return "NOT MATCH"

class CGMetricsCalculation:
    def __init__(self,
                 initial_client,
                 support_client,
                 initial_examples=MATCHERS_EXAMPLES,
                 support_examples=MATCHERS_EXAMPLES,
                 initial_llm="llama3",
                 support_llm="mistral"
                 ):
        self.ensemble_sol_matcher = EnsembleLLMSolutionMatcher(
            initial_client=initial_client,
            support_client=support_client,
            initial_examples=initial_examples,
            support_examples=support_examples,
            initial_llm=initial_llm,
            support_llm=support_llm
        )

    def remove_duplicates(self, pred_components):
        kept_unique_components_idx = []
        for comb_i, comb_j in tqdm(
                combinations(list(range(len(pred_components))), 2),
                desc="Removing duplicates"
        ):
            pred = self.ensemble_sol_matcher.predict(
                pred_solution=pred_components[comb_i],
                target_solution=pred_components[comb_j]
            )
            if pred == "MATCH":
                if comb_j not in kept_unique_components_idx and comb_i not in kept_unique_components_idx:
                    kept_unique_components_idx.append(comb_i)
            else:
                if comb_j not in kept_unique_components_idx:
                    kept_unique_components_idx.append(comb_j)
                if comb_i not in kept_unique_components_idx:
                    kept_unique_components_idx.append(comb_i)

        unique_components = [pred_components[i] for i in kept_unique_components_idx]

        return unique_components


    def get_matches(self, pred_components, target_components):

        matched_pred_components = []
        matched_target_components = []


        for pred_comp in tqdm(
                pred_components,
                desc="Matching predictions and target"
        ):
            for t_comp in target_components:
                if t_comp not in matched_target_components:
                    match_pred = self.ensemble_sol_matcher.predict(
                        pred_solution=pred_comp,
                        target_solution=t_comp
                    )
                    if match_pred == "MATCH":
                        matched_pred_components.append(pred_comp)
                        matched_target_components.append(t_comp)
                        break

        return matched_pred_components, matched_target_components

    def calc_distinctiveness(self, pred_components, unique_pred_components):
        return len(unique_pred_components)/len(pred_components) if pred_components else -1
    def calc_creativity(self,
                        matched_pred_components,
                        target_components,
                        pred_components,
                        **kwargs
                        ):

        return (len(pred_components) - len(matched_pred_components)) / len(target_components)


    def calc_upvote_weighted_intersection(self,
                                          target_components_upvote_scores,
                                          target_components,
                                          matched_target_components
                                          ):

        score = 0
        for target_comp_upvote_val, target_comp in zip(
                target_components_upvote_scores, target_components
        ):
            if target_comp in matched_target_components:
                score += target_comp_upvote_val
        return score

    def calc_crowd_intersection(self,
                                matched_target_components,
                                target_components
                                ):
        return len(matched_target_components)/len(target_components)

    def calc_did_intersection(self,
                              matched_target_components,
                              target_did_components
                              ):
        return len([
            comp for comp in target_did_components if comp in matched_target_components
        ])/len(target_did_components) if len(target_did_components) else -1

    def calculate_metrics(self,
                          pred_components,
                          target_components,
                          target_did_components,
                          target_components_upvote_scores
                          ):
        metrics_dict = {}
        metrics_dict["num_target_components"] = len(target_components)
        metrics_dict["num_target_did_components"] = len(target_did_components)
        metrics_dict["num_pred_components"] = len(pred_components)

        unique_pred_components = self.remove_duplicates(pred_components=pred_components)
        metrics_dict["num_unique_pred_components"] = len(unique_pred_components)

        matched_pred_components, matched_target_components = self.get_matches(
            pred_components=unique_pred_components,
            target_components=target_components
        )


        metrics_dict["distinctiveness"] = self.calc_distinctiveness(
            pred_components=pred_components,
            unique_pred_components=unique_pred_components
        )
        metrics_dict["creativity"] = self.calc_creativity(
            matched_pred_components=matched_pred_components,
            target_components=target_components,
            pred_components=unique_pred_components
        )
        metrics_dict["upvote-weighted intersection score"] = self.calc_upvote_weighted_intersection(
            target_components_upvote_scores=target_components_upvote_scores,
            target_components=target_components,
            matched_target_components=matched_target_components
        )
        metrics_dict["crowd-intersection"] = self.calc_crowd_intersection(
            matched_target_components=matched_target_components,
            target_components=target_components
        )
        metrics_dict["did-intersection"] = self.calc_did_intersection(
            matched_target_components=matched_target_components,
            target_did_components=target_did_components
        )

        return metrics_dict











