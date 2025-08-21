class LLMAlternativesGenerator:
    def __init__(self,
                 client,
                 modelname,
                 examples,
                 seed=2):

        self.modelname = modelname
        self.client = client
        self.examples = examples
        self.seed = seed

    def create_system_message(self):

        prompt = """You are a Reddit user that suggests different options for problems. You are provided a Reddit post with the title and its body.Then you have to generate DIFFERENT possible actions that person could do in that situation to solve the problem.
        You task is to be CREATIVE. Generate as much possible options as you can. You can generate good options as well as bad options. You need to generate a lot of them so the person could see the whole picture. Do not limit yourself at all.

        Expected output format:
        ```
        <OPTION TEXT>
        ```

        ```
        <OPTION TEXT>
        ```

        ```
        <OPTION TEXT>
        ```

        ...

        >>STOP<<


        When you are done and cannot think of anything more, generate: 
        >>STOP<<
        """
        return {
            "role": "system",
            "content": prompt
        }

    def create_user_problem_desc(self, problem, title):
        return {
            "role": "user",
            "content": f"""Here is the Reddit post\n\nTitle:\n{title}\n\nBody:\n{problem}"""
        }

    def create_expected_output(self, options):
        output = [
            f"""```\n{option}\n```""" for option in options
        ]
        output = "\n\n".join(output) + "\n>>STOP<<"

        return {
            "role": "assistant",
            "content": output
        }

    def create_input_messages(self, problem, title):
        messages = []

        messages.append(self.create_system_message())

        for example_dict in self.examples:
            messages.append(
                self.create_user_problem_desc(
                    problem=example_dict["problem"],
                    title=example_dict["title"]
                )
            )
            messages.append(self.create_expected_output(options=example_dict["options"]))

        messages.append(
            self.create_user_problem_desc(
                problem=problem,
                title=title
            )
        )
        return messages

    def run_llm(self, input_messages):
        input_params = {
            "model": self.modelname,
            "stream": False,
            "messages": input_messages
        }
        if self.seed:
            input_params["options"] = {"seed": self.seed}

        completion_out = self.client.chat.completions.create(
                model=self.modelname,
                messages=input_messages,
                stream=False,
                seed=self.seed
            )
        completion_text_output = completion_out.choices[0].message.content

        return completion_text_output

    def predict(self, problem, title):
        input_messages = self.create_input_messages(problem, title)
        parsed_output = self.run_llm(input_messages)
        return parsed_output

    def generate_more(self, problem, title, previous_output):
        input_messages = self.create_input_messages(problem, title)
        input_messages.append(
            {
                "role": "assistant",
                "content": previous_output
            }
        )

        input_messages.append(
            {
                "role": "user",
                "content": "Good, generate more options please. Don't forget: when you are done, please add >>STOP<<"
            }
        )

        parsed_output = self.run_llm(input_messages)


        return parsed_output