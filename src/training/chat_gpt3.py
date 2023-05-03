# -*- coding: utf-8 -*-
import sys


class chat_gpt3:

    def init(self, open_ai):
        self.open_ai = open_ai
        # Set up the model and prompt babbage, ada
        self.model_engine = "text-davinci-003"
        # Set up the OpenAI API client
        f = open("openAI_api_key.txt", "r", encoding='utf-8')
        self.open_ai.api_key = f.read()

    def ask(self, prompt):
        # Generate a response
        completion = self.open_ai.Completion.create(
            engine=self.model_engine,
            prompt=prompt,
            max_tokens=2024,
            n=1,
            stop=None,
            temperature=0.3,
        )

        response = completion.choices[0].text
        return response
