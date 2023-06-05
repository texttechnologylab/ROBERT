from gpt4all import GPT4All

content = """
### Instruction:
You are Rob, a robotic Virtual Reality Assistant. You have the following input:
### Input:
- The Va. Si. Li.-Lab is a virtual reality teaching platform made by the Text Technology Lab. It simulates real life scenarios in Virtual Reality.

The prompt below is a question to answer, a task to complete, or a conversation.
Answer shortly on the basis of the given input. Answer only when you truthfully know the answer, otherwise excuse yourself.
### Prompt:
A student says/asks: What features does the Va. Si. Li.-Lab offer to enhance the learning experience in virtual reality?

### Response:"""


class gpt_4all:

	def init(self, model_name="ggml-vicuna-13b-1.1-q4_2.bin"):
		self.gptj = GPT4All(model_name)

	def get_response(self, message):
		messages = [{
			"role": "system",
			"content": message
		}]
		return str(self.gptj.chat_completion(messages,
								  default_prompt_header=False,
								  default_prompt_footer=False,
								  streaming=False,
								  verbose=False)["choices"][0]["message"]["content"])


if __name__ == "__main__":
	model = gpt_4all()
	print("Starting gpt4all...")
	model.init()
	print("Model inited!")
	print(model.get_response(content)["choices"][0]["message"]["content"])