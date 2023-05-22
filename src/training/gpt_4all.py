# see here: https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-bindings/python
from gpt4all import GPT4All
#os.chdir("E:\\Python Projects\\R.O.B.E.R.T\\source\\R.O.B.E.R.T\\src\\training\\gpt4all")

try:
    print("Hi!")
    gptj = GPT4All("ggml-mpt-7b-chat")
    print("Model loaded...")
    messages = [{"role": "user", "content": "Name 3 movies"}]
    print("Before")
    print(gptj.chat_completion(messages))
    print("After")
except Exception as ex:
    print(ex)