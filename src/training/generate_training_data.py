# Script used to create training data for finetuning a model like LLama.
from chat_gpt3 import chat_gpt3
from db import db
import random
import openai
import json
import time

# Create our chatgpt instance.
chat_gpt3 = chat_gpt3()
test_data_size = 4000
db = db()


def get_parameters():
    params = []
    f = open("parameters.txt", "r", encoding='utf-8')
    for line in f.readlines():
        params.append(str(line.strip()))
    return params


def get_question_prompt(con):
    f = open("input_generateQ.txt", "r", encoding='utf-8')
    res = f.read()
    s = ""
    for i in con:
        s += str(i) + "\n"
    res = res.replace("[CONTEXT]", s)
    return res


def get_answer_prompt(con, question):
    f = open("input_generateA.txt", "r", encoding='utf-8')
    res = f.read()
    s = ""
    for i in con:
        s += str(i) + "\n"
    res = res.replace("[CONTEXT]", s)
    res = res.replace("[QUESTION]", question)
    return res


def generate_test_data():
    print("Generating a new dataset.")
    # First read in our current environment parameters
    print("===========================================")
    print("The given enviroment parameters:\n")
    params = get_parameters()
    print(params)
    print("\n")
    datasets = []
    # Loop. State how many testdata conversation we want:
    for i in range(0, test_data_size):
        try:
            # Careful with the chatgpt api
            time.sleep(1.5)
            # Next: Take a random parameter as our context
            context = random.sample(params, random.randint(1, 3))
            print("Chosen context:\n " + "\n".join(context) + "\n")

            # Make chatgpt formulate a question from that context
            q_prompt = get_question_prompt(context)
            print("The prompt:\n" + q_prompt + "\n")
            question = chat_gpt3.ask(q_prompt).strip().replace("\n", "")
            print("Formulated question:\n" + question + "\n")

            # Make chatgpt answer that question on the basis of the context
            a_prompt = get_answer_prompt(context, question)
            print("The prompt:\n" + a_prompt + "\n")
            answer = chat_gpt3.ask(a_prompt).strip().replace("\n", "")
            print("Formulated Answer:\n" + answer + "\n")

            # Store the question, context and answer
            dataset = {
                "instruction": question.replace("\n", ""),
                "input": "",
                "output": answer.replace("\n", ""),
                "context": "[ITEM]".join(context)
            }
            db.get_database()['test_datasets'].insert_one(dataset)
            print("Done with item " + str(i))
        except Exception as ex:
            print(ex)

    # At the end: Write it
    #with open("dataset.json", "w", encoding="utf-8") as outfile:
    #    outfile.write(json.dumps(datasets))


if __name__ == "__main__":
    try:
        print("Initing db...")
        db.init()
        print("Done!\nIniting chatgpt...")
        chat_gpt3.init(openai)
        print("Done!")
        generate_test_data()
    except Exception as e:
        print(str(e))
