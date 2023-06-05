# Script used to create training data for finetuning a model like LLama.
from chat_gpt3 import chat_gpt3
from gpt_4all import gpt_4all
from db import db
import random
import openai
import json
import time

# Possible: chatgpt, gpt4all
model_name = "gpt4all"
# Create our model instance.
if(model_name == "chatgpt"):
    model = chat_gpt3()
elif(model_name == "gpt4all"):
    model = gpt_4all()

# Set the amount of datasets we want to create
test_data_size = 2000
db = db()
question_types = ['simple',
                  'convoluted',
                  'rather long',
                  'short',
                  'very short',
                  'random',
                  'grammatically wrong']
types = ['question', 'instruction that can be completed by speech']


def get_parameters():
    params = []
    f = open("parameters.txt", "r", encoding='utf-8')
    for line in f.readlines():
        params.append(str(line.strip()))
    return params


def get_question_prompt(con, question_type, typ):
    f = open(model_name + "_input/" + "input_generateQ.txt", "r", encoding='utf-8')
    res = f.read()
    s = ""
    for i in con:
        s += str(i) + "\n"
    res = res.replace("[CONTEXT]", s)
    res = res.replace("[QUESTION_TYPE]", question_type)
    res = res.replace("[TYPE]", typ)
    return res


def get_answer_prompt(con, question):
    f = open(model_name + "_input/" + "input_generateA.txt", "r", encoding='utf-8')
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
            # time.sleep(1.5)
            # Next: Take a random parameter as our context
            context = random.sample(params, random.randint(1, 3))
            print("Chosen context:\n " + "\n".join(context) + "\n")

            # Make chatgpt formulate a question from that context
            # What kind of type will the question be?
            q_type = random.choice(question_types)
            typ = random.choice(types)
            q_prompt = get_question_prompt(context, q_type, typ)
            print("The prompt:\n" + q_prompt + "\n")
            question = model.get_response(q_prompt).strip().replace("\n", "")
            print("Formulated question:\n" + question + "\n")

            # Make chatgpt answer that question on the basis of the context
            a_prompt = get_answer_prompt(context, question)
            print("The prompt:\n" + a_prompt + "\n")
            answer = model.get_response(a_prompt).strip().replace("\n", "")
            print("Formulated Answer:\n" + answer + "\n")

            # Store the question, context and answer
            dataset = {
                "instruction": question.replace("\n", ""),
                "input": "",
                "output": answer.replace("\n", ""),
                "context": "[ITEM]".join(context),
                "model": model_name
            }
            db.get_database()['test_datasets_autogpt'].insert_one(dataset)
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
        print("Done!\nIniting model...")
        if(model_name == "chatgpt"):
            model.init(openai)
        elif(model_name == "gpt4all"):
            model.init()
        print("Done!")

        generate_test_data()
    except Exception as e:
        print(str(e))
