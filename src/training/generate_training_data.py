# Script used to create training data for finetuning a model like LLama.
# Possible: chatgpt, gpt4all
model_name = "chatgpt"
# Possible paraphrasing model name: "", pegasus_paraphrase
p_model_name = "pegasus_paraphrase"
generate_new_data_for_paraphrasing = False
# Determine if we want to train chat sequences. In that case, we dont
# generate new questions. We take existing ones and create conversations from it.
train_chatting = True
# Determines the absolute max amount of back and forth we want. 
max_chatting_length = 6

from db import db
import random
import openai
import traceback
import json
import time
if(model_name == "gpt4all"):
    from gpt_4all import gpt_4all
elif(model_name == "chatgpt"):
    from chat_gpt3 import chat_gpt3

if(p_model_name == "pegasus_paraphrase"):
    from pegasus_paraphrase import paraphraser


# Create our model instance.
if(model_name == "chatgpt"):
    model = chat_gpt3()
elif(model_name == "gpt4all"):
    model = gpt_4all()

# Create a potential paraphrase model
if(p_model_name == "pegasus_paraphrase"):
    p_model = paraphraser()

# Set the amount of datasets we want to create
test_data_size = 1000
# Set how many paraphrasing of each dataset we want
paraphrasing_count = 6
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


def get_chat_prompt(filename, con, chat_so_far):
    f = open(model_name + "_input/" + str(filename), "r", encoding='utf-8')
    res = f.read()
    s = ""
    for i in con:
        s += str(i) + "\n"
    res = res.replace("[CONTEXT]", s)
    c = ""
    count = 0
    for i in chat_so_far:
        c += str(i) + "\n"
        count += 1
    if(len(chat_so_far) % 2 == 0):
        c += "Student: "
    else:
        c += "Rob: "
    res = res.replace("[CHAT_SO_FAR]", c)
    return res


def get_chat_rob_prompt(con, chat_so_far):
    return get_chat_prompt("input_chat_rob.txt", con, chat_so_far)


def get_chat_student_prompt(con, chat_so_far):
    return get_chat_prompt("input_chat_student.txt", con, chat_so_far)


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


def paraphrase(inp):
    '''Takes the inp string and paraphrases it'''
    return p_model.get_response(inp, paraphrasing_count, 10)


def get_current_db_table(model=""):
    if(model != ""):
        return db.get_database()['test_datasets_' + model]
    if(train_chatting):
        return db.get_database()['test_datasets_chatting']
    else:
        return db.get_database()['test_datasets_' + model_name]


def insert_dataset(dataset):
    return get_current_db_table().insert_one(dataset)


def generate_instruction_output_test_data(params):
    '''Creates a dataset for instruction/output format'''
    # Careful with the chatgpt api
    # time.sleep(1.5)
    # Next: Take a random parameter as our context
    context = random.sample(params, random.randint(1, 3))
    print("Chosen context:\n " + "\n".join(context) + "\n")

    dataset = {}
    object_id = ""
    # Check if we are paraphrasing the data and then check if we want
    # to create new data for paraphrasing or take existing.
    if(p_model_name == "" or generate_new_data_for_paraphrasing is True):
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
        # The objectId of the just inserted object in mongodb
        object_id = insert_dataset(dataset).inserted_id
    else:
        # Else just take an existing dataset
        dataset = list(get_current_db_table(model_name).aggregate(
            [{"$sample": {"size": 1}}]))[0]
        object_id = dataset["_id"]

    # Check if we want to paraphrase the dataset
    if(p_model_name != ""):
        paraphrase_dataset(dataset, object_id)


def paraphrase_dataset(dataset, para_from):
    print("\nParaphrasing:")
    p_questions = paraphrase(dataset['instruction'])
    p_answers = paraphrase(dataset['output'])
    print("\nQuestions:")
    print(p_questions)
    print("\nAnswers:")
    print(p_answers)
    print("\n")
    for x in range(paraphrasing_count):
        p_dataset = {
            "instruction": p_questions[x],
            "input": dataset["input"],
            "output": p_answers[x],
            "context": dataset["context"],
            "model": model_name,
            "paraphrased_from": para_from,
            "p_model": p_model_name
        }
        insert_dataset(p_dataset)


def generate_chat_sequence_test_data(params):
    '''Geneartes a dataset of type chatting'''
    # Get a random, existing dataset.
    existing = list(get_current_db_table(model_name).aggregate(
        [{"$sample": {"size": 1}}]))[0]

    # Get the original context. We stored it in the database
    context = existing['context'].split("[ITEM]")
    # The input of the existing dataset is the start of the convo
    start_q = existing["instruction"]

    # Now start the convo. The conversation goes as long as the
    # model determines it to be. We put in a max amount tho
    counter = 0
    chat = []
    # the student starts the convo
    robs_turn = False
    while(True):
        if(counter >= max_chatting_length):
            break
        # time.sleep(1)
        answer = ""
        prompt = ""
        if robs_turn is False:
            prompt = get_chat_student_prompt(context, chat)
        else:
            # Build the prompt
            prompt = get_chat_rob_prompt(context, chat)

        # Let the model generate the next line
        if(counter == 0):
            answer += "Student: " + start_q
        else:
            answer += model.get_response(prompt).strip().replace("\n", "")

        # Sometimes the model generates the "Rob: " and "Student: " by itself,
        # sometimes it doesnt. Check it.
        if(robs_turn is True and answer.startswith("Rob:") is False):
            answer = "Rob: " + answer
        elif(robs_turn is False and answer.startswith("Student:") is False):
            answer = "Student: " + answer

        if(robs_turn is True):
            # We store each dialogue step where rob answered as a dataset
            dataset = {
                "instruction": chat[len(chat)-1],
                "input": '\n'.join(chat[:-1]),
                "output": answer,
                "context": existing['context'],
                "model": model_name,
                "type": "chat"
            }
            object_id = insert_dataset(dataset).inserted_id
            # Do we want to paraphrase the dialog sequence?
            if(p_model_name != ""):
                paraphrase_dataset(dataset, object_id)

        chat.append(answer)
        print(answer)
        # Idk why python sucks so hard to invert a fucking boolean,
        # but since everything else doesnt work, here we go:
        if(robs_turn is True):
            robs_turn = False
        elif(robs_turn is False):
            robs_turn = True
        counter += 1
        # This means that the chat ended.
        if(answer == ""):
            break
    print("Done with the dialogue.\n\n")


def generate_test_data():
    print("Generating a new dataset.")
    # First read in our current environment parameters
    print("===========================================")
    print("The given enviroment parameters:\n")
    params = get_parameters()
    print(params)
    print("\n")
    # Loop. State how many testdata conversation we want:
    for i in range(0, test_data_size):
        try:
            if(train_chatting):
                generate_chat_sequence_test_data(params)
            else:
                generate_instruction_output_test_data(params)
            print("Done with item " + str(i))
        except Exception as ex:
            print(ex)
            traceback.print_exc()


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
        if(p_model_name == "pegasus_paraphrase"):
            print("Initing paraphrase model...")
            p_model.init()
            print("Done!")

        generate_test_data()
    except Exception as e:
        print(str(e))
