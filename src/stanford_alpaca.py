from transformers import AutoTokenizer, LlamaForCausalLM

model = transformers.AutoModelForCausalLM.from_pretrained("<path_to_store_recovered_weights>")
tokenizer = transformers.AutoTokenizer.from_pretrained("<path_to_store_recovered_weights>")

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]