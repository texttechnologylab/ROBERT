
<div align="center">
  <img src="https://github.com/TheItCrOw/R.O.B.E.R.T./assets/49918134/a19fa9f1-d77e-49b9-912a-28012ef9f435"/>
  <hr/>
  <h1>An open-source instruction-following large language chatting model that self-instructs itself into your specific domain.</h1>
</div

[![License](https://img.shields.io/badge/Status-Under%20construction-red)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the repository for R.O.B.E.R.T., a chatting assistant that self-instructs itself into a specificially provided context by generating the needed datasets itself and then finetuning the LLaMa model on these datasets.
This project wouldn't be possible without:

- [Lighting-AI/lit-llama](https://github.com/Lightning-AI/lit-llama)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all)
- [openai/chatgpt](https://openai.com/blog/chatgpt)
- [MetaAI/LLaMa](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
- [PEGASUS](https://github.com/google-research/pegasus)

Special thanks also to the [Text Technology Lab](https://www.texttechnologylab.org/) of the Goethe University Frankfurt from which department this project originates from. 

# About

This repository aims to provide a streamlined process of generating a language model that has the ability to assist and chat with users in a specific domain context: <b>Your own R.O.B.E.R.T.</b> 

The datasets for the context will be generated by the methods provided in the [Self-Instruct paper](https://arxiv.org/abs/2212.10560), which were also used by the [Stanford Alpaca project](https://github.com/tatsu-lab/stanford_alpaca).

This project takes these approaches even further by: 
1. Not relying on chatgpt for the self-instructions (avoiding additional costs) and instead using a language model that *runs locally on your home CPU*.
2. Adding different dataset generation techniques to minimize the generation of new datasets (such as paraphrasing and chatting generation)
3. Using a finetuning method that can be easily executed on e.g. a Google VM with low GPU cost.
