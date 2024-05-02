<img src="img/Ragtime_logo.png" alt="Ragtime 🎹 LLM Ops for all">

# Presentation
**Ragtime** 🎹 is an LLMOps framework which allows you to automatically:
1. evaluate a Retrieval Augmented Generation (RAG) system
2. compare different RAGs / LLMs
3. generate Facts to allow automatic evaluation

Ragtime 🎹 allows you to evaluate **long answers** not only multiple choice questions or counting common words between an answer and a baseline. It is then required to evaluate summarizers, 

In Ragtime 🎹, a *RAG* is made of, optionally, a *Retriever*, and always, one or several *Large Language Model* (*LLM*).
- A *Retriever* takes a *question* in input and returns one or several *chunks* or *paragraphs* retrieved from a documents knowledge base
- A *LLM* is a text to text generator taking in input a *prompt*, made of a question and optional chunks, and returning an *LLMAnswer*

You can specify how *prompts* are generated and how the *LLMAnswer* has to be post-processed to return an *answer*.

# Contributing
Glad you wish to contribute! More details [here](CONTRIBUTING.md).

# How does it work?
The main idea in Ragtime 🎹 is to evaluate answers returned by a RAG based on **Facts** that you define. Indeed, it is very difficult to evaluate RAGs and/or LLMs because you cannot define a "good" answer. Indeed a LLM can return many equialent answers expressed in different ways, making impossible a simple string comparison to determine whether an answer is right or wrong. Even though many proxies have been created, counting the number of common words like in ROUGE for instance is not very precise.

In Ragtime 🎹, answers returned by a RAG or a LLM are evaluated against a set of facts. If the answer validates all the facts, then the answer is deemed correct. Conversely, if some facts are not validated, the answer is considered wrong. The number of validated facts compared to the total number of facts to validate defines a score.

You can either define facts manually, or have a LLM define them for you. **The evaluation of facts against answers is done automatically with another LLM**.

# Main objects
The main objects used in Ragtime 🎹 are:
- `AnswerGenerator`: generate `Answer`s with 1 or several `LLM`s. Each `LLM` uses a `Prompter` to get a prompt to be fed with and to post-process the `LLMAnswer` returned by the `LLM`
- `FactGenerator`: generate `Facts` from the answers with human validation equals to 1. `FactGenerator` also uses an `LLM` to generate the facts
- `EvalGenerator`: generate `Eval`s based on `Answer`s and `Facts`. Also uses a `LLM` to perform the evaluations.
- `LLM`: generates text and return `LLMAnswer` objects
- `LLMAnswer`: answer returned by an LLM. Contains a `text` field, returned by the LLM, plus a `cost`, a `duration`, a `timestamp` and a `prompt` field, being the prompt used to generate the answer
- `Prompter`: a prompter is used to generate a prompt for an LLM and to post-process the text returned by the LLM
- `Expe`: an experiment object, containing a list of `QA` objects
- `QA`: an element an `Expe`. Contains a `Question` and, optionally, `Facts`, `Chunks` and `Answers`.
- `Question`: contains a `text` field for the question's text. Can also contain a `meta` dictionary
- `Facts`: a list of `Fact`, with a `text` field being the fact in itself and an `LLMAnswer` object if the fact has been generated by an LLM
- `Chunks`: a list of `Chunk` containing the `text` of the chunk and optionally a `meta` dictionary with extra data associated with the retriever
- `Answers`: the answer to the question is in the `text` field plus an `LLMAnswer` containing all the data related to the answer generation, plus an `Eval` object related to the evaluation of the answer
- `Eval`: contains a `human` field to store human evaluation of the answer as well as a `auto` field when the evaluation is done automatically. In this case, it also contains an `LLMAnswer` object related to the automatic evaluation

Almost every object in Ragtime 🎹 has a `meta` field, which is a dictionnary where you can store all the extra data you need for your specific use case.

# Examples
You can now go to [ragtime-projects](https://github.com/recitalAI/ragtime-projects) to see examples of Ragtime 🎹 in action!

# Troubleshooting
## Setting the API keys on Windows
API keys are stored in environment variables locally on your computer. If you are using Windows, you should first set the API keys values as:
```shell
setx OPENAI_API_KEY sk-....
```
The list of environment variable names to set, depending on the APIs you need to access, is given in the [LiteLLM documentation](https://litellm.vercel.app/docs/providers).

Once the keys are set, just call `ragtime.config.init_API_keys` with the list of environment variables to make accessible to Python, for instance `init_API_keys(['OPENAI_API_KEY'])`.

## Using Google LLMs
Execute what's indicated in the [LiteLLM documentation](https://litellm.vercel.app/docs/providers/vertex#gemini-pro).
Also make sure your project has `Vertex AI` API enabled.