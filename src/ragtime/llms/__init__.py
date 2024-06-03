from ragtime.llms.llm import LLM
from ragtime.llms.lite_llm import LiteLLM

_llmsTable: dict = {}


def reference_LLM(cls, name):
    _llmsTable[name] = cls


def get_llm_from_name(name):
    llm: LLM = _llmsTable.get(name, None)
    if llm:
        return llm()
    return LiteLLM(name=llm)
