from ragtime.prompters.answer_prompters import (
    AnsPrompterBase,
    AnsPrompterWithRetrieverFR,
)
from ragtime.prompters.fact_prompters import FactPrompterFR
from ragtime.prompters.eval_prompters import EvalPrompterFR
from ragtime.prompters.question_prompters import QuestionPrompterFR

prompterTable: dict = {
    "AnsPrompterBase": AnsPrompterBase,
    "AnsPrompterWithRetrieverFR": AnsPrompterWithRetrieverFR,
    "FactPrompterFR": FactPrompterFR,
    "EvalPrompterFR": EvalPrompterFR,
    "QuestionPrompterFR": QuestionPrompterFR,
}


def reference_Prompter(name, cls):
    prompterTable[name] = cls
