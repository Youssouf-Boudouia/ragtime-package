from ragtime.prompters.prompter import Prompter, Prompt
from ragtime.prompters.answer_prompters import (
    AnsPrompterBase,
    AnsPrompterWithRetrieverFR,
)
from ragtime.prompters.fact_prompters import FactPrompterFR, Facts, Fact
from ragtime.prompters.eval_prompters import EvalPrompterFR, Eval
from ragtime.prompters.question_prompters import QuestionPrompterFR, Question, QA

prompterTable: dict = {
    "AnsPrompterBase": AnsPrompterBase,
    "AnsPrompterWithRetrieverFR": AnsPrompterWithRetrieverFR,
    "FactPrompterFR": FactPrompterFR,
    "EvalPrompterFR": EvalPrompterFR,
    "QuestionPrompterFR": QuestionPrompterFR,
}


def reference_Prompter(name, cls):
    prompterTable[name] = cls
