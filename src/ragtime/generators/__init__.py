from ragtime.generators.text_generator import TextGenerator
from ragtime.generators.answer_generator import AnsGenerator
from ragtime.generators.eval_generator import EvalGenerator
from ragtime.generators.fact_generator import FactGenerator
from ragtime.generators.question_generator import QuestionGenerator

generatorTable: dict[str, TextGenerator] = {
    "answers": AnsGenerator,
    "facts": FactGenerator,
    "evals": EvalGenerator,
    "questions": QuestionGenerator,
}
