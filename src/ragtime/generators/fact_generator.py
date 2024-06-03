from ragtime.expe import StartFrom, QA, Answer, Facts
from ragtime.generators import TextGenerator
from ragtime.config import logger
from ragtime.config import FOLDER_FACTS
from ragtime.llms import LLM


class FactGenerator(TextGenerator):
    llm: LLM
    output_folder: str = FOLDER_FACTS

    """
    Generate Facts from existing Answers
    """

    async def gen_for_qa(self, qa: QA):
        """
        Create Facts based on the first Answer in the QA having human Eval equals 1
        """

        ans: Answer = next(
            (a for a in qa.answers if a.eval and a.eval.human == 1.0), None
        )
        if not ans:
            logger.debug(
                f"No fact has been generated since no answer has been validated (human=1.0) for this question"
            )
            return

        logger.prefix += f"[FactGen][{self.llm.name}]"
        model_str: str = (
            f" associated with answer from model {ans.llm_answer.full_name}"
            if ans.llm_answer
            else ""
        )
        logger.info(
            f"Generate Facts since it has a human validated answer (eval.human == 1.0){model_str}"
        )
        prev_facts: Facts = qa.facts

        # 2.a. and 2.b : prompt generation + Text generation with LLM
        qa.facts = await self.llm.generate(
            cur_obj=Facts(),
            prev_obj=prev_facts,
            qa=qa,
            start_from=self.start_from,
            b_missing_only=self.b_missing_only,
            answer=ans,
        )
