from ragtime.generators.text_generator import *
from ragtime.llms import LLM
from ragtime.expe import StartFrom, QA, Eval, Facts
from ragtime.base import RagtimeException
from ragtime.config import logger, UNKOWN_LLM
from ragtime.config import FOLDER_EVALS


class EvalGenerator(TextGenerator):
    llm: LLM
    output_folder: str = FOLDER_EVALS

    """
    Generate Eval from Answers and Facts.
    For a given QA, send the Answer and the Facts to the LLM to get the prompt back
    The default prompt returns all the valid facts given an answer, i.e. 1 prompt -> 1 Eval
    That could be overridden to have e.g. 1 prompt per Fact, i.e. N prompt -> 1 Eval
    The conversion between the LLM answer and the Eval is made in post_process
    """

    async def gen_for_qa(self, qa: QA):
        """
        Create Eval for each QA where Facts are available
        """

        if len(qa.answers) == 0:
            logger.error(f"No Answers, cannot generate Evals")
            return
        if len(qa.facts) == 0:
            logger.error(f"No Facts, cannot generate Evals")
            return

        # Eval loop
        logger.prefix += f"[EvalGen][{self.llm.name}]"
        for ans in (a for a in qa.answers if a.text):
            llm_name: str = ans.llm_answer.name if ans.llm_answer else UNKOWN_LLM
            logger.debug(f'Generate Eval for answer generated with "{llm_name}"')
            prev_eval: Eval = ans.eval

            # 2.a. and 2.b : prompt generation + Text generation with LLM
            ans.eval = await self.llm.generate(
                cur_obj=Eval(),
                prev_obj=prev_eval,
                qa=qa,
                start_from=self.start_from,
                b_missing_only=self.b_missing_only,
                answer=ans,
                facts=qa.facts,
            )

            # save previous human eval if any
            if prev_eval and prev_eval.human:
                ans.eval.human = prev_eval.human


class TwoFactsEvalGenerator(TextGenerator):
    """
    Generate Eval from Answers and Facts. Converts first the Answer to a list of Facts and
    perform evaluation
    """

    async def gen_for_qa(self, qa: QA):
        """
        Create Eval for each QA where Facts are available
        """

        if len(qa.answers) == 0:
            logger.error(f"No Answers, cannot generate Evals")
            return
        if len(qa.facts) == 0:
            logger.error(f"No Facts, cannot generate Evals")
            return

        # Eval loop
        for ans in (a for a in qa.answers if a.text):
            llm_name: str = (
                ans.llm_answer.name if ans.llm_answer else "unkown LLM (manual ?)"
            )
            logger.debug(f'Generate Facts for answer generated with "{llm_name}"')
            prev_eval: Eval = ans.eval

            # Use 1st LLM to generate facts from the Answer
            ans_facts: Facts = await self.llms[0].generate(
                cur_obj=Facts(),
                prev_obj=None,
                qa=qa,
                start_from=self.start_from,
                b_missing_only=self.b_missing_only,
                answer=ans,
            )

            # Use 2nd LLM to generate Eval
            logger.debug(f"Then generate Eval using answer facts and gold facts")
            cur_eval: Eval = Eval()
            cur_eval.meta["answer_facts"] = [
                af.text for af in ans_facts
            ]  # stores the answer's facts in the current eval
            ans.eval = await self.llms[1].generate(
                cur_obj=cur_eval,
                prev_obj=prev_eval,
                qa=qa,
                start_from=self.start_from,
                b_missing_only=self.b_missing_only,
                answer_facts=ans_facts,
                gold_facts=qa.facts,
            )

            # save previous human eval if any
            if prev_eval and prev_eval.human:
                ans.eval.human = prev_eval.human
