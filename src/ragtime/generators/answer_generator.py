from ragtime.generators import TextGenerator
from ragtime.retrievers.retriever import Retriever
from ragtime.expe import QA, Answer, Answers, StartFrom
from ragtime.config import logger
from ragtime.llms import LLM
from typing import Optional, List

from ragtime.config import FOLDER_ANSWERS


class AnsGenerator(TextGenerator):
    llms: List[LLM]
    retriever: Optional[Retriever] = None
    output_folder: str = FOLDER_ANSWERS
    """
    Object to write answers in the expe
    To use a Retriever, first implement one and give it as parameter when constructing the object
    Besides, subclasses can override the following methods:
    - post_process : to add "meta" fields based on the llm_answer
    Prompts can be changed in the LLM subclass
    """

    def write_chunks(self, qa: QA):
        """Write chunks in the current qa if a Retriever has been given when creating the object. Ignore otherwise"""
        if self.retriever:
            qa.chunks.empty()
            self.retriever.retrieve(qa=qa)

    async def gen_for_qa(self, qa: QA):
        """
        Args
        - qa (QA) : the QA (expe row) to work on
        - start_from : a value in the StartFrom Enum, among:
            - beginning: retrieve chunks (if a Retriever is given, ignore otherwise), compute prompts,
        computer llm answers, compute meta on answers
            - prompt: reuse chunks, compute prompts, llm answers and meta
            - llm: reuse chunks and prompts, compute llm answers and meta
            - post_process: reuse chunks, prompts and llm answers, compute meta only
        - b_missing_only: True to generate LLM Answers only when the Answer object has no "llm_answer"
        Useful to complete a previous experiment where all the Answers have not been generated (happens sometimes due
        to external server failures)
        """
        # Get chunks -> fills the Chunks in the QA
        logger.prefix += "[AnsGen]"
        if self.retriever:
            # Compute chunks if there are not any or there are some and user asked to start à Chunks step or before and did not mention to
            # complete only the missing ones
            if (not qa.chunks) or (
                qa.chunks
                and self.start_from <= StartFrom.chunks
                and not self.b_missing_only
            ):
                logger.info(f"Compute chunks")
                self.write_chunks(qa=qa)
            else:  # otherwise reuse the chunks already in the QA object
                logger.info(f"Reuse existing chunks")

        new_answers: Answer = Answers()
        original_prefix: str = logger.prefix

        for llm in self.llms:
            logger.prefix = f"{original_prefix}[{llm.name}]"
            logger.info(f"* Start with LLM")

            # Get existing Answer if any
            prev_ans: Optional[Answer] = [
                a
                for a in qa.answers
                if (
                    a.llm_answer
                    and (
                        a.llm_answer.name == llm.name
                        or a.llm_answer.full_name == llm.name
                    )
                )
            ]
            if prev_ans:
                # prev_ans is None if no previous Answer has been generated for the current LLM
                prev_ans = prev_ans[0]
                logger.debug(f"An Answer has already been generated with this LLM")
            else:
                prev_ans = None

            # Get Answer from LLM
            ans: Answer = await llm.generate(
                cur_obj=Answer(),
                prev_obj=prev_ans,
                qa=qa,
                start_from=self.start_from,
                b_missing_only=self.b_missing_only,
                question=qa.question,
                chunks=qa.chunks,
            )
            # get previous human eval if any
            if prev_ans and prev_ans.eval:
                ans.eval.human = prev_ans.eval.human
            new_answers.append(ans)
        # end of the per LLM loop, answers have been generated or retrieved, write them in qa
        qa.answers = new_answers
