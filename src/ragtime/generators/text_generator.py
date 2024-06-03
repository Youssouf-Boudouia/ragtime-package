from abc import abstractmethod, ABC

from ragtime.base import RagtimeBase
from ragtime.expe import StartFrom, QA
from ragtime.llms import LLM, LiteLLM
from ragtime.prompters.prompter import Prompter
from ragtime.base import RagtimeException
from ragtime.config import logger
from ragtime.expe import Expe

from typing import Optional, List
import asyncio

from ragtime.exporters import Json
from ragtime.pipeline.validator import (
    LLMValidator,
    PrompterValidator,
    ExporterListValidator,
)


class TextGenerator(RagtimeBase, ABC):
    llms: List[LLMValidator]
    prompter: PrompterValidator
    export_to: ExporterListValidator = []

    b_use_chunks: bool = False
    save_every: Optional[int] = 0
    start_from: Optional[StartFrom] = StartFrom.beginning
    b_missing_only: Optional[bool] = False
    output_folder: str = None

    """
    Abstract class for QuestionGenerator, AnswerGenerator, FactGenerator, EvalGenerator
    """

    __save_temp: Json = Json(
        b_overwrite=True,
        b_add_suffix=True,
    )
    __save: Json = Json(
        b_overwrite=True,
        b_add_suffix=True,
    )

    def generate(self, expe: Expe):
        """
        Main method calling "gen_for_qa" for each QA in an Expe. Returns False if completed with error, True otherwise
        The main step in generation are :
        - beginning: start of the process - when start_from=beginning, the whole process is executed
            - chunks: only for Answer generation - chunk retrieval, if a Retriever is associated with the Answer Generator object
        Takes a Question and returns the Chunks
        - prompt: prompt generation, either directly using the question or with the chunks if any
        Takes a Question + optional Chunks and return a Prompt
        - llm: calling the LLM(s) with the generated prompts
        Takes a Prompt and return a LLMAnswer
        - post_process: post-processing the aswer returned by the LLM(s)
        Takes LLMAnswer + other information and updates the Answer object
        Args:
            - expe: Expe object to generate for
            - start_from: allows to start generation from a specific step in the process
            - b_missing_only: True to execute the steps only when the value is absent, False to execute everything
            even if a value already exists
            - only_llms: restrict the llms to be computed again - used in conjunction with start_from -
            if start from beginning, chunks or prompts, compute prompts and llm answers for the list only -
            if start from llm, recompute llm answers for these llm only - has not effect if start
        """

        nb_q: int = len(expe)

        async def _generate_for_qa(num_q: int, qa: QA):
            logger.prefix = f"({num_q}/{nb_q})"
            logger.info(
                f'*** {self.__class__.__name__} for question \n"{qa.question.text}"'
            )
            try:
                await self.gen_for_qa(qa=qa)
            except Exception as e:
                logger.exception(
                    f"Exception caught - saving what has been done so far:\n{e}"
                )
                self.__save.save(expe)
                self.__save_temp.save(expe, file_name=f"Stopped_at_{num_q}_of_{nb_q}_")
                return
            logger.info(f'End question "{qa.question.text}"')
            if self.save_every and (num_q % self.save_every == 0):
                self.__save.save(expe)

        loop = asyncio.get_event_loop()
        tasks = [_generate_for_qa(num_q, qa) for num_q, qa in enumerate(expe, start=1)]
        logger.info(f"{len(tasks)} tasks created")
        loop.run_until_complete(asyncio.gather(*tasks))

    def write_chunks(self, qa: QA):
        """
        Write chunks in the current qa if a Retriever has been given when creating the object.
        Ignore otherwise
        """
        raise NotImplementedError("Must implement this if you want to use it!")

    @abstractmethod
    async def gen_for_qa(self, qa: QA):
        """
        Method to be implemented to generate Answer, Fact and Eval
        """
        raise NotImplementedError("Must implement this!")
