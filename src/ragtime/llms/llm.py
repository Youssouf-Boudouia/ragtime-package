from abc import abstractmethod

from ragtime.prompters import Prompter
from ragtime.base import RagtimeBase
from ragtime.expe import QA, Prompt, LLMAnswer, WithLLMAnswer, StartFrom
from ragtime.config import logger, DEFAULT_MAX_TOKENS


from typing import Optional
import asyncio


class LLM(RagtimeBase):
    """
    Base class for text to text LLMs.
    Class deriving from LLM must implement `complete`.
    A Prompter must be provided at creation time.
    Instantiates a get_prompt so as to be able change the prompt LLM-wise.
    """

    name: Optional[str] = None
    prompter: Prompter
    max_tokens: int = DEFAULT_MAX_TOKENS
    _semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

    async def generate(
        self,
        cur_obj: WithLLMAnswer,
        prev_obj: WithLLMAnswer,
        qa: QA,
        start_from: StartFrom,
        b_missing_only: bool,
        **kwargs,
    ) -> WithLLMAnswer:
        """
        Generate prompt and execute LLM
        Returns the retrieved or created object containing the LLMAnswer
        If None, LLMAnswer retrieval or generation went wrong and post-processing
        must be skipped
        """
        # await self._semaphore.acquire()
        logger.prefix = f"[{self.name}]"

        assert not prev_obj or (cur_obj.__class__ == prev_obj.__class__)
        cur_class_name: str = cur_obj.__class__.__name__

        # Get prompt
        if not (prev_obj and prev_obj.llm_answer and prev_obj.llm_answer.prompt) or (
            start_from <= StartFrom.prompt and not b_missing_only
        ):
            logger.debug(
                f"Either no {cur_class_name} / LLMAnswer / Prompt exists yet, or you asked to regenerate Prompt ==> generate prompt"
            )
            prompt = self.prompter.get_prompt(**kwargs)
        else:
            logger.debug(f"Reuse existing Prompt")
            prompt = prev_obj.llm_answer.prompt

        # Generates text
        result: WithLLMAnswer = cur_obj
        if not (prev_obj and prev_obj.llm_answer) or (
            start_from <= StartFrom.llm and not b_missing_only
        ):
            logger.debug(
                f"Either no {cur_class_name} / LLMAnswer exists yet, or you asked to regenerate it ==> generate LLMAnswer"
            )
            try:
                result.llm_answer = await self.complete(prompt)
            except Exception as e:
                logger.exception(f"Exception while generating - skip it\n{e}")
                result = None
        else:
            logger.debug(f"Reuse existing LLMAnswer in {cur_class_name}")
            result = prev_obj

        # Post-process
        if result.llm_answer and (
            not (prev_obj and prev_obj.llm_answer)
            or not b_missing_only
            and start_from <= StartFrom.post_process
        ):
            logger.debug(f"Post-process {cur_class_name}")
            self.prompter.post_process(qa=qa, cur_obj=result)
        else:
            logger.debug("Reuse post-processing")
        # self._semaphore.release()

        return result

    @abstractmethod
    async def complete(self, prompt: Prompt) -> LLMAnswer:
        raise NotImplementedError("Must implement this!")
