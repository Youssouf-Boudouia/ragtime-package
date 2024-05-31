from ragtime.llms import LLM
from ragtime.expe import Prompt, LLMAnswer
from ragtime.config import logger

from litellm import completion_cost, acompletion
from litellm.exceptions import RateLimitError

from datetime import datetime
import asyncio


class LiteLLM(LLM):
    """
    Simple extension of LLM based on the litellm library.
    Allows to call LLMs by their name in a stantardized way.
    The default get_prompt method is not changed.
    The generate method uses the standard litellm completion method.
    Default values of temperature (0.0)
    Number of retries when calling the API (3) can be changed.
    The proper API keys and endpoints have to be specified in the keys.py module.
    """

    name: str
    temperature: float = 0.0
    num_retries: int = 3

    async def complete(self, prompt: Prompt) -> LLMAnswer:
        messages: list[dict] = [
            {"content": prompt.system, "role": "system"},
            {"content": prompt.user, "role": "user"},
        ]
        retry: int = 1
        wait_step: float = 3.0
        start_ts: datetime = datetime.now()
        answer: dict = None
        while retry < self.num_retries:
            try:
                time_to_wait: float = wait_step
                answer = await acompletion(
                    messages=messages,
                    model=self.name,
                    temperature=self.temperature,
                    num_retries=self.num_retries,
                    max_tokens=self.max_tokens,
                )
                break
            except RateLimitError as e:
                logger.debug(
                    f"Rate limit reached - will retry in {time_to_wait:.2f}s\n\t{str(e)}"
                )
                await asyncio.sleep(time_to_wait)
                retry += 1
            except Exception as e:
                logger.exception(
                    f"The following exception occurred with prompt {prompt}"
                    + "\n"
                    + str(e)
                )
                return None

        # TODO:
        # remove this patch to a better error handling

        try:
            full_name: str = answer["model"]
            text: str = answer["choices"][0]["message"]["content"]
            duration: float = (
                answer._response_ms / 1000 if hasattr(answer, "_response_ms") else None
            )  # sometimes _response_ms is not present
            cost: float = float(completion_cost(answer))
            return LLMAnswer(
                name=self.name,
                full_name=full_name,
                prompt=prompt,
                text=text,
                timestamp=start_ts,
                duration=duration,
                cost=cost,
            )
        except Exception as e:
            logger.debug(f"Faile to process the Answer. {e}")
        return LLMAnswer()
