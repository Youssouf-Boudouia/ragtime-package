from ragtime.llms import LLM, get_llm_from_name

from ragtime.prompters import Prompter, prompterTable
from ragtime.exporters import Exporter, exporterTable


from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from typing import Any, List, Union


def prompter_from_name(prompter: Union[Prompter, str]) -> Prompter:
    if isinstance(prompter, Prompter):
        return prompter
    if isinstance(prompter, str) and prompterTable.get(prompter, None):
        return prompterTable[prompter]()
    raise ValueError("You have to provide a Prompter in order to create LLMs.")


PrompterValidator = Annotated[Prompter, BeforeValidator(prompter_from_name)]


def _llm_from_name(llm: Union[LLM, str]) -> LLM:
    """
    Converts a list of names to corresponding llm instance.
    LLM names to be instantiated as LiteLLMs come from https://litellm.vercel.app/docs/providers
    """
    if isinstance(llm, LLM):
        return llm
    try:
        return get_llm_from_name(llm)
    except:
        raise ValueError(
            f"All name in the list must be convertible to LLM instance. {llm}"
        )


LLMValidator = Annotated[LLM, BeforeValidator(_llm_from_name)]


def _exporter_list(input: Union[str, Exporter]) -> List[Exporter]:
    exporter_class: Exporter = exporterTable.get(input, None)
    if not exporter_class:
        raise ValueError(
            f"All exporter in the list 'export_to' must be convertible to Exporter instance. {input}"
        )
    return exporter_class(**value)


ExporterListValidator = Annotated[list[Exporter], BeforeValidator(_exporter_list)]

from ragtime.generators import TextGenerator, generatorTable


def _generator(input: dict[str, any]):
    output: dict[str, TextGenerator]
    for key, value in input.items():
        generator: TextGenerator = generatorTable.get(key, None)
        if not generator:
            raise ValueError(f"Generator not found. {key}")
        output[key] = generator(**value)
    return output


GeneratorValidator = Annotated[dict[str, TextGenerator], BeforeValidator(_generator)]
