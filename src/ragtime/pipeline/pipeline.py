from ragtime.exporters import Json
from ragtime.expe import Expe

from pydantic import BaseModel, DirectoryPath
from pathlib import Path
from typing import Optional

from ragtime.pipeline.validator import GeneratorValidator


class Pipeline(BaseModel):
    file_name: str = None
    folder_name: Optional[DirectoryPath] = None
    generators: GeneratorValidator
    __steps: list[str] = ["questions", "answers", "facts", "evals"]

    def _next_steps(self, start_from: str = None, stop_after: str = None):
        start_from = start_from if (start_from in self.__steps) else self.__steps[0]
        stop_after = stop_after if (stop_after in self.__steps) else self.__steps[-1]

        start_from = self.__steps.index(start_from)
        stop_after = self.__steps.index(stop_after, start_from) + 1
        for step in self.__steps[start_from:stop_after]:
            generator = self.generators.get(step, None)
            if generator:
                yield generator

    def run(self, start_from: str = None, stop_after: str = None):
        next_input_file: Path = self.folder_name / self.file_name
        expe: Expe = Expe(next_input_file.parent, next_input_file.name)
        for generator in self._next_steps(start_from, stop_after):
            generator.generate(expe)
            for exporter in self.export_to:
                exporter.save(expe, next_input_file.parent, next_input_file.name)
            next_input_file = Json().save(
                expe, next_input_file.parent, next_input_file.name
            )
