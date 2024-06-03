from pathlib import Path
from typing import Any
import random

from ragtime.config import logger
from ragtime.generators.text_generator import *
from ragtime.retrievers.indexer import Indexer
from ragtime.expe import QA, Question
from llama_index.core import Document

from ragtime.config import FOLDER_QUESTIONS


class QuestionGenerator(TextGenerator):
    llm: LLM
    output_folder: str = FOLDER_QUESTIONS

    """
    Generate Questions from a set of documents.
    """

    docs_path: Path = None
    nb_quest: int = 10
    docs: list[Document] = []
    expe: Expe = Expe()  # ??
    indexer: Any = None

    def __init__(self, nb_quest: int, docs_path: Path, llms: list[LLM] = None):

        super().__init__(llms=llms)
        self.nb_quest = nb_quest
        if docs_path:
            self.docs_path = docs_path
            self.indexer = Indexer(name=self.docs_path)
            self.add_documents()

    def add_documents(self):

        documents, index = self.indexer.create_or_load_nodes(create_index=False)
        random.shuffle(documents)
        documents = documents[: min(self.nb_quest, len(documents))]
        for doc in documents:
            # Create a new QA object for each question
            qa: QA = QA()
            qa.question.meta = {"Node id": doc.id_} | doc.metadata | {"chunk": doc.text}
            self.expe.append(qa)

        self.docs = documents

    async def gen_for_qa(self, qa: QA):

        original_prefix: str = logger.prefix
        logger.prefix = f"{original_prefix}[QestGen][{self.llms[0].name}]"
        logger.info(f"* Start question generation")
        prev_question: Question = qa.question
        qa.question = await self.llm.generate(
            cur_obj=Question(),
            prev_obj=prev_question,
            qa=qa,
            start_from=self.start_from,
            b_missing_only=self.b_missing_only,
            chunk=qa.question.meta["chunk"],
        )
        qa.question.meta = prev_question.meta
