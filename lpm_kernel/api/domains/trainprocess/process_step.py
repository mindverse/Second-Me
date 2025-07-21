from enum import Enum
from typing import List


class ProcessStep(Enum):
    """Training process steps"""

    LIST_DOCUMENTS = "list_documents"
    GENERATE_DOCUMENT_EMBEDDINGS = "generate_document_embeddings"
    CHUNK_DOCUMENT = "process_chunks"
    CHUNK_EMBEDDING = "chunk_embedding"
    EXTRACT_DIMENSIONAL_TOPICS = "extract_dimensional_topics"
    GENERATE_SHADES = "generate_shades"
    GENERATE_BIOGRAPHY = "generate_biography"
    MODEL_DOWNLOAD = "model_download"

    GENERATE_BASE = "generate_base"
    BIO_QA_GENERATION = "bio_qa_generation"
    WIKI_DATA_GENERATION = "wiki_data_generation"
    GENERATE_MEMQA_ENTITY = "generate_memqa_entity"
    GENERATE_MEMQA_RELATION = "generate_memqa_relation"
    GENERATE_MEMQA_DESCRIPTION = "generate_memqa_description"
    GENERATE_MEMQA_DIVERSITY = "generate_memqa_diversity"
    SYNTHETIC_DATA_GENERATION = "synthetic_data_generation"
    SYNTHETIC_NO_NOTES_DATA_GENERATION = "synthetic_no_notes_data_generation"
    CONVERT_DATA = "convert_data"
    DATA_FILTERING = "data_filtering"

    TRAIN = "train"
    MERGE_WEIGHTS = "merge_weights"
    CONVERT_MODEL = "convert_model"

    @classmethod
    def get_ordered_steps(cls) -> List["ProcessStep"]:
        """Get ordered steps"""
        return [
            cls.MODEL_DOWNLOAD,
            cls.LIST_DOCUMENTS,
            cls.GENERATE_DOCUMENT_EMBEDDINGS,
            cls.CHUNK_DOCUMENT,
            cls.CHUNK_EMBEDDING,
            cls.EXTRACT_DIMENSIONAL_TOPICS,
            cls.GENERATE_SHADES,
            cls.GENERATE_BIOGRAPHY,
            cls.GENERATE_BASE,
            cls.BIO_QA_GENERATION,
            cls.WIKI_DATA_GENERATION,
            cls.GENERATE_MEMQA_ENTITY,
            cls.GENERATE_MEMQA_RELATION,
            cls.GENERATE_MEMQA_DESCRIPTION,
            cls.GENERATE_MEMQA_DIVERSITY,
            cls.SYNTHETIC_DATA_GENERATION,
            cls.SYNTHETIC_NO_NOTES_DATA_GENERATION,
            cls.CONVERT_DATA,
            cls.DATA_FILTERING,
            cls.TRAIN,
            cls.MERGE_WEIGHTS,
            cls.CONVERT_MODEL,
        ]

    def get_method_name(self) -> str:
        """Get the corresponding method name for this step"""
        return self.value
