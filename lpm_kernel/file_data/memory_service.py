from pathlib import Path
import os
import uuid
from datetime import datetime
from lpm_kernel.common.logging import logger
from lpm_kernel.models.memory import Memory
from lpm_kernel.common.repository.database_session import DatabaseSession
from lpm_kernel.file_data.process_factory import ProcessorFactory
from lpm_kernel.file_data.document_service import DocumentService
from lpm_kernel.file_data.document_dto import CreateDocumentRequest
from .process_status import ProcessStatus
from sqlalchemy import select


class StorageService:
    def __init__(self, config):
        self.config = config
        # get raw content directory configuration
        raw_content_dir = config.get("USER_RAW_CONTENT_DIR", "resources/raw_content")
        base_dir = config.get("LOCAL_BASE_DIR", ".")

        logger.info(f"Initializing storage service, base_dir: {base_dir}")
        logger.info(f"Raw content directory configuration: {raw_content_dir}")

        # if path is not absolute, build full path based on base_dir
        if not os.path.isabs(raw_content_dir):
            # replace environment variable
            raw_content_dir = raw_content_dir.replace("${RESOURCE_DIR}", "resources")
            raw_content_dir = raw_content_dir.replace(
                "${RAW_CONTENT_DIR}", "resources/raw_content"
            )
            # build full path based on base_dir
            raw_content_dir = os.path.join(base_dir, raw_content_dir)
            logger.info(f"Building complete path: {raw_content_dir}")

        # convert to Path object and ensure directory exists
        self.base_path = Path(raw_content_dir).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage path created: {self.base_path}")

        self.document_service = DocumentService()

    def check_file_exists(self, filename: str, filesize: int) -> Memory:
        """Check if file already exists

        Args:
            filename: file name
            filesize: file size

        Returns:
            Memory: if file exists, return corresponding Memory object; otherwise return None
        """
        db = DatabaseSession()
        with db._session_factory() as session:
            # find record with same file name and size
            query = select(Memory).where(
                Memory.name == filename, Memory.size == filesize
            )
            result = session.execute(query)
            memory = result.scalar_one_or_none()

            if memory:
                logger.info(f"Found duplicate file: {filename}, size: {filesize}")
                # check if file really exists
                if os.path.exists(memory.path):
                    return memory
                logger.warning(f"File in database does not exist on disk: {memory.path}")
            return None

    def save_file(self, file, metadata=None):
        """Save file and process document

        Args:
            file: uploaded file object
            metadata: file metadata

        Returns:
            tuple: (Memory object, Document object)

        Raises:
            ValueError: if file already exists
        """
        logger.info(f"Starting to save file: {file.filename}")
        logger.debug(f"File metadata: {metadata}")

        try:
            # get file size
            file.seek(0, os.SEEK_END)
            filesize = file.tell()
            file.seek(0)  # reset file pointer to start position

            # check if file already exists
            # existing_memory = self.check_file_exists(file.filename, filesize)
            # if existing_memory:
            #     raise ValueError(f"File '{file.filename}' already exists")

            # save file to disk
            filepath, filename, filesize = self._save_file_to_disk(file)
            logger.info(f"File saved to disk: {filepath}, size: {filesize} bytes")

            # create Memory record
            memory = None
            document = None

            db = DatabaseSession()
            session = db._session_factory()
            try:
                # create and save Memory record
                memory = Memory(
                    name=filename,
                    size=filesize,
                    path=str(filepath),
                    metadata=metadata or {},
                )
                session.add(memory)
                session.commit()
                logger.info(f"Memory record created successfully: {memory.id}")

                # process document
                document = self._process_document(filepath, filename, metadata)
                if document:
                    memory.document_id = document.id
                    session.add(memory)
                    session.commit()
                    logger.info(f"Memory record updated, associated document ID: {document.id}")

                # refresh memory object to ensure all fields are up to date
                session.refresh(memory)

            except Exception as e:
                session.rollback()
                logger.error(f"Database operation failed: {str(e)}", exc_info=True)
                raise
            finally:
                session.close()

            return memory, document

        except Exception as e:
            logger.error(f"Error occurred during file saving: {str(e)}", exc_info=True)
            raise

    def _save_file_to_disk(self, file):
        """Save file to disk

        Args:
            file: uploaded file object

        Returns:
            tuple: (file path, file name, file size)
        """
        try:
            # ensure directory exists
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensuring storage directory exists: {self.base_path}")

            # generate file name and path
            filename = file.filename
            # Add timestamp to filename to ensure uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = os.path.splitext(filename)
            timestamped_filename = f"{name_parts[0]}_{timestamp}{name_parts[1]}"
            filepath = self.base_path / timestamped_filename
            logger.info(f"Preparing to save file to: {filepath}")

            # save file
            file.save(str(filepath))
            filesize = os.path.getsize(filepath)
            logger.info(f"File saved successfully: {filepath}, size: {filesize} bytes")

            return filepath, filename, filesize

        except Exception as e:
            logger.error(f"Failed to save file to disk: {str(e)}", exc_info=True)
            raise

    def _process_document(self, filepath, filename, metadata=None):
        """Process document and create Document record

        Args:
            filepath: file path
            metadata: file metadata

        Returns:
            Document: created Document object, return None if processing fails
        """
        try:
            logger.info(f"Starting to process document: {filepath}")
            doc = ProcessorFactory.auto_detect_and_process(str(filepath))
            logger.info(
                f"Document processing completed, type: {doc.mime_type}, size: {doc.document_size}"
            )

            request = CreateDocumentRequest(
                name=filename,
                title=metadata.get("name", filename) if metadata else filename,
                mime_type=doc.mime_type,
                user_description=metadata.get("description", "Uploaded document")
                if metadata
                else "Uploaded document",
                document_size=doc.document_size,
                url=str(filepath),
                raw_content=doc.raw_content,
                extract_status=doc.extract_status,
                embedding_status=ProcessStatus.INITIALIZED,
            )

            saved_doc = self.document_service.create_document(request)
            logger.info(f"Document record created: {saved_doc.id}")
            return saved_doc

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}", exc_info=True)
            return None
