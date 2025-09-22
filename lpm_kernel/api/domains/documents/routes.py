import logging
from pathlib import Path

from dotenv import load_dotenv
from flask import Blueprint, jsonify, request
from flask_pydantic import validate

from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.configs.config import Config
from lpm_kernel.file_data.chunker import DocumentChunker
from lpm_kernel.file_data.document_service import document_service
from lpm_kernel.kernel.chunk_service import ChunkService

logger = logging.getLogger(__name__)
document_bp = Blueprint("documents", __name__, url_prefix="/api")

# Ensure .env file is loaded
load_dotenv()


@document_bp.route("/documents/list", methods=["GET"])
def list_documents():
    """
    List all documents
    Query Parameters:
        include_l0 (bool): Whether to include L0 data (chunks and embeddings)
    """
    try:
        # get query params
        include_l0 = request.args.get("include_l0", "").lower() == "true"
        if include_l0:
            documents = document_service.list_documents_with_l0()
            return jsonify(APIResponse.success(data=documents))
        else:
            documents = document_service.list_documents()
            return jsonify(
                APIResponse.success(data=[doc.to_dict() for doc in documents])
            )
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Error listing documents: {str(e)}"))


@document_bp.route("/documents/page", methods=["GET"])
def page_documents():
    """
    List documents with pagination
    Query Parameters:
        page (int): Page number
        page_size (int): Number of items per page
        include_l0 (bool): Whether to include L0 data (chunks and embeddings)
    """
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 100))
        include_l0 = request.args.get("include_l0", "").lower() == "true"
        if include_l0:
            documents = document_service.page_documents_with_l0(page, page_size)
            if not documents:
                return jsonify(
                    APIResponse.error(message="No documents found for the given page")
                )
            return jsonify(APIResponse.success(data=documents))
        else:
            documents = document_service.page_documents(page, page_size)
            if not documents:
                return jsonify(
                    APIResponse.error(message="No documents found for the given page")
                )
            return jsonify(
                APIResponse.success(data=[doc.to_dict() for doc in documents])
            )

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Error listing documents: {str(e)}"))


@document_bp.route("/documents/scan", methods=["POST"])
@validate()
def scan_documents():
    """Scan documents from configured directory and store them in database"""
    try:
        # 2. Get project root directory and construct the full path
        config = Config.from_env()
        relative_path = config.get("USER_RAW_CONTENT_DIR").lstrip("/")
        project_root = Path(__file__).parent.parent.parent.parent.parent
        full_path = project_root / relative_path

        # 3. Scan and process files
        processed_doc_dtos = document_service.scan_directory(
            directory_path=str(full_path), recursive=True
        )

        logger.info(f"Scan completed. Processed {len(processed_doc_dtos)} documents")

        # 4. Return processing results
        return jsonify(
            APIResponse.success(data=[doc_dto.dict() for doc_dto in processed_doc_dtos])
        )

    except Exception as e:
        logger.error(f"Unexpected error in scan_documents: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(message=f"Unexpected error in scan_documents: {str(e)}")
        )


@document_bp.route("/documents/analyze", methods=["POST"])
def analyze_documents():
    """Analyze all unanalyzed documents"""
    try:
        analyzed_doc_dtos = document_service.analyze_all_documents()
        return jsonify(
            APIResponse.success(
                data={
                    "total": len(analyzed_doc_dtos),
                    "documents": [doc.dict() for doc in analyzed_doc_dtos],
                }
            )
        )
    except Exception as e:
        logger.error(f"Error analyzing documents: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(message=f"Error analyzing documents: {str(e)}")
        )


@document_bp.route("/documents/<int:document_id>/l0", methods=["GET"])
def get_document_l0(document_id: int):
    """Get document L0 data including chunks and embeddings"""
    try:
        l0_data = document_service.get_document_l0(document_id)
        return jsonify(APIResponse.success(data=l0_data))
    except Exception as e:
        logger.error(f"Error getting document L0 data: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(message=f"Error getting document L0 data: {str(e)}")
        )


@document_bp.route("/documents/<int:document_id>/chunks", methods=["GET"])
def get_document_chunks(document_id: int):
    """Get chunks for the specified document"""
    try:
        logger.info(f"Attempting to retrieve chunks for document_id: {document_id}")

        chunks = document_service.get_document_chunks(document_id)

        if not chunks:
            logger.warning(f"No chunks found for document_id: {document_id}")
            return jsonify(
                APIResponse.error(message=f"No chunks found for document {document_id}")
            )

        return jsonify(
            APIResponse.success(
                data={
                    "document_id": document_id,
                    "total_chunks": len(chunks),
                    "chunks": chunks,
                }
            )
        )

    except Exception as e:
        logger.error(
            f"Error getting document chunks for document_id {document_id}: {str(e)}",
            exc_info=True,
        )
        return jsonify(
            APIResponse.error(
                message=f"Error getting document chunks for document_id {document_id}: {str(e)}"
            )
        )


@document_bp.route("/documents/chunks/process", methods=["POST"])
def process_all_chunks():
    """Process chunks for all documents in batch"""
    try:
        config = Config.from_env()
        chunker = DocumentChunker(
            chunk_size=int(config.get("DOCUMENT_CHUNK_SIZE")),
            overlap=int(config.get("DOCUMENT_CHUNK_OVERLAP")),
        )

        documents = document_service.list_documents()
        processed, failed = 0, 0

        chunk_service = ChunkService()
        for doc in documents:
            try:
                if not doc.raw_content:
                    logger.warning(f"Document {doc.id} has no content, skipping...")
                    failed += 1
                    continue

                # Split into chunks and save
                chunks = chunker.split(doc.raw_content)
                for chunk in chunks:
                    chunk.document_id = doc.id
                    chunk_service.save_chunk(chunk)

                processed += 1
                logger.info(
                    f"Document {doc.id} processed: {len(chunks)} chunks created"
                )

            except Exception as e:
                logger.error(f"Failed to process document {doc.id}: {str(e)}")
                failed += 1

        return jsonify(
            APIResponse.success(
                data={
                    "total": len(documents),
                    "processed": processed,
                    "failed": failed,
                }
            )
        )

    except Exception as e:
        logger.error(f"Chunk processing failed: {str(e)}")
        return jsonify(APIResponse.error(message=f"Chunk processing failed: {str(e)}"))


@document_bp.route("/documents/<int:document_id>/chunk/embedding", methods=["POST"])
def process_document_embeddings(document_id: int):
    """Process embeddings for all chunks of the specified document"""
    try:
        # Call service to process embeddings
        processed_chunks = document_service.generate_document_chunk_embeddings(
            document_id
        )

        if not processed_chunks:
            logger.warning(f"No chunks found for document {document_id}")
            return jsonify(
                APIResponse.error(message=f"No chunks found for document {document_id}")
            )

        return jsonify(
            APIResponse.success(
                data={
                    "document_id": document_id,
                    "total_chunks": len(processed_chunks),
                    "processed_chunks": len(
                        [c for c in processed_chunks if c.has_embedding]
                    ),
                }
            )
        )

    except Exception as e:
        logger.error(
            f"Error processing embeddings for document {document_id}: {str(e)}",
            exc_info=True,
        )
        return jsonify(
            APIResponse.error(
                message=f"Error processing embeddings for document {document_id}: {str(e)}"
            )
        )


@document_bp.route("/documents/<int:document_id>/chunk/embedding", methods=["GET"])
def get_document_embeddings(document_id: int):
    """Get embeddings status for all chunks of the specified document"""
    try:
        # Get query parameters, determine whether to return complete embedding vectors
        include_vectors = request.args.get("include_vectors", "").lower() == "true"

        chunks = document_service.get_document_chunks(document_id)
        if not chunks:
            return jsonify(
                APIResponse.error(message=f"No chunks found for document {document_id}")
            )

        # Get embeddings from ChromaDB
        chunk_embeddings = document_service.get_chunk_embeddings_by_document_id(
            document_id
        )

        chunks_info = [
            {
                "id": chunk.id,
                "content": chunk.content[:100] + "..."
                if len(chunk.content) > 100
                else chunk.content,
                "has_embedding": chunk.has_embedding,
                "embedding_length": len(chunk_embeddings.get(chunk.id, []))
                if chunk_embeddings.get(chunk.id)
                else 0,
                "embedding_vector": chunk_embeddings.get(chunk.id)
                if include_vectors
                else None,  # Decide whether to include vectors based on parameters
                "tags": chunk.tags,
                "topic": chunk.topic,
            }
            for chunk in chunks
        ]

        return jsonify(
            APIResponse.success(
                data={
                    "document_id": document_id,
                    "total_chunks": len(chunks),
                    "chunks_with_embeddings": len(
                        [c for c in chunks if c.has_embedding]
                    ),
                    "chunks": chunks_info,
                }
            )
        )

    except Exception as e:
        logger.error(
            f"Error getting embeddings for document {document_id}: {str(e)}",
            exc_info=True,
        )
        return jsonify(
            APIResponse.error(
                message=f"Error getting embeddings for document {document_id}: {str(e)}"
            )
        )


@document_bp.route("/documents/<int:document_id>/embedding", methods=["POST"])
def process_document_embedding(document_id: int):
    """Process document-level embedding"""
    try:
        embedding = document_service.process_document_embedding(document_id)
        if embedding is None:
            return jsonify(
                APIResponse.error(
                    message=f"Failed to process embedding for document {document_id}"
                )
            )

        return jsonify(
            APIResponse.success(
                data={"document_id": document_id, "embedding_length": len(embedding)}
            )
        )

    except ValueError as e:
        logger.error(f"Document not found: {str(e)}")
        return jsonify(APIResponse.error(message=f"Document not found: {str(e)}"))
    except Exception as e:
        logger.error(f"Error processing document embedding: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(message=f"Error processing document embedding: {str(e)}")
        )


@document_bp.route("/documents/<int:document_id>/embedding", methods=["GET"])
def get_document_embedding(document_id: int):
    """Get document-level embedding"""
    try:
        # Get query parameters, determine whether to return complete embedding vector
        include_vector = request.args.get("include_vector", "").lower() == "true"

        embedding = document_service.get_document_embedding(document_id)
        if embedding is None:
            return jsonify(
                APIResponse.error(
                    message=f"No embedding found for document {document_id}"
                )
            ), 404
        return jsonify(
            APIResponse.success(
                data={
                    "document_id": document_id,
                    "embedding_length": len(embedding),
                    "embedding_vector": embedding if include_vector else None,
                }
            )
        )

    except Exception as e:
        logger.error(f"Error getting document embedding: {str(e)}", exc_info=True)
        return jsonify(
            APIResponse.error(message=f"Error getting document embedding: {str(e)}")
        )


@document_bp.route("/documents/verify-embeddings", methods=["GET"])
def verify_document_embeddings():
    """Verify all document embeddings and return statistics"""
    try:
        verbose = request.args.get("verbose", "").lower() == "true"
        results = document_service.verify_document_embeddings(verbose=verbose)
        return jsonify(APIResponse.success(data=results))

    except Exception as e:
        logger.error(f"Error verifying document embeddings: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Error verifying document embeddings: {str(e)}"))


@document_bp.route("/documents/repair", methods=["POST"])
def repair_documents():
    """Repair documents with missing analysis and embeddings"""
    try:
        # First, fix missing document analysis (summaries and insights)
        fixed_analysis_count = document_service.fix_missing_document_analysis()
        
        # Get verification results after fixing analysis
        verification_results = document_service.verify_document_embeddings(verbose=False)
        
        # Process documents with missing embeddings
        documents_fixed = 0
        for doc in document_service._repository.list():
            embedding = document_service.get_document_embedding(doc.id)
            if doc.raw_content and embedding is None:
                try:
                    document_service.process_document_embedding(doc.id)
                    # Also process chunk embeddings
                    document_service.generate_document_chunk_embeddings(doc.id)
                    documents_fixed += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc.id} embedding: {str(e)}")
        
        # Get final verification results
        final_results = document_service.verify_document_embeddings(verbose=False)
        
        return jsonify(APIResponse.success(
            data={
                "analysis_fixed": fixed_analysis_count,
                "embeddings_fixed": documents_fixed,
                "initial_state": verification_results,
                "final_state": final_results
            }
        ))

    except Exception as e:
        logger.error(f"Error repairing documents: {str(e)}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Error repairing documents: {str(e)}"))
