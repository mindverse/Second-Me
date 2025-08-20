import os
import json
import logging
import psutil
import time
import subprocess
import torch  # Add torch import for CUDA detection
import threading
import queue
from typing import Iterator, Any, Optional, Generator, Dict
from datetime import datetime
from flask import Response
from openai import OpenAI
from lpm_kernel.api.domains.kernel2.dto.server_dto import ServerStatus, ProcessInfo
from lpm_kernel.configs.config import Config
import uuid

logger = logging.getLogger(__name__)

class LocalLLMService:
    """Service for managing local LLM client and server"""
    
    def __init__(self):
        self._client = None
        self._stopping_server = False
        
    @property
    def client(self) -> OpenAI:
        config = Config.from_env()
        """Get the OpenAI client for local LLM server"""
        if self._client is None:
            base_url = config.get("LOCAL_LLM_SERVICE_URL")
            if not base_url:
                raise ValueError("LOCAL_LLM_SERVICE_URL environment variable is not set")
                
            self._client = OpenAI(
                base_url=base_url,
                api_key="sk-no-key-required"
            )
        return self._client

    def start_server(self, model_path: str, use_gpu: bool = True) -> bool:
        """
        Start the llama-server service with GPU acceleration when available
        
        Args:
            model_path: Path to the GGUF model file
            use_gpu: Whether to use GPU acceleration if available
            
        Returns:
            bool: True if server started successfully, False otherwise
        """
        try:
            # Check if server is already running
            status = self.get_server_status()
            if status.is_running:
                logger.info("LLama server is already running")
                return True

            # Check for CUDA availability if GPU was requested
            cuda_available = torch.cuda.is_available() if use_gpu else False
            cuda_available = False
            gpu_info = ""
            
            if use_gpu and cuda_available:
                gpu_device = torch.cuda.current_device()
                gpu_info = f" using GPU: {torch.cuda.get_device_name(gpu_device)}"
                gpu_memory = torch.cuda.get_device_properties(gpu_device).total_memory / (1024**3)
                
                logger.info(f"CUDA is available. Using GPU acceleration{gpu_info}")
                logger.info(f"CUDA device capabilities: {torch.cuda.get_device_capability(gpu_device)}")
                logger.info(f"CUDA memory: {gpu_memory:.2f} GB")
                
                # Pre-initialize CUDA to speed up first inference
                logger.info("Pre-initializing CUDA context to speed up first inference")
                torch.cuda.init()
                torch.cuda.empty_cache()
            elif use_gpu and not cuda_available:
                logger.warning("CUDA was requested but is not available. Using CPU instead.")
            else:
                logger.info("Using CPU for inference (GPU not requested)")

            # Check for GPU optimization marker
            gpu_optimized = False
            model_dir = os.path.dirname(model_path)
            gpu_marker_path = os.path.join(model_dir, "gpu_optimized.json")
            if os.path.exists(gpu_marker_path):
                try:
                    with open(gpu_marker_path, 'r') as f:
                        gpu_data = json.load(f)
                        if gpu_data.get("gpu_optimized", False):
                            gpu_optimized = True
                            logger.info(f"Found GPU optimization marker created on {gpu_data.get('optimized_on', 'unknown date')}")
                except Exception as e:
                    logger.warning(f"Error reading GPU marker file: {e}")

            # Get the correct path to the llama-server executable
            base_dir = os.getcwd()
            server_path = os.path.join(base_dir, "llama.cpp", "build", "bin", "llama-server")
            
            # For Windows, add .exe extension if needed
            if os.name == 'nt' and not server_path.endswith('.exe'):
                server_path += '.exe'
                
            # Verify executable exists
            if not os.path.exists(server_path):
                logger.error(f"llama-server executable not found at: {server_path}")
                return False
                
            # Start server with optimal parameters for faster startup
            cmd = [
                server_path,
                "-m", model_path,
                "--host", "0.0.0.0",
                "--port", "8080",
                "--ctx-size", "2048",     # Default context size (adjust based on needs)
                "--parallel", "2",        # Enable request parallelism
                "--cont-batching"         # Enable continuous batching
            ]
            
            # Set up environment with CUDA variables to ensure GPU detection
            env = os.environ.copy()
            if not (cuda_available and use_gpu):
                # Only set for CPU mode to avoid conflicts
                if "CUDA_VISIBLE_DEVICES" in env:
                    del env["CUDA_VISIBLE_DEVICES"]
            
            # Add GPU-related parameters if CUDA is available
            if cuda_available and use_gpu:
                # Force GPU usage with optimal parameters for faster loads
                cmd.extend([
                    "--n-gpu-layers", "999",  # Use all layers on GPU
                    "--tensor-split", "0",    # Use the first GPU for all operations
                    "--main-gpu", "0",        # Use GPU 0 as the primary device
                    "--mlock"                 # Lock memory to prevent swapping during inference
                ])
                
                # Set CUDA environment variables to help with GPU detection
                env["CUDA_VISIBLE_DEVICES"] = "0"  # Force using first GPU
                
                # Ensure comprehensive library paths for CUDA
                cuda_lib_paths = [
                    "/usr/local/cuda/lib64",
                    "/usr/lib/cuda/lib64",
                    "/usr/local/lib",
                    "/usr/lib/x86_64-linux-gnu",
                    "/usr/lib/wsl/lib"  # For Windows WSL environments
                ]
                
                # Build a comprehensive LD_LIBRARY_PATH
                current_ld_path = env.get("LD_LIBRARY_PATH", "")
                for path in cuda_lib_paths:
                    if os.path.exists(path) and path not in current_ld_path:
                        current_ld_path = f"{path}:{current_ld_path}" if current_ld_path else path
                
                env["LD_LIBRARY_PATH"] = current_ld_path
                logger.info(f"Setting LD_LIBRARY_PATH to: {current_ld_path}")
                
                # If this is Windows, use different approach for CUDA libraries
                if os.name == 'nt':
                    # Windows typically has CUDA in PATH already if installed
                    logger.info("Windows system detected, using system CUDA libraries")
                else:
                    # On Linux, try to find CUDA libraries in common locations
                    for cuda_path in [
                        # Common CUDA paths
                        "/usr/local/cuda/lib64",
                        "/usr/lib/cuda/lib64",
                        "/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
                        "/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
                    ]:
                        if os.path.exists(cuda_path):
                            # Add CUDA path to library path
                            env["LD_LIBRARY_PATH"] = f"{cuda_path}:{env.get('LD_LIBRARY_PATH', '')}"
                            env["CUDA_HOME"] = os.path.dirname(cuda_path)
                            logger.info(f"Found CUDA at {cuda_path}, setting environment variables")
                            break

                # NOTE: CUDA support and rebuild should be handled at build/setup time (e.g., Docker build or setup script).
                # The runtime check and rebuild logic has been removed for efficiency and reliability.
                # Ensure llama.cpp is built with CUDA support before running the server if GPU is required.

                # Pre-heat GPU to ensure faster initial response
                if torch.cuda.is_available():
                    logger.info("Pre-warming GPU to reduce initial latency...")
                    dummy_tensor = torch.zeros(1, 1).cuda()
                    del dummy_tensor
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    logger.info("GPU warm-up complete")
                
                logger.info("Using GPU acceleration for inference with optimized settings")
            else:
                # If GPU isn't available or supported, optimize for CPU
                cmd.extend([
                    "--threads", str(max(1, os.cpu_count() - 1)),  # Use all CPU cores except one
                ])
                logger.info(f"Using CPU-only mode with {max(1, os.cpu_count() - 1)} threads")
            
            logger.info(f"Starting llama-server with command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env
            )
            
            # Wait for server to start (longer wait for GPU initialization)
            wait_time = 5 if cuda_available and use_gpu else 3
            logger.info(f"Waiting {wait_time} seconds for server to start...")
            time.sleep(wait_time)
            
            # Check if process is still running
            if process.poll() is None:
                # Log initialization success
                if cuda_available and use_gpu:
                    logger.info(f"✅ LLama server started successfully with GPU acceleration{gpu_info}")
                else:
                    logger.info("✅ LLama server started successfully in CPU-only mode")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Failed to start llama-server: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting llama-server: {str(e)}")
            return False

    def stop_server(self) -> ServerStatus:
        """
        Stop the llama-server service.
        Find and forcibly terminate all llama-server processes
        
        Returns:
            ServerStatus: Service status object containing information about whether processes are still running
        """
        try:
            if self._stopping_server:
                logger.info("Server is already in the process of stopping")
                return self.get_server_status()
            
            self._stopping_server = True
        
            try:
                # Find all possible llama-server processes and forcibly terminate them
                terminated_pids = []
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    try:
                        cmdline = proc.cmdline()
                        if any("llama-server" in cmd for cmd in cmdline):
                            pid = proc.pid
                            logger.info(f"Force terminating llama-server process, PID: {pid}")
                            
                            # Directly use kill signal to forcibly terminate
                            proc.kill()
                            
                            # Ensure the process has been terminated
                            try:
                                proc.wait(timeout=0.2)  # Slightly increase wait time to ensure process termination
                                terminated_pids.append(pid)
                                logger.info(f"Successfully terminated llama-server process {pid}")
                            except psutil.TimeoutExpired:
                                # If timeout, try to terminate again
                                logger.warning(f"Process {pid} still running, sending SIGKILL again")
                                try:
                                    import os
                                    import signal
                                    os.kill(pid, signal.SIGKILL)  # Use system-level SIGKILL signal
                                    terminated_pids.append(pid)
                                    logger.info(f"Successfully force killed llama-server process {pid} with SIGKILL")
                                except ProcessLookupError:
                                    # Process no longer exists
                                    terminated_pids.append(pid)
                                    logger.info(f"Process {pid} no longer exists after kill attempt")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                
                if terminated_pids:
                    logger.info(f"Terminated llama-server processes: {terminated_pids}")
                else:
                    logger.info("No running llama-server process found")
                
                # Check again if any llama-server processes are still running
                return self.get_server_status()
            
            finally:
                self._stopping_server = False
            
        except Exception as e:
            logger.error(f"Error stopping llama-server: {str(e)}")
            self._stopping_server = False
            return ServerStatus.not_running()

    def get_server_status(self) -> ServerStatus:
        """
        Get the current status of llama-server
        Returns: ServerStatus object
        """
        try:
            base_dir = os.getcwd()
            server_path = os.path.join(base_dir, "llama.cpp", "build", "bin", "llama-server")
            server_exec_name = os.path.basename(server_path)
            
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.cmdline()
                    # Check both for the executable name and the full path
                    if any(server_exec_name in cmd for cmd in cmdline) or any("llama-server" in cmd for cmd in cmdline):
                        with proc.oneshot():
                            process_info = ProcessInfo(
                                pid=proc.pid,
                                cpu_percent=proc.cpu_percent(),
                                memory_percent=proc.memory_percent(),
                                create_time=proc.create_time(),
                                cmdline=cmdline,
                            )
                            return ServerStatus.running(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
            return ServerStatus.not_running()
            
        except Exception as e:
            logger.error(f"Error checking llama-server status: {str(e)}")
            return ServerStatus.not_running()

    def _parse_response_chunk(self, chunk):
        """Parse different response chunk formats into a standardized format."""
        try:
            if chunk is None:
                logger.warning("Received None chunk")
                return None
                
            # logger.info(f"Parsing response chunk: {chunk}")
            # Handle custom format
            if isinstance(chunk, dict) and "type" in chunk and chunk["type"] == "chat_response":
                logger.info(f"Processing custom format response: {chunk}")
                return {
                    "id": str(uuid.uuid4()),  # Generate a unique ID
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "models/lpm",
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk.get("content", "")
                            },
                            "finish_reason": "stop" if chunk.get("done", False) else None
                        }
                    ]
                }
            
            # Handle OpenAI format
            if not hasattr(chunk, 'choices'):
                logger.warning(f"Chunk has no choices attribute: {chunk}")
                return None
                
            choices = getattr(chunk, 'choices', [])
            if not choices:
                logger.warning("Chunk has empty choices")
                return None
                
            # logger.info(f"Processing OpenAI format response: choices={choices}")
            delta = choices[0].delta
            
            # Create standard response structure
            response_data = {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "models/lpm",
                "system_fingerprint": chunk.system_fingerprint if hasattr(chunk, 'system_fingerprint') else None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            # Keep even if content is None, let the client handle it
                            "content": delta.content if hasattr(delta, 'content') else ""
                        },
                        "finish_reason": choices[0].finish_reason
                    }
                ]
            }
            
            # If there is neither content nor finish_reason, skip
            if not (hasattr(delta, 'content') or choices[0].finish_reason):
                logger.debug("Skipping chunk with no content and no finish_reason")
                return None
                
            return response_data
            
        except Exception as e:
            logger.error(f"Error parsing response chunk: {e}, chunk: {chunk}")
            return None

    def handle_stream_response(self, response_iter: Iterator[Any]) -> Response:
        """Handle streaming response from the LLM server"""
        # Create a queue for thread communication
        message_queue = queue.Queue()
        # Create an event flag to notify when model processing is complete
        completion_event = threading.Event()
        # Create a variable to track if heartbeat is needed after first response
        first_response_received = False
        
        def heartbeat_thread():
            """Thread function for sending heartbeats"""
            start_time = time.time()
            heartbeat_interval = 10  # Send heartbeat every 10 seconds
            heartbeat_count = 0
            
            logger.info("[STREAM_DEBUG] Heartbeat thread started")
            
            try:
                # Send initial heartbeat
                message_queue.put((b": initial heartbeat\n\n", "[INITIAL_HEARTBEAT]"))
                last_heartbeat_time = time.time()
                
                while not completion_event.is_set():
                    current_time = time.time()
                    
                    # Check if we need to send a heartbeat
                    if current_time - last_heartbeat_time >= heartbeat_interval:
                        heartbeat_count += 1
                        elapsed = current_time - start_time
                        logger.info(f"[STREAM_DEBUG] Sending heartbeat #{heartbeat_count} at {elapsed:.2f}s")
                        message_queue.put((f": heartbeat #{heartbeat_count}\n\n".encode('utf-8'), "[HEARTBEAT]"))
                        last_heartbeat_time = current_time
                    
                    # Short sleep to prevent CPU spinning
                    time.sleep(0.1)
                
                logger.info(f"[STREAM_DEBUG] Heartbeat thread stopping after {heartbeat_count} heartbeats")
            except Exception as e:
                logger.error(f"[STREAM_DEBUG] Error in heartbeat thread: {str(e)}", exc_info=True)
                message_queue.put((f"data: {{\"error\": \"Heartbeat error: {str(e)}\"}}\n\n".encode('utf-8'), "[ERROR]"))
        
        def model_response_thread():
            """Thread function for processing model responses"""
            chunk = None
            start_time = time.time()
            chunk_count = 0
            
            try:
                logger.info("[STREAM_DEBUG] Model response thread started")
                
                # Process model responses
                for chunk in response_iter:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    chunk_count += 1
                    
                    logger.info(f"[STREAM_DEBUG] Received chunk #{chunk_count} after {elapsed_time:.2f}s")
                    
                    if chunk is None:
                        logger.warning("[STREAM_DEBUG] Received None chunk, skipping")
                        continue
                    
                    # Check if it's an end marker
                    if chunk == "[DONE]":
                        logger.info(f"[STREAM_DEBUG] Received [DONE] marker after {elapsed_time:.2f}s")
                        message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
                        break
                    
                    # Handle error responses
                    if isinstance(chunk, dict) and "error" in chunk:
                        logger.warning(f"[STREAM_DEBUG] Received error response: {chunk}")
                        data_str = json.dumps(chunk)
                        message_queue.put((f"data: {data_str}\n\n".encode('utf-8'), "[ERROR]"))
                        message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
                        break
                    
                    # Handle normal responses
                    response_data = self._parse_response_chunk(chunk)
                    if response_data:
                        data_str = json.dumps(response_data)
                        content = response_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        content_length = len(content) if content else 0
                        logger.info(f"[STREAM_DEBUG] Sending chunk #{chunk_count}, content length: {content_length}, elapsed: {elapsed_time:.2f}s")
                        message_queue.put((f"data: {data_str}\n\n".encode('utf-8'), "[CONTENT]"))
                    else:
                        logger.warning(f"[STREAM_DEBUG] Parsed response data is None for chunk #{chunk_count}")
                
                # Handle the case where no responses were received
                if chunk_count == 0:
                    logger.info("[STREAM_DEBUG] No chunks received, sending empty message")
                    thinking_message = {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": "models/lpm",
                        "system_fingerprint": None,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": ""  # Empty content won't affect frontend display
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    data_str = json.dumps(thinking_message)
                    message_queue.put((f"data: {data_str}\n\n".encode('utf-8'), "[THINKING]"))
                
                # Model processing is complete, send end marker
                if chunk != "[DONE]":
                    logger.info(f"[STREAM_DEBUG] Sending final [DONE] marker after {elapsed_time:.2f}s")
                    message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
                
            except Exception as e:
                logger.error(f"[STREAM_DEBUG] Error processing model response: {str(e)}", exc_info=True)
                message_queue.put((f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode('utf-8'), "[ERROR]"))
                message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
            finally:
                # Set completion event to notify heartbeat thread to stop
                completion_event.set()
                logger.info(f"[STREAM_DEBUG] Model response thread completed with {chunk_count} chunks")
        
        def generate():
            """Main generator function for generating responses"""
            # Start heartbeat thread
            heart_thread = threading.Thread(target=heartbeat_thread, daemon=True)
            heart_thread.start()
            
            # Start model response processing thread
            model_thread = threading.Thread(target=model_response_thread, daemon=True)
            model_thread.start()
            
            try:
                # Get messages from queue and return to client
                while True:
                    try:
                        # Use short timeout to get message, prevent blocking
                        message, message_type = message_queue.get(timeout=0.1)
                        logger.debug(f"[STREAM_DEBUG] Yielding message type: {message_type}")
                        yield message
                        
                        # If end marker is received, exit loop
                        if message_type == "[DONE]":
                            logger.info("[STREAM_DEBUG] Received [DONE] marker, ending generator")
                            break
                    except queue.Empty:
                        # Queue is empty, continue trying to get message
                        # Check if model thread has completed but didn't send [DONE]
                        if completion_event.is_set() and not model_thread.is_alive():
                            logger.warning("[STREAM_DEBUG] Model thread completed without [DONE], ending generator")
                            yield b"data: [DONE]\n\n"
                            break
                        pass
            except GeneratorExit:
                # Client closed connection
                logger.info("[STREAM_DEBUG] Client closed connection (GeneratorExit)")
                completion_event.set()
            except Exception as e:
                logger.error(f"[STREAM_DEBUG] Error in generator: {str(e)}", exc_info=True)
                try:
                    yield f"data: {{\"error\": \"Generator error: {str(e)}\"}}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                except:
                    pass
                completion_event.set()
            finally:
                # Ensure completion event is set
                completion_event.set()
                # Wait for threads to complete
                if heart_thread.is_alive():
                    heart_thread.join(timeout=1.0)
                if model_thread.is_alive():
                    model_thread.join(timeout=1.0)
                logger.info("[STREAM_DEBUG] Generator completed")
        
        # Return response
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache, no-transform',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
                'Transfer-Encoding': 'chunked'
            }
        )


# Global instance
local_llm_service = LocalLLMService()
