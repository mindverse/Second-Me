from enum import Enum
import json
import os
from typing import Dict, List, Optional

from lpm_kernel.api.domains.trainprocess.progress_enum import Status
from lpm_kernel.api.domains.trainprocess.train_progress import TrainProgress
from lpm_kernel.api.domains.trainprocess.process_step import ProcessStep
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()

class TrainProgressHolder:
    """Progress management class"""

    def __init__(self, model_name: str = None):
        progress_dir = os.path.join(os.getcwd(), "data", "progress")
        if not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
        
        # Generate progress file name based on model name
        progress_file = "trainprocess_progress.json"  # Default name
        if model_name:
            progress_file = f"trainprocess_progress_{model_name}.json"
            
        self.progress_file = os.path.normpath(os.path.join(progress_dir, progress_file))
        if not self.progress_file.startswith(progress_dir):
            raise ValueError("Invalid progress file path")
        self.progress = TrainProgress()

        # Stage mapping for process steps
        self._stage_mapping = {
            ProcessStep.MODEL_DOWNLOAD: "downloading_the_base_model",

            ProcessStep.LIST_DOCUMENTS: "activating_the_memory_matrix",
            ProcessStep.GENERATE_DOCUMENT_EMBEDDINGS: "activating_the_memory_matrix",
            ProcessStep.CHUNK_DOCUMENT: "activating_the_memory_matrix",
            ProcessStep.CHUNK_EMBEDDING: "activating_the_memory_matrix",

            ProcessStep.EXTRACT_DIMENSIONAL_TOPICS: "synthesize_your_life_narrative",
            ProcessStep.GENERATE_SHADES: "synthesize_your_life_narrative",
            ProcessStep.GENERATE_BIOGRAPHY: "synthesize_your_life_narrative",
            
            ProcessStep.GENERATE_BASE: "memory_reconstruction",
            
            ProcessStep.BIO_QA_GENERATION: "deep_comprehension",
            ProcessStep.WIKI_DATA_GENERATION: "deep_comprehension",
            ProcessStep.GENERATE_MEMQA_ENTITY: "deep_comprehension",
            ProcessStep.GENERATE_MEMQA_RELATION: "deep_comprehension",
            ProcessStep.GENERATE_MEMQA_DESCRIPTION: "deep_comprehension",
            ProcessStep.GENERATE_MEMQA_DIVERSITY: "deep_comprehension",
            
            ProcessStep.SYNTHETIC_DATA_GENERATION: "memory_expansion",
            ProcessStep.SYNTHETIC_NO_NOTES_DATA_GENERATION: "memory_expansion",
            ProcessStep.CONVERT_DATA: "memory_expansion",
            ProcessStep.DATA_FILTERING: "memory_expansion",

            ProcessStep.TRAIN: "training_to_create_second_me",
            ProcessStep.MERGE_WEIGHTS: "training_to_create_second_me",
            ProcessStep.CONVERT_MODEL: "training_to_create_second_me",
        }
        
        self._load_progress()

    def _load_progress(self):
        """Load progress file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    saved_progress = json.load(f)
                    self.progress.data = saved_progress
                    logger.debug(self.progress.data)
                    
                    self.progress.stage_map = {}
                    for stage in self.progress.data["stages"]:
                        stage_name = stage["name"].lower().replace(" ", "_")
                        self.progress.stage_map[stage_name] = stage
                    
                    self.progress.steps_map = {}
                    for stage_name, stage in self.progress.stage_map.items():
                        self.progress.steps_map[stage_name] = {}
                        for step in stage["steps"]:
                            step_name = step["name"].lower().replace(" ", "_")
                            self.progress.steps_map[stage_name][step_name] = step                    
                    # Check and reset any in_progress status to failed
                    self._reset_in_progress_status()
            except Exception as e:
                logger.error(f"Error loading progress: {str(e)}")
                # Reset progress on any error
                self.progress = TrainProgress()
                
    def _reset_in_progress_status(self):
        """Reset any in_progress status to failed after loading from file"""
        need_save = False
        
        # Check overall status
        if self.progress.data["status"] == "in_progress":
            self.progress.data["status"] = "failed"
            need_save = True
            logger.info("Reset overall in_progress status to failed")
        
        # Check each stage
        for stage in self.progress.data["stages"]:
            if stage["status"] == "in_progress":
                stage["status"] = "failed"
                need_save = True
                logger.info(f"Reset stage '{stage['name']}' in_progress status to failed")
            
            # Check each step in the stage
            for step in stage["steps"]:
                if step["status"] == "in_progress":
                    step["status"] = "failed"
                    step["completed"] = False
                    need_save = True
                    logger.info(f"Reset step '{step['name']}' in_progress status to failed")
        
        # Save changes if any were made
        if need_save:
            progress_dict = self.progress.to_dict()
            with open(self.progress_file, "w") as f:
                json.dump(progress_dict, f, indent=2)
            logger.info("Saved progress after resetting in_progress statuses")

    def _save_progress(self):
        """Save progress"""
        progress_dict = self.progress.to_dict()
        try:
            with open(self.progress_file, "w") as f:
                json.dump(progress_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress to {self.progress_file}: {str(e)}", exc_info=True)

    def is_step_completed(self, step: ProcessStep) -> bool:
        """Check if a step is completed"""
        stage_name = self._stage_mapping[step]
        step_name = step.value
        step_info = self.progress.steps_map[stage_name][step_name]
        return step_info.get("completed", False)

    def mark_step_status(self, step: ProcessStep, status: Status):
        """Mark a step with the specified status
        
        Args:
            step: The process step to mark
            status: The status to set for the step
        """
        stage_name = self._stage_mapping[step]
        step_name = step.value
        self.progress.update_progress(stage_name, step_name, status)
        self._save_progress()

    def reset_progress(self):
        """Reset all progress"""
        logger.info(f"Resetting progress for model: {self.progress_file}")
        self.progress = TrainProgress()
        self._save_progress()

    def get_last_successful_step(self) -> Optional[ProcessStep]:
        """Get the last successfully completed step"""
        ordered_steps = ProcessStep.get_ordered_steps()
        for step in reversed(ordered_steps):
            if self.is_step_completed(step):
                return step
        return None
