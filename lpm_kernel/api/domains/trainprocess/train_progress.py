from typing import List, Dict, Optional, Union, Any
import json
from dataclasses import dataclass, field
from enum import Enum
from lpm_kernel.api.domains.trainprocess.progress_enum import Status


class TrainProgress:
    def __init__(self):
        # Define the complete data structure directly in the format matching the desired JSON output
        self.data = {
            "stages": [
                {
                    "name": "Downloading the Base Model",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Model Download",
                            "completed": False,
                            "status": "pending",
                            "have_output": False,
                            "path": None,
                            "current_file":None
                        }
                    ]
                },
                {
                    "name": "Activating the Memory Matrix",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "List Documents",
                            "completed": False,
                            "status": "pending",
                            "have_output": False,
                            "path": None
                        },
                        {
                            "name": "Generate Document Embeddings",
                            "completed": False,
                            "status": "pending",
                            "have_output": False,
                            "path": None
                        },
                        {
                            "name": "Process Chunks",
                            "completed": False,
                            "status": "pending",
                            "have_output": False,
                            "path": None
                        },
                        {
                            "name": "Chunk Embedding",
                            "completed": False,
                            "status": "pending",
                            "have_output": False,
                            "path": None
                        }
                    ]
                },
                {
                    "name": "Synthesize Your Life Narrative",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Extract Dimensional Topics",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/L2/data_pipeline/raw_data/topics.json"
                        },
                        {
                            "name": "Generate Biography",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "From database"
                        },
                        {
                            "name": "Map Your Entity Network",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/L1/graphrag_indexing_output/subjective/entities.parquet"
                        }
                    ]
                },
                {
                    "name": "Prepare Training Data for Deep Comprehension",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Decode Preference Patterns",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/L2/data/preference.json"
                        },
                        {
                            "name": "Reinforce Identity",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/L2/data/selfqa.json"
                        },
                        {
                            "name": "Augment Content Retention",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/L2/data/diversity.json"
                        }
                    ]
                },
                {
                    "name": "Training to create Second Me",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Train",
                            "completed": False,
                            "status": "pending",
                            "have_output": False,
                            "path": None
                        },
                        {
                            "name": "Merge Weights",
                            "completed": False,
                            "status": "pending",
                            "have_output": False,
                            "path": None
                        },
                        {
                            "name": "Convert Model",
                            "completed": False,
                            "status": "pending",
                            "have_output": False,
                            "path": None
                        }
                    ]
                }
            ],
            "overall_progress": 0.0,
            "current_stage": None,
            "status": "pending"
        }
        
        # Create stage name to stage data mapping
        self.stage_map = {}
        for stage in self.data["stages"]:
            stage_name = stage["name"].lower().replace(" ", "_")
            self.stage_map[stage_name] = stage
            
        # Create step name to step data mapping for each stage
        self.steps_map = {}
        for stage_name, stage in self.stage_map.items():
            self.steps_map[stage_name] = {}
            for step in stage["steps"]:
                step_name = step["name"].lower().replace(" ", "_")
                self.steps_map[stage_name][step_name] = step

    def update_progress(self, stage: str, step: str, currentStepStatus: Union[Status, str], stageProgress: Optional[float] = None, file_name: Optional[str] = None):
        """Update progress status
        Args:
            stage: Stage key (snake_case format)
            step: Step key (snake_case format)
            currentStepStatus: Status (enum or string)
            stageProgress: Optional progress value (0-100)
            file_name: Optional name of the file being processed (for download tracking)
        """
        stage_data = self.stage_map[stage]
        status_value = currentStepStatus.value if isinstance(currentStepStatus, Status) else currentStepStatus
        step_data = self.steps_map[stage][step]
        
        # Update step status
        step_data["status"] = status_value
        step_data["completed"] = status_value == "completed"
        
        # Update file name if provided (for tracking downloads)
        if file_name is not None:
            step_data["current_file"] = file_name
        
        # Update stage progress
        self._update_stage_progress(stage_data, stageProgress)
        
        # Update stage status and current step
        self._update_stage_status(stage_data, step_data)
        
        # Update overall progress
        self._update_overall_progress()
        
        # Update overall status
        self._update_overall_status()

    def _update_stage_progress(self, stage_data: Dict, stageProgress: Optional[float] = None):
        """Update the progress of a stage
        
        Args:
            stage_data: Stage data dictionary
            stageProgress: Optional progress value (0-100)
        """
        if stageProgress is not None:
            stage_data["progress"] = stageProgress
        else:
            completed_steps = sum(1 for s in stage_data["steps"] if s["completed"])
            total_steps = len(stage_data["steps"])
            stage_data["progress"] = (completed_steps / total_steps) * 100.0

    def _update_stage_status(self, stage_data: Dict, step_data: Dict):
        """Update the status and current step of a stage
        
        Args:
            stage_data: Stage data dictionary
            step_data: Step data dictionary
        """
        if all(step["completed"] for step in stage_data["steps"]):
            stage_data["status"] = "completed"
            stage_data["current_step"] = None
            next_stage = None
            for stage_name, stage_info in self.stage_map.items():
                if stage_info["status"] != "completed":
                    next_stage = stage_name
                    break
            self.data["current_stage"] = next_stage
        elif any(step["status"] == "failed" for step in stage_data["steps"]):
            stage_data["status"] = "failed"
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]
        elif any(step["status"] == "suspended" for step in stage_data["steps"]):
            stage_data["status"] = "suspended"
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]
        else:
            stage_data["status"] = "in_progress"
            stage_data["current_step"] = step_data["name"]
            self.data["current_stage"] = stage_data["name"]

    def _update_overall_progress(self):
        """Update the overall progress based on all stages"""
        completed_progress = sum(s["progress"] for s in self.data["stages"])
        self.data["overall_progress"] = completed_progress / len(self.data["stages"])

    def _update_overall_status(self):
        """Update the overall status based on all stages"""
        if all(s["status"] == "completed" for s in self.data["stages"]):
            self.data["status"] = "completed"
        elif any(s["status"] == "failed" for s in self.data["stages"]):
            self.data["status"] = "failed"
        elif any(s["status"] == "suspended" for s in self.data["stages"]):
            self.data["status"] = "suspended"
        elif any(s["status"] == "in_progress" for s in self.data["stages"]):
            self.data["status"] = "in_progress"
        else:
            self.data["status"] = "pending"

    def to_dict(self) -> dict:
        """Convert progress status to dictionary format"""
        return self.data
    
    def reset(self):
        """Reset all progress statuses"""
        self.__init__()
