from typing import Dict, Optional, Union

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
                            "current_file": None
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
                            "path": "resources/data/stage2/topics/topic.json"
                        },
                        {
                            "name": "Generate Shades",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "From database"
                        },
                        {
                            "name": "Generate Biography",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "From database"
                        }
                    ]
                },
                {
                    "name": "Memory Reconstruction",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Generate Base",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/stage1/final.json"
                        },
                    ]
                },
                {
                    "name": "Deep Comprehension",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Bio QA Generation",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/stage2/processed/stage2_qa.json"
                        },
                        {
                            "name": "Wiki Data Generation",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/stage2/wiki/wiki_res.json"
                        },
                        {
                            "name": "Generate MemQA Entity",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/stage2/processed/subjective_entity.json"
                        },
                        {
                            "name": "Generate MemQA Relation",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/stage2/processed/subjective_relation.json"
                        },
                        {
                            "name": "Generate MemQA Description",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/stage2/processed/subjective_description.json"
                        },
                        {
                            "name": "Generate MemQA Diversity",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/stage2/processed/diversity.json"
                        }
                    ]
                },
                {
                    "name": "Memory Expansion",
                    "progress": 0.0,
                    "status": "pending",
                    "current_step": None,
                    "steps": [
                        {
                            "name": "Synthetic Data Generation",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/stage3/synthetic_data_with_notes_answers.json"
                        },
                        {
                            "name": "Synthetic No Notes Data Generation",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/stage3/synthetic_data_no_notes_answers.json"
                        },
                        {
                            "name": "Convert Data",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/merged.json"
                        },
                        {
                            "name": "Data Filtering",
                            "completed": False,
                            "status": "pending",
                            "have_output": True,
                            "path": "resources/data/merged.json"
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

    def update_progress(self, stage: str, step: str, currentStepStatus: Union[Status, str],
                        stageProgress: Optional[float] = None, file_name: Optional[str] = None):
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

    def update_step_progress(self, stage: str, step: str, current: int, total: int, description: Optional[str] = None):
        """Update progress within a step (for real-time progress tracking)
        
        Args:
            stage: Stage key (snake_case format)
            step: Step key (snake_case format)  
            current: Current item being processed
            total: Total items to process
            description: Optional description of what is being processed
        """
        stage_data = self.stage_map[stage]
        step_data = self.steps_map[stage][step]

        # Calculate step progress
        step_progress = (current / total) * 100.0 if total > 0 else 0.0

        # Update step with progress info
        step_data["current"] = current
        step_data["total"] = total
        step_data["step_progress"] = step_progress
        if description:
            step_data["description"] = description

        # Update step status to in_progress if not already completed/failed
        if step_data["status"] in ["pending"]:
            step_data["status"] = "in_progress"

        # Calculate stage progress based on step completion and current step progress
        completed_steps = sum(1 for s in stage_data["steps"] if s["completed"])
        total_steps = len(stage_data["steps"])

        # Find current step index
        current_step_index = None
        for i, s in enumerate(stage_data["steps"]):
            if s["name"] == step_data["name"]:
                current_step_index = i
                break

        if current_step_index is not None:
            # Progress = completed steps + current step progress
            stage_progress = (completed_steps + (step_progress / 100.0)) / total_steps * 100.0
            stage_data["progress"] = min(stage_progress, 100.0)

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
