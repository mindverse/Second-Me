"""
Role related data structure
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import secrets
import string
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from lpm_kernel.common.repository.database_session import Base


def generate_role_uuid(prefix: str = "role") -> str:
    """
    generate role's uuid, combination of 6 letters and digits, not case sensitive

    Args:
        prefix: UUID's prefix, default is 'role'

    Returns:
        str: generated UUID, with the format of 'prefix_{6 letters and digits, not case sensitive}'
    """
    # only use lowercase letters and digits
    alphabet = string.ascii_lowercase + string.digits
    # generate a random string
    random_str = "".join(secrets.choice(alphabet) for _ in range(6))
    return f"{prefix}_{random_str}"


class Role(Base):
    """Role database structure"""

    __tablename__ = "roles"

    id = Column(Integer, primary_key=True)
    uuid = Column(
        String(64), nullable=False, unique=True, default=generate_role_uuid
    )  # add uuid field
    name = Column(String(100), nullable=False, unique=True)  # role name
    description = Column(String(500))  # role description
    system_prompt = Column(Text, nullable=False)  # role system prompt
    icon = Column(String(100))  # role icon
    is_active = Column(Boolean, default=True)
    enable_l0_retrieval = Column(Boolean, default=True)
    enable_l1_retrieval = Column(Boolean, default=True)
    create_time = Column(DateTime, nullable=False, default=datetime.now)
    update_time = Column(
        DateTime, nullable=False, default=datetime.now, onupdate=datetime.now
    )


@dataclass
class RoleDTO:
    """Role's data transfer object"""

    id: int
    uuid: str  # add uuid field
    name: str
    description: str
    system_prompt: str
    icon: Optional[str] = None
    is_active: bool = True
    enable_l0_retrieval: bool = True
    enable_l1_retrieval: bool = True
    create_time: datetime = None
    update_time: datetime = None

    @classmethod
    def from_model(cls, model: Role) -> "RoleDTO":
        """Create DTO from database model"""
        return cls(
            id=model.id,
            uuid=model.uuid,  # add uuid field
            name=model.name,
            description=model.description,
            system_prompt=model.system_prompt,
            icon=model.icon,
            is_active=model.is_active,
            enable_l0_retrieval=model.enable_l0_retrieval,
            enable_l1_retrieval=model.enable_l1_retrieval,
            create_time=model.create_time,
            update_time=model.update_time,
        )

    def to_dict(self) -> Dict[str, Any]:
        """transfer to dict"""
        return {
            "id": self.id,
            "uuid": self.uuid,  # add uuid field
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "icon": self.icon,
            "is_active": self.is_active,
            "enable_l0_retrieval": self.enable_l0_retrieval,
            "enable_l1_retrieval": self.enable_l1_retrieval,
            "create_time": self.create_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.create_time
            else None,
            "update_time": self.update_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.update_time
            else None,
        }


@dataclass
class CreateRoleRequest:
    """create Role request object"""

    name: str
    description: str
    system_prompt: str
    icon: Optional[str] = None
    enable_l0_retrieval: Optional[bool] = True
    enable_l1_retrieval: Optional[bool] = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreateRoleRequest":
        return cls(
            name=data["name"],
            description=data["description"],
            system_prompt=data["system_prompt"],
            icon=data.get("icon"),
            enable_l0_retrieval=data.get("enable_l0_retrieval", True),
            enable_l1_retrieval=data.get("enable_l1_retrieval", True),
        )


@dataclass
class UpdateRoleRequest:
    """update Role request object"""

    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    icon: Optional[str] = None
    is_active: Optional[bool] = None
    enable_l0_retrieval: Optional[bool] = None
    enable_l1_retrieval: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpdateRoleRequest":
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            system_prompt=data.get("system_prompt"),
            icon=data.get("icon"),
            is_active=data.get("is_active"),
            enable_l0_retrieval=data.get("enable_l0_retrieval"),
            enable_l1_retrieval=data.get("enable_l1_retrieval"),
        )

@dataclass
class ShareRoleRequest:
    """share Role request object"""
    role_id: str
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShareRoleRequest":
        return cls(
            role_id=data["role_id"]
        )
