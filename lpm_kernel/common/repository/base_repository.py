from typing import Generic, TypeVar, Optional, List, Type
from sqlalchemy import select
from lpm_kernel.common.repository.database_session import DatabaseSession, Base

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    def __init__(self, model: Type[T]):
        self.model = model
        self._db = DatabaseSession()

    def get_by_id(self, id: int) -> Optional[T]:
        with self._db.session() as session:
            return session.get(self.model, id)

    def create(self, entity: T) -> T:
        with self._db.session() as session:
            try:
                session.add(entity)
                session.commit()
                session.refresh(entity)
                return self.model.from_dict(entity.to_dict())
            except Exception as e:
                session.rollback()
                raise

    def update(self, entity: T) -> Optional[T]:
        with self._db.session() as session:
            try:
                updated = session.merge(entity)
                session.commit()
                return updated
            except Exception as e:
                session.rollback()
                raise

    def delete(self, id: int) -> bool:
        with self._db.session() as session:
            try:
                entity = session.get(self.model, id)
                if entity:
                    session.delete(entity)
                    session.commit()
                    return True
                return False
            except Exception as e:
                session.rollback()
                raise

    def list(self, filters: dict = None) -> List[T]:
        """Return all records that match the given filters without pagination"""
        with self._db.session() as session:
            try:
                query = select(self.model)
                if filters:
                    query = query.filter_by(**filters)
                results = session.scalars(query).all()
                return [self.model.from_dict(item.to_dict()) for item in results]
            except Exception as e:
                session.rollback()
                raise

    def page(self, limit: int = 100, offset: int = 0, filters: dict = None) -> List[T]:
        """Return paginated records that match the given filters"""
        with self._db.session() as session:
            try:
                query = select(self.model)
                if filters:
                    query = query.filter_by(**filters)
                query = query.limit(limit).offset(offset)
                results = session.scalars(query).all()
                return [self.model.from_dict(item.to_dict()) for item in results]
            except Exception as e:
                session.rollback()
                raise
