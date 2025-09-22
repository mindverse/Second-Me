from typing import Optional, List, Type, TypeVar
import aiomysql
import logging

from .base_repository import BaseRepository

T = TypeVar("T")

logger = logging.getLogger(__name__)


class MySQLRepository(BaseRepository[T]):
    def __init__(self, entity_class: Type[T]):
        super().__init__()
        self.entity_class = entity_class
        # Get table name defined by the entity class
        self.table_name = getattr(entity_class, "__tablename__", None)
        if not self.table_name:
            raise ValueError(
                f"Entity class {entity_class.__name__} must define __tablename__"
            )

    async def get_by_id(self, id: int) -> Optional[T]:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    await cursor.execute(
                        f"SELECT * FROM {self.table_name} WHERE id = %s", (id,)
                    )
                    result = await cursor.fetchone()
                    return self.entity_class.from_dict(result) if result else None
        except Exception as e:
            logger.error(f"Database error in get_by_id: {str(e)}")
            raise  # Directly throw the original exception, let the unified error handling handle it

    async def create(self, entity: T) -> T:
        with self.db.session() as session:
            session.add(entity)
            session.flush()  # Ensure ID generation
            session.refresh(entity)  # Refresh the object
            # Convert to dictionary in session context
            result = entity.to_dict()
            return self.model.from_dict(result)  # Create a new object instance

    async def update(self, entity: T) -> T:
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                data = entity.to_dict()
                # Remove id from update data
                entity_id = data.pop("id")
                # Don't update create_time
                data.pop("create_time", None)

                set_clause = ", ".join([f"{k} = %s" for k in data.keys()])
                values = list(data.values()) + [entity_id]

                query = f"UPDATE {self.table_name} SET {set_clause} WHERE id = %s"
                await cursor.execute(query, values)
                await conn.commit()
                return entity

    async def delete(self, id: int) -> bool:
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE id = %s", (id,)
                )
                await conn.commit()
                return cursor.rowcount > 0

    async def list(
        self, filters: dict = None, limit: int = 1000, offset: int = 0
    ) -> List[T]:
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                query = f"SELECT * FROM {self.table_name}"
                values = []

                if filters:
                    where_conditions = []
                    for key, value in filters.items():
                        where_conditions.append(f"{key} = %s")
                        values.append(value)
                    if where_conditions:
                        query += " WHERE " + " AND ".join(where_conditions)

                query += " LIMIT %s OFFSET %s"
                values.extend([limit, offset])

                await cursor.execute(query, values)
                results = await cursor.fetchall()
                return [self.entity_class.from_dict(row) for row in results]
