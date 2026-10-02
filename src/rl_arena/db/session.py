"""Асинхронный движок SQLAlchemy и фабрика сессий."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(database_url: str) -> AsyncEngine:
    """Создать движок; соединения открываются лениво, старт не падает без БД."""
    return create_async_engine(database_url, pool_pre_ping=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Создать фабрику сессий; `expire_on_commit=False` — объекты живы после commit."""
    return async_sessionmaker(engine, expire_on_commit=False)
