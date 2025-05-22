from sqlalchemy.orm import DeclarativeBase # Updated for SQLAlchemy 2.0
from app.db.session import engine

class Base(DeclarativeBase):
    pass

def init_db():
    # Import all SQLAlchemy models here before calling Base.metadata.create_all
    # This ensures they are registered with SQLAlchemy's metadata.
    from app.schemas.document import DocumentDB # Import your SQLAlchemy models
    # from app.schemas.another_model import AnotherModelDB # If you add more

    Base.metadata.create_all(bind=engine)