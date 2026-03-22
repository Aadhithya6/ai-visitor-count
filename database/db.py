import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, LargeBinary, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import enum

Base = declarative_base()

class EventType(enum.Enum):
    ENTRY = "entry"
    EXIT = "exit"

class Face(Base):
    __tablename__ = 'faces'
    
    face_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    events = relationship("Event", back_populates="face", cascade="all, delete-orphan")
    embeddings = relationship("FaceEmbedding", back_populates="face", cascade="all, delete-orphan")

class FaceEmbedding(Base):
    __tablename__ = 'face_embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(String, ForeignKey('faces.face_id'), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Stored as numpy blob
    created_at = Column(DateTime, default=datetime.utcnow)
    
    face = relationship("Face", back_populates="embeddings")

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(String, ForeignKey('faces.face_id'), nullable=False)
    event_type = Column(Enum(EventType), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String, nullable=False)
    
    face = relationship("Face", back_populates="events")

class DatabaseManager:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        # Drop all tables to apply the new schema without migrations
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        return self.Session()
