import json

from model.message import Message
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

engine = create_engine("sqlite:///your-database.db")
Base.metadata.create_all(engine)
engine.connect()
jsonvec = json.dumps([22.2, 11.1])
test_mess = Message(
    from_="from@from.se", to="to@to.se", subject="A subject", vector=[22.2, 11.1]
)


class MessageStore:
    def __init__(self, database_url="sqlite:///messages.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_message(self, message: Message):
        """Add a message to the database."""
        session = self.Session()
        try:
            session.add(message)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_message_by_id(self, message_id):
        """Retrieve a message by its ID."""
        session = self.Session()
        try:
            return session.query(Message).filter_by(id=message_id).first()
        finally:
            session.close()

    def get_all_messages(self):
        """Retrieve all messages."""
        session = self.Session()
        try:
            return session.query(Message).all()
        finally:
            session.close()

    def get_messages_by_folder(self, folder):
        """Retrieve all messages sent by a specific folder."""
        session = self.Session()
        try:
            return session.query(Message).filter_by(folder=folder).all()
        finally:
            session.close()

    def update_message(self, message_id, new_content):
        """Update the content of a message by its ID."""
        session = self.Session()
        try:
            message = session.query(Message).filter_by(id=message_id).first()
            if message:
                message.content = new_content
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete_message(self, message_id):
        """Delete a message by its ID."""
        session = self.Session()
        try:
            message = session.query(Message).filter_by(id=message_id).first()
            if message:
                session.delete(message)
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_messages_after(self, timestamp):
        """Retrieve all messages sent after a specific timestamp."""
        with self._session_scope() as session:
            return (
                session.query(Message)
                .filter(Message.timestamp > timestamp)
                .order_by(Message.timestamp)
                .all()
            )


# stmt = select(Message)
# with Session(engine) as session:
#     result = session.execute(select(Message).order_by(Message.id))
#     # print(result.all())
#     # db.commit()

#     for mess in result:
#         # vec = mess.vector
#         # print(mess.id, vec)
#         print(mess.Message.id, mess.Message.vector)
