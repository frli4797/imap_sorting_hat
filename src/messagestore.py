from os.path import join

from sqlalchemy import and_, create_engine, delete, insert, update
from sqlalchemy.orm import sessionmaker

from model import Base, Email
from settings import Settings


class MessageStore:
    def __init__(self, settings: Settings, database_url):
        self.__settings = settings
        if database_url is None:
            database_url = "sqlite:///" + join(
                self.__settings.data_directory, "message.db"
            )
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.Session().commit()

    def add_message(self, message: Email):
        """Add a message to the database."""
        session = self.Session()
        try:
            session.add(message)
            session.commit()
            session.refresh(message)
            return message
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_message(self, id_):
        session = self.Session()
        try:
            return session.query(Email).filter_by(id=id_).first()
        finally:
            session.close()

    def get_message_by_id(self, message_id):
        """Retrieve a message by its ID."""
        session = self.Session()
        try:
            return session.query(Email).filter_by(message_id=message_id).first()
        finally:
            session.close()

    def get_all_messages(self):
        """Retrieve all messages."""
        session = self.Session()
        try:
            return session.query(Email).all()
        finally:
            session.close()

    def get_messages_by_folder(self, folder) -> list[Email]:
        """Retrieve all messages sent by a specific folder."""
        session = self.Session()
        try:
            return session.query(Email).filter_by(folder=folder).all()
        finally:
            session.close()

    def get_messages_by_uids(self, folder, uids: list):
        session = self.Session()
        try:
            query = session.query(Email).filter(
                and_(Email.folder.is_(folder)), Email.uid.in_(uids)
            )
            return query.all()
        finally:
            session.close()

    def update_message(self, message: Email):
        """Update the content of a message by its ID."""
        session = self.Session()

        try:
            session.execute(
                update(Email),
                vars(message),
            )
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def add_messages(self, messages: list[Email]):
        session = self.Session()
        new_data = []

        for message in messages:
            new_data.append(vars(message))

        try:
            session.execute(
                insert(Email),
                new_data,
            )
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete_message(self, message):
        """Delete a message by its ID."""
        session = self.Session()
        try:
            session.execute(delete(Email).where(Email.id == message.id))
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
                session.query(Email)
                .filter(Email.timestamp > timestamp)
                .order_by(Email.timestamp)
                .all()
            )
