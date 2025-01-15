import os
import sys
import time
import unittest
from random import randrange

from messagestore import MessageStore
from model import Base, Email
from settings import Settings

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
)  # noqa: F821


def message_factory():
    r = randrange(10000000)
    m = Email(
        body=f"Body {r}",
        folder="INBOX",
        uid=r,
        from_="them@email.tld",
        to="me@other.tld",
        subject="A subject line",
        message_id=f"<id{r} {int(round(time.time() * 1000))}>",
    )
    m.hash = m.mesg_hash()
    return m


class TestEmailService(unittest.TestCase):
    def setUp(self):
        self.store = MessageStore(
            settings=Settings(True), database_url="sqlite:///testdata.db"
        )
        self.engine = self.store.engine

    def tearDown(self):
        Base.metadata.drop_all(self.engine)

    def test_add_message(self):
        expected = message_factory()

        self.store.add_message(expected)
        all_emails = self.store.get_all_messages()
        self.assertEqual(all_emails[0].hash, expected.hash)

    # def test_insert_message(self):
    #     messages = list()
    #     for i in range(5):
    #         t = message_factory()
    #         t.folder = "Odd"
    #         if i % 2 == 0:
    #             t.folder = "Even"
    #         messages.append(t)
    #     message_to_insert = messages[3]

    #     self.store.add_messages(messages)
    #     message_to_insert = self.store.insert_message(message_to_insert)
    #     self.assertEqual(message_to_insert.id, None)

    def test_delete_message(self):
        message = message_factory()
        message = self.store.add_message(message)
        id_ = message.id
        self.store.delete_message(message)
        returned = self.store.get_message(id_)
        self.assertEqual(None, returned)

    def test_update_message(self):
        message = message_factory()
        message = self.store.add_message(message)
        ts = message.timestamp
        message.folder = "New"
        self.store.update_message(message)
        updated = self.store.get_message(message.id)
        self.assertEqual(updated.folder, "New")
        self.assertNotEqual(updated.timestamp, ts)

    def test_update_changes_ts(self):
        message = message_factory()
        message = self.store.add_message(message)
        ts = message.timestamp
        message.folder = "New"
        self.store.update_message(message)
        updated = self.store.get_message(message.id)
        self.assertGreater(updated.timestamp, ts)

    def test_by_folder(self):
        messages = list()
        for i in range(10):
            t = message_factory()
            t.folder = "Odd"
            if i % 2 == 0:
                t.folder = "Even"
            messages.append(t)
            self.store.add_message(t)

        self.assertEqual(len(self.store.get_messages_by_folder("Even")), 5)

    def test_message_by_uid(self):
        messages = list()
        last_uid = ""
        for i in range(10):
            t = message_factory()
            t.folder = "Odd"
            if i % 2 == 0:
                t.folder = "Even"
            last_uid = t.uid
            messages.append(t)
            self.store.add_message(t)

        self.assertNotEqual(
            len(self.store.get_messages_by_uids(uids=[last_uid], folder="Even")), 1
        )
        self.assertEqual(
            len(self.store.get_messages_by_uids(uids=[last_uid], folder="Odd")), 1
        )

    def test_empty_vector(self):
        message = message_factory()
        message = self.store.add_message(message)
        id_ = message.id
        new_mess = self.store.get_message(id_)
        self.assertFalse(message.emdeddings)
        self.assertFalse(new_mess.hasVector())

    def test_store_vector(self):
        message = message_factory()
        message.emdeddings = [3.33, 1.11]
        message = self.store.add_message(message)
        id_ = message.id
        new_mess = self.store.get_message(id_)
        self.assertListEqual(message.emdeddings, [3.33, 1.11])
        self.assertListEqual(new_mess.emdeddings, [3.33, 1.11])

    def test_bulk_add_messages(self):
        messages = list()
        for i in range(100):
            t = message_factory()
            t.folder = "Odd"
            if i % 2 == 0:
                t.folder = "Even"
            messages.append(t)

        self.store.add_messages(messages)
        to_find = messages[4]
        found = self.store.get_message_by_id(message_id=to_find.message_id)
        self.assertEqual(found.uid, to_find.uid)
