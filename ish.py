# ish.py

'''
# imap_sorting_hat = "ish"
Magically sort email into smart folders

- No rule programming. Instead, just move a few emails into a smart folder and **ish** will quickly learn what the messages have in common.
- Any folder can be labeled a smart folder.
- Uses the lates OpenAI language model technology to quickly sort emails into corresponding folders.
- Compatible with all imap email clients.
- Works for all common languages.

Status: Early development
'''

import email
import imaplib
import logging
import os
import re
import shelve
import sys
from getpass import getpass
from hashlib import sha256
from os.path import exists, isdir, join
from time import perf_counter
from typing import Dict, List

import backoff
import bs4
import imapclient
import numpy as np
import yaml
from openai import OpenAI, RateLimitError
from openai.types import CreateEmbeddingResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(level=logging.INFO)

base_logger = logging.getLogger("ish")
base_logger.setLevel(level=logging.DEBUG)

re_header_item = re.compile(r'(\w+): (.*)')
re_address = re.compile(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
re_newline = re.compile(r'[\r\n]+')
re_symbol_sequence = re.compile(r'(?<=\s)\W+(?=\s)')
re_whitespace = re.compile(r'\s+')

def html2text(html: str) -> str:
    '''Convert html to plain-text using beautifulsoup'''
    soup = bs4.BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator=' ')
    return text

def mesg_to_text(mesg: email.message.Message) -> str:
    '''Convert an email message to plain-text'''
    text = ''
    for part in mesg.walk():
        if part.get_content_type() == 'text/plain':
            text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        elif part.get_content_type() == 'text/html':
            text += html2text(part.get_payload(decode=True).decode('utf-8', errors='ignore'))

    text = re_symbol_sequence.sub('', text)
    text = re_whitespace.sub(' ', text)
    return text

class Settings(dict):
    '''Settings for the application'''

    @staticmethod
    def get_user_directory():
        if os.name == 'nt':
            dir = os.path.expandvars('%USERPROFILE%')
        elif os.name == 'posix':
            dir = os.path.expandvars('$HOME')
        elif os.name == 'mac':
            dir = os.path.expandvars('$HOME')
        return dir

    @property
    def settings_file(self):
        ishd = os.path.join(self.get_user_directory(), '.ish')
        os.makedirs(ishd, exist_ok=True)
        ishfile = os.path.join(ishd, 'settings.yaml')
        return ishfile

    def read(self):
        with open(self.settings_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.update(data)

    def write(self):
        with open(self.settings_file, 'w') as f:
            yaml.dump(dict(self), f)
            self.logger.info('Settings saved to %s', self.settings_file)

    def __init__(self):
        self.logger = base_logger.getChild(self.__class__.__name__)

        self['source_folders'] = []
        self['destination_folders'] = []
        self['ignore_folders'] = []
        self['host'] = ''
        self['username'] = ''
        self['password'] = ''
        self['data_directory'] = ''
        self['openai_api_key'] = ''
        self['openai_model'] = 'text-embedding-ada-002'

        if exists(self.settings_file):
            self.read()

    def update_data_settings(self):
        while not isdir(self['data_directory']):
            if self['data_directory'] == '':
                self['data_directory'] = join(self.get_user_directory(), '.ish', 'data')
            else:
                print(f'Invalid data directory: {self["data_directory"]}')

            self['data_directory'] = input(f'Enter data directory[{self["data_directory"]}]: ').strip() or self['data_directory']
            try:
                os.makedirs(self['data_directory'], exist_ok=True)
            except IOError:
                pass

        self.write()

    def update_openai_settings(self):
        self['openai_api_key'] = input('Enter OpenAI API key (see https://beta.openai.com)): ')
        self.write()

    def update_login_settings(self):
        self.update({
            'host': input(f'Enter imap server [{self["host"]}]: ').strip() or self['host'],
            'username': input(f'Enter username [{self["username"]}]: ').strip() or self['username'],
            'password': getpass('Enter password: ')
        })
        self.write()

    def update_folder_settings(self, folders: List[str]):
        source_folders = set(self['source_folders'])
        destination_folders = set(self['destination_folders'])
        ignore_folders = set(self['ignore_folders'])
        all_folders = source_folders | destination_folders | ignore_folders

        for folder in folders:
            if folder not in all_folders:
                opt = None
                while opt not in ['s', 'd', 'i']:
                    opt = input(f'Folder {folder} is not configured. What do you want to do with it? [s]ource, [d]estination, [i]gnore: ')

                if opt == 's':
                    source_folders.add(folder)
                elif opt == 'd':
                    destination_folders.add(folder)
                elif opt == 'i':
                    ignore_folders.add(folder)

        missing_folders = all_folders - set(folders)
        if missing_folders:
            for folder in missing_folders:
                self.logger.info(f'Folder {folder} is missing. It will be removed from the settings.')

            source_folders -= missing_folders
            destination_folders -= missing_folders
            ignore_folders -= missing_folders

        self['source_folders'] = sorted(source_folders)
        self['destination_folders'] = sorted(destination_folders)
        self['ignore_folders'] = sorted(ignore_folders)

        self.write()


class ISH:
    def __init__(self) -> None:
        self.logger = base_logger.getChild(self.__class__.__name__)
        self.settings = Settings()
        self.client:OpenAI = None
        self.imap_conn = None
        self.hkey = b'BODY[HEADER.FIELDS (SUBJECT FROM TO CC BCC)]'
        self.bkey = b'BODY[]'
        self.max_chars = 16384
        self.classifier: RandomForestClassifier = None

    @property
    def msgs_file(self) -> str:
        return join(self.settings['data_directory'], 'msgs')

    @property
    def embd_file(self) -> str:
        return join(self.settings['data_directory'], 'embd')
    
    @property
    def model_file(self) -> str:
        return join(self.settings['data_directory'], 'model.pkl')

    def mesg_hash(self, mesg: str) -> str:
        return sha256(mesg['body'].encode('utf-8')).hexdigest()[:12]

    def connect_imap(self) -> bool:
        if not self.settings['host'] or not self.settings['username']:
            return False

        try:
            imap_conn = imapclient.IMAPClient(self.settings['host'], ssl=True)
            imap_conn.login(self.settings['username'], self.settings['password'])
        except imaplib.IMAP4.error as e:
            self.logger.error(e)
            return False
        self.logger.debug(imap_conn.capabilities())
        self.imap_conn = imap_conn
        return True

    def connect_openai(self) -> bool:
        if not self.settings['openai_api_key']:
            return False

        # check if api key is valid
        try:
            self.client = OpenAI(api_key=self.settings['openai_api_key'])
        except OpenAI.APIError as e:
            self.logger.error(e)
            return False
        return True

    def configure_and_connect(self):
        '''Configure ish and connect to imap and openai'''
        settings = self.settings

        self.settings.update_data_settings()

        while not self.connect_imap():
            settings.update_login_settings()

        while not self.connect_openai():
            settings.update_openai_settings()

        folders = [t[2] for t in self.imap_conn.list_folders()]
        settings.update_folder_settings(folders)

        print('Configuration complete')

    def connect_noninteractive(self) -> bool:
        '''Connect to imap and openai without user interaction'''
        if not self.connect_imap():
            self.logger.error(f'Failed to connect to imap server. Configure, or check your settings in {self.settings.settings_file}')
            return False

        if not self.connect_openai():
            self.logger.error(f'Failed to connect to openai. Configure or check your settings in {self.settings.settings_file}')
            return False

        return True

    def parse_mesg(self, mesg: dict) -> dict:
        '''Parse a raw message into a string'''
        header = mesg[self.hkey].decode('utf-8')
        raw_body = mesg[self.bkey]
        payload = email.message_from_bytes(raw_body)
        body_text = mesg_to_text(payload)
        header_lines = re_newline.split(header)

        header_dict = {}
        for item in header_lines:
            m = re_header_item.match(item)
            if m:
                header_dict[m.group(1)] = m.group(2)

        # remove spam prefix because we want spam training data to be as similar as possible to non-spam training data
        header_dict['Subject'] = header_dict.get('Subject', '').removeprefix('**SPAM**').strip()
        
        from_addr = []
        to_addr = []

        try:
            from_addr = [m.group(1) for m in re_address.finditer(header_dict['FROM'])]
            to_addr = [m.group(1) for m in re_address.finditer(header_dict['TO'] + header_dict.get('CC', ''))]
        except:
            self.logger.warning("Did not find a TO or CC in the headers.")

        mesg_dict = {
            'from': from_addr ,
            'tocc': to_addr,
            'body': f'Subject: {header_dict["Subject"]}. {body_text}',
        }

        return mesg_dict
    @staticmethod
    def backoff_debug(details):
        base_logger.info ("Backing off {wait:0.1f} seconds after {tries} tries ")

    @backoff.on_exception(backoff.expo, RateLimitError, on_backoff=backoff_debug)
    def get_embedding(self, text):
        e:CreateEmbeddingResponse = self.client.embeddings.create(input = [text], model=self.settings['openai_model'])
        return e

    def get_msgs(self, folder:str, uids:List[int]) -> Dict[int, str]:
        '''Fetch new messages through cache {uid: 'msg'}'''
        d = {}
        new_uids = []
        self.logger.info(f"Getting {len(uids)} messages from {folder}")
        with shelve.open(self.msgs_file, writeback=False) as fm:
            for uid in uids:
                try:
                    hash = fm[f'{folder}:{uid}']
                    mesg = fm[f'{hash}.mesg']
                    d[uid] = mesg
                    continue
                except KeyError:
                    new_uids.append(uid)

            if len(new_uids) > 0:
                self.logger.info(f"Found {len(new_uids)} not in cache")

                msgs = self.imap_conn.fetch(new_uids, [self.hkey, self.bkey])
                for uid in new_uids:
                    mesg = self.parse_mesg(msgs[uid])
                    hash = self.mesg_hash(mesg)
                    try:
                        fm[f'{folder}:{uid}'] = hash
                        fm[f'{hash}.mesg'] = mesg
                        d[uid] = mesg
                    except Exception as e:
                        self.logger.error(f"Error trying to add to shelf. Folder {folder} Uid {uid} Has {hash}", exc_info=e)
                self.logger.info(f"Found {len(new_uids)} to cache")
        
        self.logger.info(f"Total messages found/added {len(d)} in {folder}.")
        return d

    def get_embeddings(self, folder:str, uids:List[int]) -> Dict[int, np.ndarray]:
        '''Get embeddings using OpenAI API through cache {uid: embedding}'''
        dhash = {}
        dembd = {}
        self.logger.info(f"Getting {len(uids)} embeddings from {folder}")
        # with embd and msgs db open at the same time
        with shelve.open(self.msgs_file, writeback=False) as fm:
            new_uids = [] # uids that need a new hash
            self.logger.debug(f"Finding hashes for {len(uids)}")
            for uid in uids:
                try:
                    dhash[uid] = fm[f'{folder}:{uid}']
                except KeyError:
                    new_uids.append(uid)
                    continue
            
            self.logger.info(f"Found {len(new_uids)} out of {len(uids)} needing hash")
            dmesg = self.get_msgs(folder, new_uids)

            self.logger.debug(f"Adding hashes for {len(dmesg)} messages")
            for uid, mesg in dmesg.items():
                hash = self.mesg_hash(mesg)
                dhash[uid] = hash
                fm[f'{folder}:{uid}'] = hash
            self.logger.debug("Added hashes for messages.")
            fm.close()
        
            new_uids = [] # uids that need a new embedding
            self.logger.debug(f"Finding embedding for {len(uids)}")
        with shelve.open(self.embd_file, writeback=False) as fe: 
            for uid, hash in dhash.items(): # dhash is {uid: hash}
                try:
                    embd = fe[f'{hash}.embd']
                    dembd[uid] = embd
                    continue
                except KeyError:
                    new_uids.append(uid)
            self.logger.debug(f"Found {len(new_uids)} uids needing embeddings")

            if len(new_uids) > 0:
                msgs = self.get_msgs(folder, new_uids)


                messages = list(msg['body'][:self.max_chars] for msg in msgs.values() )
                e = self.client.embeddings.create(input = messages, model=self.settings['openai_model'])
                """ 
                if len(e.data) == len( msgs.keys()):
                    self.logger.info("EQUAL")
                else:
                    self.logger.info("NOT equal") """

                self.logger.info(f"Adding embeddings for {len(new_uids)} messages")
                for uid, msg in msgs.items():
                    dembd[uid] = self.get_embedding(msg['body'][:self.max_chars])
                    fe[f'{dhash[uid]}.embd'] = dembd[uid]

        self.logger.info(f"Total embeddings found/added {len(dembd)} in {folder}.")
        return dembd

    def learn_folders(self, folders:List[str]) -> RandomForestClassifier:
        imap_conn = self.imap_conn
        embed_array = []
        folder_array = []

        t0 = perf_counter()

        for folder in folders:
            imap_conn.select_folder(folder)
            self.logger.info(f"Learning folder {folder}")
            # Retrieve the UIDs of all messages in the folder
            uids = imap_conn.search(['ALL'])
            embd = self.get_embeddings(folder, uids[:80])
            embed_array.extend(embd.values())
            folder_array.extend([folder] * len(embd))

        t1 = perf_counter()
        self.logger.info(f'Fetched {len(embed_array)} embeddings in {t1-t0:.2f} seconds')

        # Train a classifier
        self.logger.info("Training classifier...")

        all_embeddings:List = []
        for embd in embed_array:
            embedding = embd.data[0].embedding
            all_embeddings.append(embedding)

        X = np.array(all_embeddings)
        y = np.array([folder for folder in folder_array])

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        clf = RandomForestClassifier(n_estimators=100, random_state=0)

        self.classifier = clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        self.logger.info(f'Accuracy: {accuracy:.2f}')
        self.logger.info(f'Classifier: {self.classifier}')

        t2 = perf_counter()

        self.logger.info(f'Trained classifier in {t2-t1:.2f} seconds')
        
        joblib.dump(self.classifier, self.model_file) 
        self.logger.info(f"Saved classifier.")

    def classify_messages(self, source_folders:List[str], inference_mode=False) -> None:
        imap_conn = self.imap_conn

        if inference_mode:
            classifier = joblib.load(self.model_file)
        else:
            classifier = self.classifier
        
        skipped = 0
        moved = 0
        folder_results = {}
        for folder in source_folders:
            imap_conn.select_folder(folder)
            uid = []
            self.logger.info("Classifying messages for folder {folder}")
            # Retrieve the UIDs of all messages in the folder
            if inference_mode:
                uids = imap_conn.search(criteria=[b'UNSEEN'])
            else:
                uids = imap_conn.search(['ALL'])

            embd = self.get_embeddings(folder, uids[:160])
            mesgs = self.get_msgs(folder, uids[:160])

            for uid, embd in embd.items():
                dest_folder = classifier.predict([embd.data[0].embedding])[0]
                proba = classifier.predict_proba([embd.data[0].embedding])[0]
                ranks = sorted(zip(proba, classifier.classes_), reverse=True)

                (top_probability, top_class) = ranks[0]
                if top_probability > 0.25:
                    print(f'\n{uid:3} From {mesgs[uid]["from"][0]}: {mesgs[uid]["body"][0:100]}')
                    for p, c in ranks:
                        print(f'{p:.2f}: {c}')
                    self.move_message(skipped, moved, folder, uid, dest_folder, inference_mode)
                else:
                    skipped += 1

        self.logger.info(f"Finished moved {moved} and skipped {skipped}")

    def move_message(self, skipped, moved, folder, uid, dest_folder, inference_mode):
        opt = None

        if inference_mode:
            self.imap_conn.select_folder(folder)
            self.imap_conn.remove_flags([uid], [imapclient.SEEN])
            self.imap_move_message(folder, uid, dest_folder)
            moved += 1
        else:
            while opt not in ['y', 'n', 'q']:
                opt = input(f'Move message to {dest_folder}? [y]yes, [n]no, [q]quit:')
                if opt == 'y':
                    self.imap_move_message(folder, uid, dest_folder)
                    moved += 1
                elif opt == 'q':
                    self.logger.info(f"Quitting. Finished moved {moved} and skipped {skipped}")
                    exit(0)
                elif opt == 'n':
                    skipped += 1

    def imap_move_message(self, folder, uid, dest_folder, flag_messages=True):
        self.imap_conn.select_folder(folder)
        if flag_messages:
            self.imap_conn.add_flags([uid], [imapclient.FLAGGED])
            # imap_conn.remove_flags([uid], [imapclient.SEEN])
        self.imap_conn.copy([uid], dest_folder)
        self.imap_conn.add_flags([uid],imapclient.DELETED, silent=True)
        self.imap_conn.uid_expunge([uid])
        self.logger.info(f'REALLY moved from {folder} to {dest_folder}: {uid}')


    def run(self, interactive:bool) -> int:
        try:
            if interactive:
                self.configure_and_connect()
            else:
                if not self.connect_noninteractive():
                    return 1

            settings = self.settings

            for f in settings['source_folders']:
                self.logger.info(f'Source folder: {f}')

            for f in settings['destination_folders']:
                self.logger.info(f'Destination folder: {f}')

            if interactive:
                self.learn_folders(settings['destination_folders'])

            self.classify_messages(settings['source_folders'], inference_mode=(not interactive))
        except Exception as e:
            base_logger.error("Something went wrong. Unknown error.", e)
        finally:
            base_logger.debug("Cleaning up imap connection.")
            self.imap_conn.logout()
        return 0

def main():
    ish = ISH()
    r = ish.run(interactive=False)
    sys.exit(r)

if __name__ == '__main__':
    main()

