import shelve
import sys
from getpass import getpass
from hashlib import sha256
from os.path import exists, isdir, join
from time import perf_counter
from typing import Dict, List
from ish import Settings

settings = Settings()

msg_file =                  join(settings['data_directory'], 'msgs')
embd_file =                 join(settings['data_directory'], 'embd')
migrated_embeddings_file =  join(settings['data_directory'], 'embd_raw')

with shelve.open(embd_file, writeback=True) as fe, shelve.open(migrated_embeddings_file, writeback=True) as f_migrated: 
    keys = fe.keys()
    for key in keys:
        embedding = fe[key]
        print(f"Key {key} Embedding {embedding.__class__.__name__}")
        raw_embd = embedding.data[0].embedding
        print(f"Raw embedding {len(raw_embd)}")
        f_migrated[key] = raw_embd
    f_migrated.sync()
