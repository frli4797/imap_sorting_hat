# imap_sorting_hat = "ish"

Magically sort email into smart folders. **ish** works by downloading plain text versions of all the emails in the source email folders and move those unread to the destination folders, by using a multi class classifier.

Initially the classifier needs to be trained on what your emails look like and where you like to keep them. This is done by downloading all emails (as plain text) from your destination folders, caching those locally. That cache, essentially a write through cache is used to aquire text embeddings from OpenAI for all the emails seen. Also the embeddings will be cached on disk. The embeddings will constitute the data to train a RandomForest on, and all the destination folders will be used as the classes for the classifier.

The model, after trained, will the get stored as well, and then used whenever a new email message has been discovered, assuming that **ish** is being run in non-interactive and polling mode. Once a new email (unseen/unread) message is discovered **ish** will classify that message and then move it according to its prior experience (training).

**ish** can also be run in interactive mode. It will then try to move **all** messages from the source folder(s), but ask the user about every message. This can be a good option when first training the model, and also to ensure that you don't end up with email in random folders in a cold start situation.

- No rule programming. Instead, just move a few emails into a smart folder and **ish** will quickly learn what the messages have in common.
- Any folder can be labeled a smart folder.
- Uses the lates OpenAI language model technology to quickly sort emails into corresponding folders.
- Compatible with all imap email clients.
- Works for all common languages.

This is copied (but not forked) from @kenseehart/[imap_sorting_hat](https://github.com/kenseehart/imap_sorting_hat) to support some additional changes and experimentation for my own learning.

## Configuring

To configure **ish** there needs to be a directory in which **ish** will put e-mail text and cached embeddings, used to train the model. It will also store the model as a pickle. Per default all these things will be contained within a directory, `${HOME}/.ish`

```text
.ish
├── data
│   ├── embd
│   ├── model.pkl
│   └── msgs
├── settings.yaml
```

### Example setting.yaml

```yaml
host: imap.mail.me.com
username: mymail@mydomain.tld
password: this-is-a-mock-password
source_folders:
- INBOX
destination_folders:
- News
- Notifications
- School
- Travel
ignore_folders:
- Archive
- Deleted Messages
- Drafts
- Sent Messages
- Junk
openai_api_key: ll-ddjg-RI51oFV-0Du9Xo4ERraVFd0UvcwFPP0wUkTB2tC
openai_model: text-embedding-3-small
```

## Running

### Command line

Run the **ish** by issuing
`python3 ish.py`
The main program has a few parameters that can be used.

```text
usage: ish.py [-h] [--learn-folders] [--interactive] [--dry-run] [--daemon] [--config-path CONFIG_PATH] [--verbose]
options:
  -h, --help            show this help message and exit
  --learn-folders, -l   Learn based on the contents of the destination folders
  --interactive, -i     Prompt user before moving anything
  --dry-run, -n         Don't actually move emails
  --daemon, -D          Run in daemon mode (NOT IMPLEMENTED)
  --config-path, -C CONFIG_PATH
                        Path for config file and data. Will default to /Users/fredriklilja/.ish
  --verbose, -v         Verbose/debug mode
```

### In Docker

You can also run **ish** in a Docker container.
`docker run -it  -v ./.ish:/opt/ish/config -e ISH_DAEMON=True -e ISH_DEBUG=True -e ISH_LEARN=True frli4797/ish`

## Take care

I leave no guarantees that this will work with your mailprovider, nor that it will work for well for your language. This is an application that I've been tinkering with as I got fed up with sifting through my email and wanted to learn something new. Email might be destroyed, misplaces or lost from using this little tool. Take care. Make backups.  

## Future development

- [x] Make it work (kind of).
- [x] Create command line parameters for the usual tasks and tweaks, such as traing, inference, dry-run
- [x] Optimize embedding calls to OpenAI by batching
- [x] Dockerize
- [ ] Use dev container
- [x] Service mode/daemon mode to be able to run this as a service, polling the imap server every so often
- [ ] Add Ollama as a potential source of embeddings

## Other

I made some experiments with other embedding models using Ollama. Unfortunately the precision really suffered in these experimiments, especially with a mailbox with messages on multiple languages.

Thanks to: [@kenseehart](https://github.com/kenseehart)
