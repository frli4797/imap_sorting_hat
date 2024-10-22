# imap_sorting_hat = "ish"

Magically sort email into smart folders. This is copied (but not forked) from @kenseehart/[imap_sorting_hat](https://github.com/kenseehart/imap_sorting_hat) to support some additional changes and experimentation for my own learning.

- No rule programming. Instead, just move a few emails into a smart folder and **ish** will quickly learn what the messages have in common.
- Any folder can be labeled a smart folder.
- Uses the lates OpenAI language model technology to quickly sort emails into corresponding folders.
- Compatible with all imap email clients.
- Works for all common languages.

## Future development

- [x] Make it work (kind of).
- [x] Create command line parameters for the usual tasks and tweaks, such as traing, inference, dry-run
- [x] Optimize embedding calls to OpenAI by batching
- [ ] Dockerize
- [ ] Use dev container
- [ ] Daemonize to be able to run this as a service
- [ ] Add Ollama as a potential source of embeddings

Thanks to: [@kenseehart](https://github.com/kenseehart)
