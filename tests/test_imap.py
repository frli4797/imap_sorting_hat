import string
from unittest import mock

import pytest
from email.message import EmailMessage

from ish import imap as imap_mod
from ish.imap import (
    html2text,
    get_header,
    mesg_to_text,
    ImapHandler,
    HEADER_KEY,
    BODY_KEY,
)
from ish.message import Message


def test_html2text_returns_printable_str_and_no_tags():
    html = "<html><body><p>Hello&nbsp;there &copy; <b>bold</b></p></body></html>"
    out = html2text(html)
    assert isinstance(out, str)
    # No raw html angle brackets should remain
    assert "<" not in out and ">" not in out
    # All characters should be within string.printable (function filters)
    assert all(ch in string.printable for ch in out)


def test_get_header_simple_cases():
    raw = b"Subject: Hello\r\nFrom: John <john@example.com>\r\nTo: alice@example.com\r\n"
    assert get_header(raw, "SUBJECT") == "Hello"
    assert "John" in get_header(raw, "FROM")
    assert "alice@example.com" in get_header(raw, "TO")


def make_multipart_message():
    m = EmailMessage()
    m.set_content("This is plain text\n> quoted\nNew line")
    m.add_alternative("<html><body><p>HTML part text</p></body></html>", subtype="html")
    return m


def test_mesg_to_text_combines_plain_and_html_and_removes_arrows():
    m = make_multipart_message()
    out = mesg_to_text(m)
    assert "This is plain text" in out
    assert "HTML part text" in out
    # Quoted '>' should be removed by regex
    assert ">" not in out


def test_parse_mesg_present_and_missing_keys():
    m = make_multipart_message()
    header_bytes = b"Subject: MyTest\r\nFrom: Foo Bar <foo@example.com>\r\nTo: x@y\r\n"
    body_bytes = m.as_bytes()
    p = {HEADER_KEY: header_bytes, BODY_KEY: body_bytes}
    parsed = imap_mod.ImapHandler(Settings:=None).parse_mesg(p)  # type: ignore
    assert isinstance(parsed, Message)
    assert hasattr(parsed, "body") 
    # When keys missing we should still get dict with empty-ish strings
    parsed2 = imap_mod.ImapHandler(Settings:=None).parse_mesg({})
    assert isinstance(parsed2, Message)
    
    # Assert that from_addr, toaddr and body are present in parsed2 object
    assert hasattr(parsed2, "from_addr")
    assert hasattr(parsed2, "to_addr")
    assert hasattr(parsed2, "body") 



def make_handler_with_mock_conn():
    # Build a handler and attach a mock imap client
    handler = ImapHandler(settings=mock.MagicMock(), readonly=False)
    conn = mock.MagicMock()
    handler._ImapHandler__imap_conn = conn
    return handler, conn


def test_move_non_iterable_raises_value_error():
    handler, _ = make_handler_with_mock_conn()
    with pytest.raises(ValueError):
        handler.move("INBOX", "not-an-iter", "Dest")


def test_move_with_empty_uids_returns_zero():
    handler, conn = make_handler_with_mock_conn()
    assert handler.move("INBOX", [], "Dest") == 0
    conn.select_folder.assert_not_called()


def test_move_uses_move_when_supported():
    handler, conn = make_handler_with_mock_conn()
    handler._ImapHandler__capabilities = ["MOVE"]
    # Setup expectations
    uids = [101, 102]
    result = handler.move("INBOX", uids, "SmartDest", flag_messages=True, flag_unseen=True)
    # Should have selected folder and flagged/unflagged
    conn.select_folder.assert_called_once_with("INBOX", handler._ImapHandler__readonly)
    conn.add_flags.assert_any_call(uids, [imap_mod.imapclient.FLAGGED])
    conn.remove_flags.assert_any_call(uids, [imap_mod.imapclient.SEEN])
    conn.move.assert_called_once_with(uids, "SmartDest")
    assert result == len(uids)


def test_move_falls_back_to_copy_when_move_not_supported():
    handler, conn = make_handler_with_mock_conn()
    handler._ImapHandler__capabilities = []  # has_move -> False
    uids = [201]
    res = handler.move("INBOX", uids, "OtherDest", flag_messages=False, flag_unseen=False)
    conn.copy.assert_called_once_with(uids, "OtherDest")
    conn.add_flags.assert_called_with(uids, [imap_mod.imapclient.DELETED], silent=True)
    conn.uid_expunge.assert_called_once_with(uids)
    assert res == len(uids)


def test_list_folders_reconnects_on_imap_client_error():
    handler, conn = make_handler_with_mock_conn()
    # First call raises error, second succeeds
    conn.list_folders.side_effect = [
        imap_mod.IMAPClientError("Connection lost"),
        [(b'\\Noselect', b'/', b'INBOX'), (b'\\All', b'/', b'Archive')],
    ]
    result = handler.list_folders()
    assert conn.list_folders.call_count == 2


def test_search_reconnects_on_imap_client_error():
    handler, conn = make_handler_with_mock_conn()
    conn.search.side_effect = [
        imap_mod.IMAPClientError("Connection timeout"),
        [1, 2, 3],
    ]
    result = handler.search("INBOX")
    assert result == [1, 2, 3]
    assert conn.select_folder.call_count == 2


def test_fetch_reconnects_on_imap_client_error():
    handler, conn = make_handler_with_mock_conn()
    conn.fetch.side_effect = [
        imap_mod.IMAPClientError("Connection reset"),
        {1: {b"BODY[HEADER.FIELDS (SUBJECT FROM TO CC BCC)]": b"Subject: Test\r\n"}},
    ]
    result = handler.fetch([1])
    assert 1 in result
    assert conn.fetch.call_count == 2


def test_move_retries_on_io_error():
    handler, conn = make_handler_with_mock_conn()
    handler._ImapHandler__capabilities = ["MOVE"]
    conn.select_folder.side_effect = [IOError("Network error"), None]
    conn.move.return_value = None
    result = handler.move("INBOX", [1, 2], "Archive")
    assert result == 2


def test_connect_imap_returns_false_when_connection_fails():
    handler = ImapHandler(settings=mock.MagicMock(), readonly=False)
    handler._ImapHandler__settings.imap_host = "invalid.host"
    handler._ImapHandler__settings.username = "user"
    handler._ImapHandler__settings.password = "pass"
    with mock.patch("imapclient.IMAPClient", side_effect=IOError("Host unreachable")):
        result = handler.connect_imap()
    assert result is False


def test_close_handles_connection_already_closed():
    handler = ImapHandler(settings=mock.MagicMock(), readonly=False)
    conn = mock.MagicMock()
    conn.logout.side_effect = imap_mod.IMAP4.error("Connection already closed")
    handler._ImapHandler__imap_conn = conn
    handler.close()
    assert handler._ImapHandler__imap_conn is None