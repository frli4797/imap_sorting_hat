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
    assert isinstance(parsed, dict)
    assert "Subject:" in parsed["body"]
    # When keys missing we should still get dict with empty-ish strings
    parsed2 = imap_mod.ImapHandler(Settings:=None).parse_mesg({})
    assert isinstance(parsed2, dict)
    assert "from" in parsed2 and "tocc" in parsed2 and "body" in parsed2


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