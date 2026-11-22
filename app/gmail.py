import base64
import re
from difflib import SequenceMatcher
from email.message import EmailMessage
from email.utils import parseaddr

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.config import settings

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
PROCESSED_LABEL = "ai-processed"
ESCALATED_LABEL = "needs-human"
DUPLICATE_SIMILARITY = 0.7  # 70% or more of the body text matches
DUPLICATE_MIN_COUNT = 3  # "sent more than 2 times"


class Gmail:
    def __init__(self) -> None:
        creds = Credentials(
            token=None,
            refresh_token=settings.google_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            scopes=SCOPES,
        )
        self.service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        self.users = self.service.users()
        self.address = self.users.getProfile(userId="me").execute()["emailAddress"]
        self.label_ids = self._ensure_labels()

    def _ensure_labels(self) -> dict[str, str]:
        labels = self.users.labels().list(userId="me").execute().get("labels", [])
        ids = {label["name"]: label["id"] for label in labels}
        for name in (PROCESSED_LABEL, ESCALATED_LABEL):
            if name not in ids:
                label = self.users.labels().create(userId="me", body={"name": name}).execute()
                ids[name] = label["id"]
        return ids

    def list_new_message_ids(self, max_results: int = 10) -> list[str]:
        response = self.users.messages().list(
            userId="me",
            q=f"in:inbox -label:{PROCESSED_LABEL}",
            maxResults=max_results,
        ).execute()
        return [message["id"] for message in response.get("messages", [])]

    def get_message(self, message_id: str) -> dict:
        return self.users.messages().get(userId="me", id=message_id, format="full").execute()

    def get_thread(self, thread_id: str) -> dict:
        return self.users.threads().get(userId="me", id=thread_id, format="full").execute()

    def add_label(self, message_id: str, name: str) -> None:
        self.users.messages().modify(
            userId="me", id=message_id, body={"addLabelIds": [self.label_ids[name]]}
        ).execute()

    def is_escalated(self, thread: dict) -> bool:
        label_id = self.label_ids[ESCALATED_LABEL]
        return any(label_id in m.get("labelIds", []) for m in thread["messages"])

    def send_reply(self, message: dict, body: str) -> dict:
        headers = get_headers(message)
        subject = headers.get("subject", "")
        reply = EmailMessage()
        reply["To"] = headers.get("reply-to") or headers["from"]
        reply["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        reply["In-Reply-To"] = headers["message-id"]
        reply["References"] = f"{headers.get('references', '')} {headers['message-id']}".strip()
        reply.set_content(body)
        raw = base64.urlsafe_b64encode(reply.as_bytes()).decode()
        return self.users.messages().send(
            userId="me", body={"raw": raw, "threadId": message["threadId"]}
        ).execute()

    # --- Cleanup: spam and repeated messages -------------------------------
    # These move mail to Trash (recoverable for 30 days) rather than deleting it
    # permanently. Permanent delete needs the broader https://mail.google.com/
    # scope; trash works with the gmail.modify scope this class already uses.

    def trash_message(self, message_id: str) -> None:
        self.users.messages().trash(userId="me", id=message_id).execute()

    def _list_ids(self, max_results: int, **params) -> list[str]:
        """List message ids, following pages until max_results is reached."""
        ids: list[str] = []
        page_token = None
        while len(ids) < max_results:
            response = self.users.messages().list(
                userId="me",
                maxResults=min(500, max_results - len(ids)),
                pageToken=page_token,
                **params,
            ).execute()
            ids.extend(m["id"] for m in response.get("messages", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return ids

    def trash_spam(self, max_results: int = 500, dry_run: bool = False) -> list[str]:
        """Move everything Gmail has filed under Spam to Trash. Returns the ids."""
        ids = self._list_ids(max_results, labelIds=["SPAM"], includeSpamTrash=True)
        if not dry_run:
            for message_id in ids:
                self.trash_message(message_id)
        return ids

    def find_duplicates(
        self,
        query: str = "in:inbox",
        max_results: int = 200,
        threshold: float = DUPLICATE_SIMILARITY,
        min_count: int = DUPLICATE_MIN_COUNT,
    ) -> list[list[str]]:
        """Group messages from the same sender whose bodies are >= threshold similar.

        Only groups with at least min_count messages are returned. Each group is
        a list of message ids, oldest first. Our own sent replies are ignored, and
        so is anything in an active conversation (see _in_active_thread), so a
        customer's follow-ups like "still waiting on order #123" are never trashed.
        """
        active_cache: dict[str, bool] = {}
        by_sender: dict[str, list[tuple[int, str, str]]] = {}
        for message_id in self._list_ids(max_results, q=query):
            message = self.get_message(message_id)
            sender = parseaddr(get_headers(message).get("from", ""))[1].lower()
            if not sender or sender == self.address.lower():
                continue
            if self._in_active_thread(message["threadId"], active_cache):
                continue
            text = _normalize(strip_quoted(get_body(message)))
            by_sender.setdefault(sender, []).append(
                (int(message.get("internalDate", 0)), message_id, text)
            )

        groups: list[list[str]] = []
        for messages in by_sender.values():
            if len(messages) < min_count:
                continue
            messages.sort()  # oldest first
            clusters: list[tuple[str, list[str]]] = []  # (representative text, ids)
            for _, message_id, text in messages:
                for rep_text, ids in clusters:
                    if _similar(rep_text, text, threshold):
                        ids.append(message_id)
                        break
                else:
                    clusters.append((text, [message_id]))
            groups.extend(ids for _, ids in clusters if len(ids) >= min_count)
        return groups

    def _in_active_thread(self, thread_id: str, cache: dict[str, bool]) -> bool:
        """True if the thread is an ongoing conversation rather than a lone message.

        That means it has more than one message (follow-ups, or our replies) or
        it has been escalated to a human.
        """
        if thread_id not in cache:
            thread = self.users.threads().get(
                userId="me", id=thread_id, format="minimal"
            ).execute()
            cache[thread_id] = len(thread["messages"]) > 1 or self.is_escalated(thread)
        return cache[thread_id]

    def trash_duplicates(self, dry_run: bool = True, **find_kwargs) -> list[str]:
        """Trash repeated messages, keeping the oldest copy in each group.

        dry_run defaults to True because similarity is a heuristic: call it once,
        check the returned ids, then call again with dry_run=False.
        """
        to_trash = [mid for group in self.find_duplicates(**find_kwargs) for mid in group[1:]]
        if not dry_run:
            for message_id in to_trash:
                self.trash_message(message_id)
        return to_trash


def get_headers(message: dict) -> dict[str, str]:
    return {h["name"].lower(): h["value"] for h in message["payload"].get("headers", [])}


def get_body(message: dict) -> str:
    def find_text(part: dict) -> str:
        data = part.get("body", {}).get("data")
        if part.get("mimeType") == "text/plain" and data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        for child in part.get("parts", []):
            if text := find_text(child):
                return text
        return ""

    return find_text(message["payload"]) or message.get("snippet", "")


def has_attachments(message: dict) -> bool:
    def walk(part: dict) -> bool:
        return bool(part.get("filename")) or any(walk(p) for p in part.get("parts", []))

    return walk(message["payload"])


def strip_quoted(text: str) -> str:
    """Drop the quoted earlier messages that mail clients add below a reply."""
    lines = []
    for line in text.splitlines():
        if line.startswith("On ") and line.rstrip().endswith("wrote:"):
            break
        if not line.startswith(">"):
            lines.append(line)
    return "\n".join(lines).strip()


def _normalize(text: str, limit: int = 5000) -> str:
    """Lowercase and collapse whitespace so formatting differences don't count."""
    return re.sub(r"\s+", " ", text).strip().lower()[:limit]


def _similar(a: str, b: str, threshold: float) -> bool:
    if not a and not b:
        return True
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    # The quick ratios are cheap upper bounds; skip the full compare when they fail.
    return (
        matcher.real_quick_ratio() >= threshold
        and matcher.quick_ratio() >= threshold
        and matcher.ratio() >= threshold
    )


def format_thread(thread: dict, own_address: str) -> str:
    """Render a Gmail thread as plain text, oldest message first."""
    first = get_headers(thread["messages"][0])
    parts = [f"Subject: {first.get('subject', '')}"]
    for message in thread["messages"]:
        sender = get_headers(message).get("from", "")
        role = "Harbor & Pine support" if own_address.lower() in sender.lower() else "Customer"
        parts.append(f"{role} ({sender}):\n{strip_quoted(get_body(message))}")
    return "\n\n".join(parts)