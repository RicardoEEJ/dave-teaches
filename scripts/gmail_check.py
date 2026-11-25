import sys

from pathlib import Path

# Add the project root (the folder containing app/) to the import path,
# based on this file's location rather than the current working directory.
sys.path.append(str(Path(__file__).resolve().parent.parent))


from app.gmail import Gmail, get_body, get_headers

gmail = Gmail()
print("Connected as", gmail.address)

message_id = gmail.list_new_message_ids()[0]
message = gmail.get_message(message_id)
headers = get_headers(message)
print(headers["from"], "|", headers.get("subject"))
print(get_body(message)[:500])

gmail.send_reply(message, "Thanks, we received your email.")
print("Reply sent")

