# Subagente: tester (bounded context: usuarios)
def test_normalize_user_record():
    record = {"Name": "  Ada ", "EMAIL": "ADA@example.com"}
    assert normalize(record) == {"name": "ada", "email": "ada@example.com"}

def test_handles_missing_keys():
    assert normalize({}) == {"name": None, "email": None}

def test_lowercases_emails_only():
    record = {"name": "Ada", "email": "ADA@example.com"}
    assert normalize(record)["email"] == "ada@example.com"
    assert normalize(record)["name"] == "Ada"