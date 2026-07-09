# mcp_server/identity.py
@mcp.tool()
def normalize_user_record(raw: dict) -> dict:
    if "@" not in raw.get("email", ""):
        raise ValueError("invalid email shape")
    return {"name": raw["name"].strip().title(), "email": raw["email"].lower()}