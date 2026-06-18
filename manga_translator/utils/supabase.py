import os
from supabase import create_client, Client

def create_service_role_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    missing = [name for name, value in (
        ("SUPABASE_URL", url),
        ("SUPABASE_SERVICE_ROLE_KEY", key),
    ) if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return create_client(url, key)


def upload_file(supabase: Client, file, path, bucket) -> str:
    response = supabase.storage.from_(bucket).upload(
        path=path,
        file=file,
        file_options={
            "cache-control": "3600", 
            "upsert": "true" # 为了支持重试，如果后续ts步骤失败，会触发重新翻译
        }
    )
    return response.path