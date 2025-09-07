#!/usr/bin/env python3
import os
from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")
LOCAL_PRICES_ROOT = Path("data-lake/prices")

def drive_client():
    sa_path = os.path.expanduser("~/.config/gspread/service_account.json")
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    gauth = GoogleAuth()
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(sa_path, scopes=scopes)
    return GoogleDrive(gauth)

def list_all_files(drive, parent_id):
    # Recursive listing (folders + files)
    q = f"'{parent_id}' in parents and trashed=false"
    children = drive.ListFile({'q': q}).GetList()
    for c in children:
        yield c
        if c['mimeType'] == 'application/vnd.google-apps.folder':
            yield from list_all_files(drive, c['id'])

def main():
    if not DRIVE_FOLDER_ID:
        raise SystemExit("Missing env DRIVE_FOLDER_ID")

    drive = drive_client()

    # Find the "prices" folder under the root lake folder
    q = (
        f"mimeType='application/vnd.google-apps.folder' and trashed=false and "
        f"name='prices' and '{DRIVE_FOLDER_ID}' in parents"
    )
    prices_folders = drive.ListFile({'q': q}).GetList()
    if not prices_folders:
        print("No 'prices' folder found in Drive lake yet.")
        return
    prices_root_id = prices_folders[0]['id']

    LOCAL_PRICES_ROOT.mkdir(parents=True, exist_ok=True)

    # Walk and download parquet files
    for entry in list_all_files(drive, prices_root_id):
        if entry['mimeType'] == 'application/vnd.google-apps.folder':
            # replicate folder structure locally
            # Build a relative path from folder chain using names
            # We'll fetch parents on demand
            continue
        title = entry['title']
        if not title.endswith(".parquet"):
            continue

        # Build local path from parents by querying parent chain names
        # (cheap version: query parents names; Drive allows multiple parents but we model one)
        parents = entry.get('parents', [])
        rel_parts = [title]
        parent_id = parents[0]['id'] if parents else None
        while parent_id and parent_id != prices_root_id:
            p = drive.CreateFile({'id': parent_id})
            p.FetchMetadata(fields='id,title,parents,mimeType')
            rel_parts.insert(0, p['title'])
            plist = p.get('parents', [])
            parent_id = plist[0]['id'] if plist else None

        out_path = LOCAL_PRICES_ROOT.joinpath(*rel_parts)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue  # simple skip; could add size/hash checks

        print(f"Downloading {out_path.relative_to(LOCAL_PRICES_ROOT)}")
        f = drive.CreateFile({'id': entry['id']})
        f.GetContentFile(str(out_path))

    print("Sync complete.")

if __name__ == "__main__":
    main()
