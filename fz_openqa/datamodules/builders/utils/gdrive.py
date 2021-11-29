import os
import pickle
import shutil
from typing import Dict
from typing import List

from apiclient import errors
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class Gdrive:
    def __init__(
        self,
        client_secret_file: str = f"{os.getcwd()}/fz_openqa/datamodules/builders/utils/client-secrets.json",  # noqa: E501
        api_name: str = "drive",
        api_version: str = "v3",
        scopes: List[str] = ["https://www.googleapis.com/auth/drive"],
    ):
        super().__init__()

        self._instance = self._create_service(client_secret_file, api_name, api_version, scopes)

    @staticmethod
    def _create_service(client_secret_file, api_name, api_version, *scopes, prefix=""):
        """Instantiate a Google Drive Api service

        Args:
            client_secret_file (json) :  JSON file containing credentials to access the api
            api_name (str) : name of Google api to access
            api_version (str) : version of api
            scope (str) : allows access to the Application Data folder.

        """
        scopes = [scope for scope in scopes[0]]
        cred = None
        working_dir = os.getcwd()
        token_dir = "token files"
        pickle_file = f"token_{api_name}_{api_version}{prefix}.pickle"

        # Check if token dir exists first, if not, create the folder
        if not os.path.exists(os.path.join(working_dir, token_dir)):
            os.mkdir(os.path.join(working_dir, token_dir))

        if os.path.exists(os.path.join(working_dir, token_dir, pickle_file)):
            with open(os.path.join(working_dir, token_dir, pickle_file), "rb") as token:
                cred = pickle.load(token)

        if not cred or not cred.valid:
            if cred and cred.expired and cred.refresh_token:
                cred.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scopes)
                cred = flow.run_local_server()

            with open(os.path.join(working_dir, token_dir, pickle_file), "wb") as token:
                pickle.dump(cred, token)

        try:
            service = build(api_name, api_version, credentials=cred)
            print(api_name, api_version, "service created successfully")
            return service
        except Exception as e:
            print(e)
            print(f"Failed to create service instance for {api_name}")
            os.remove(os.path.join(working_dir, token_dir, pickle_file))
            return None

    def _retrieve_all_files(self, folder_id: str = "1mxQF7zm85cgP8jIvuRokCxopEDwmFlHb") -> Dict:
        """Retrieve a Dict of file resources.

        Args:
            folder_id (str) : Id of Drive folder to interact with.
        Returns:
            Dict of file resources.
        """
        result = {}
        try:
            param = {"q": f"'{folder_id}' in parents and trashed=false"}
            files = self._instance.files().list(**param).execute()

            for f in files["files"]:
                result[f["name"]] = f["id"]
        except (errors.HttpError, errors) as e:
            print("An error occurred: %s" % e)
        return result

    def upload_to_drive(self, path_to_file: str) -> Dict:
        """Update or create file on Gdrive

        Args:
            path_to_file (str) : Path defining what file to upload
        Returns:
            Response from the Google Drive Api
        """
        file_name = path_to_file.split("/")[-1]
        shutil.make_archive(path_to_file, "zip", path_to_file)
        file_list = self._retrieve_all_files(folder_id="1mxQF7zm85cgP8jIvuRokCxopEDwmFlHb")

        content = MediaFileUpload(f"{path_to_file}.zip", mimetype="*/*", resumable=True)
        if f"{file_name}.zip" in file_list.keys():
            file_id = file_list[file_name]
            # Update existing file based on id
            file = self._instance.files().update(fileId=file_id, media_body=content).execute()

        else:
            file_metadata = {
                "name": f"{file_name}.zip",
                "parents": ["1mxQF7zm85cgP8jIvuRokCxopEDwmFlHb"],
                "mimeType": "*/*",
            }
            # Create new file
            file = self._instance.files().create(body=file_metadata, media_body=content).execute()

        return file
