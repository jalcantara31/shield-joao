__all__ = ["Cognite"]

from barrier.exceptions import CogniteConnectionError
from .data_loader import DataLoader

from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import OAuthClientCredentials
from loguru import logger



class Cognite(DataLoader):
    
    def __init__(self,
                     name           : str,
                     client_id      : str,
                     client_secret  : str,
                     tenant_id      : str,
                     project        : str="publicdata",
                    ):
            """
            Initializes a new instance of the class.

            Parameters:
            ----------
            name : str
                The name of the client.
            client_id : str
                The client ID for authentication.
            client_secret : str
                The client secret for authentication.
            tenant_id : str
                The tenant ID for the Azure Active Directory.
            project : str, optional
                The project name (default is "publicdata").

            Raises:
            ------
            CogniteConnectionError
                If there is an error connecting to Cognite.
            """
            
            self.name = name
            self.tenant_id = tenant_id
            self.project = project
            self.base_url = f"https://api.cognitedata.com"
            self.url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
            self._client_id = client_id
            self._client_secret = client_secret 
            self._credentials = OAuthClientCredentials(
                token_url = self.url,
                client_id = self._client_id,
                client_secret = self._client_secret,
                scopes = [f"{self.base_url}/.default"],
            )
            self._config = ClientConfig(
                client_name=name , 
                project=project, 
                credentials=self._credentials, 
                base_url=self.base_url)
            self._client = CogniteClient( self._config )

            try:
                token_status = self._client.iam.token.inspect()
                logger.info(f"{token_status.projects[0].project_url_name}")
            except Exception as e:
                print(e)
                raise CogniteConnectionError(self.name)
