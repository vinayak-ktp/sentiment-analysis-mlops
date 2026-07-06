import os
import sys

# Add the project root to sys.path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
import os
import warnings

import dagshub
import mlflow
from dotenv import load_dotenv

from src.logger import logging

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

load_dotenv()

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
repo_name = os.getenv("DAGSHUB_REPO_NAME")

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{tracking_uri}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri(tracking_uri)
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')

    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
