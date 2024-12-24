import requests
from typing import List, Dict
from colorama import Fore, Style
from sqlalchemy import create_engine, MetaData
from sqlalchemy.sql import text
from dotenv import load_dotenv
import os
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.objects import SQLTableNodeMapping

load_dotenv()

# Get the environment variables
host = os.getenv('MYSQL_DB_HOST')
user = os.getenv('MYSQL_DB_USER')
password = os.getenv('MYSQL_DB_PASSWORD')
database = os.getenv('MYSQL_SALES_DB_NAME')

appsheet_app_id = os.getenv('APPSHEET_APP_ID')
appsheet_api_key = os.getenv('APPSHEET_API_KEY')

# Construct the connection string
connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"

# Create the engine
engine = create_engine(connection_string)

metadata_obj = MetaData()
sql_database = SQLDatabase(engine)
table_node_mapping = SQLTableNodeMapping(sql_database)
sql_retriever = SQLRetriever(sql_database)

def fetch_page(input: dict):
    response = requests.get(input["url"])
    #return {"response":response.text}

def appsheet_add(data: dict):
    print(Fore.RED + "[appsheet_add] is being executed" + Style.RESET_ALL)
    print(f"Received data: {data}")

    table_name = data.pop("table_name", None)
    if table_name is None:
        return {"status": "error", "response": "table_name is required"}

    # Remove empty strings
    row_data = {k: v for k, v in data.items() if v != ""}

    products_url = f"https://api.appsheet.com/api/v2/apps/{appsheet_app_id}/tables/{table_name}/Action"

    headers = {
        "Content-Type": "application/json",
        "ApplicationAccessKey": appsheet_api_key
    }

    request = {
        "Action": "Add",
        "Properties": {"Locale": "en-US"},
        "Rows": [row_data]  # AppSheet expects array of rows
    }

    print(f"Sending request: {request}")
    response = requests.post(products_url, headers=headers, json=request)

    if response.status_code == 200:
        try:
            return {"status": "success", "order_id": response.json().get('Rows')[0]['ID_KEY']}
        except Exception as e:
            return {"status": "error", "response": f"API Success but error parsing response: {str(e)}"}
    else:
        return {"status": "error", "response": response.text}

def appsheet_edit(rows: List[Dict] | Dict, table_name: str):
    print(Fore.RED + "[appsheet_edit] is being executed" + Style.RESET_ALL)

    if isinstance(rows, dict):
        rows = [rows]

    products_url = f"https://api.appsheet.com/api/v2/apps/{appsheet_app_id}/tables/{table_name}/Action"

    headers = {
        "Content-Type": "application/json",
        "ApplicationAccessKey": appsheet_api_key
    }

    request = {
        "Action": "Edit",
        "Properties": {"Locale": "en-US"},
        "Rows": rows
    }

    print(request)

    response = requests.post(products_url, headers=headers, json=request)
    return response

