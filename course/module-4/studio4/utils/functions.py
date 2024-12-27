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
import json

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

# Main functions





# API
def create_order_api(data: dict):
    
    print(Fore.RED + "[create_order] is being executed" + Style.RESET_ALL)
    
    # Extract required fields from data
    order_data = {
        'ID_CLIENTE': data.get('ID_CLIENTE', ''),
        'TIPO_DE_ENTREGA': data.get('TIPO_DE_ENTREGA', ''),
        'METODO_DE_PAGO': data.get('METODO_DE_PAGO', ''),
        'DIRECCION': data.get('DIRECCION', ''),
        'NOTA': data.get('NOTA', '')
    }

    # Validate required fields
    if not order_data['ID_CLIENTE']:
        return {"status": "error", "response": "ID_CLIENTE is required"}

    if not order_data['TIPO_DE_ENTREGA']:
        return {"status": "error", "response": "TIPO_DE_ENTREGA is required"}

    if not order_data['METODO_DE_PAGO']:
        return {"status": "error", "response": "METODO_DE_PAGO is required"}

    response = appsheet_add(order_data, 'PEDIDOS')

    #HERE WE COULD JUST RETURN THE RESPONSE, BUT WE NEED TO SPECIFY THE ORDER_ID.
    if response['status'] == 'success':
        try:
            response_json = json.loads(response['response'])
            return {"status": "success", "order_id": response_json['Rows'][0]['ID_KEY']}
        except Exception as e:
            return {"status": "error", "response": f"API Success but error parsing response: {str(e)}"}
    else:
        return {"status": "error", "response": response['status']}

def add_products_to_order_api(data: dict):
    print(Fore.RED + "[add_products_to_order] is being executed" + Style.RESET_ALL)
    print(f"Received data: {data}")

    order_id = data.get('order_id')
    products = data.get('products', [])

    if not order_id:
        return {"status": "error", "response": "order_id is required"}

    if not products or len(products) == 0:
        return {"status": "error", "response": "products array is required"}

    # Transform products into rows for single request
    rows = [{
        "ID_PRODUCTO": product["ID_PRODUCTO"],
        "TIPO": product["TIPO"],
        "COLOR": product["COLOR"],
        "CANTIDAD": product["CANTIDAD"],
        "ID_PEDIDO": order_id
    } for product in products]

    # Single API call with all products
    result = appsheet_add(rows, "PRODUCTOS PEDIDOS")

    if result.get("status") != "success":
        return {
            "status": "error",
            "response": f"Error adding products: {result.get('response')}"
        }

    try:
        #response_json = json.loads(result['response'])['Rows'] # we need the response to have rows.
        return {"status": "success", "order_id": order_id}
    except Exception as e:
        return {"status": "error", "response": f"API Success but error parsing response: {str(e)}"}

def save_order(data: dict):
    print(Fore.RED + "[save_order] is being executed" + Style.RESET_ALL)

    order_id = data.get("order_id")

    if not order_id:
        return {"status": "error", "response": "order_id is required"}

    appsheet_edit({"ID_KEY": order_id, "GUARDADO": 1}, "PEDIDOS")

    return {"status": "success", "order_id": order_id}

# Utils functions
def fetch_page(input: dict):
    response = requests.get(input["url"])
    #return {"response":response.text}

def appsheet_add(data: dict|List[dict], table_name: str = None):
    print(Fore.RED + "[appsheet_add] is being executed" + Style.RESET_ALL)
    print(f"Received data: {data}")
    print(f"Table name parameter: {table_name}")

    # Get table_name from parameter or data
    if table_name is None:
        table_name = data.pop("table_name", None)
    if table_name is None:
        return {"status": "error", "response": "table_name is required"}
    
    if isinstance(data, dict):
        data = [data]
    
    row_data = []
    for item in data:

        # Remove empty strings
        row = {k: v for k, v in item.items() if v != ""}
        row_data.append(row)

    products_url = f"https://api.appsheet.com/api/v2/apps/{appsheet_app_id}/tables/{table_name}/Action"
    headers = {
        "ApplicationAccessKey": appsheet_api_key,
        "Accept": "*/*", 

    }

    request = {
        "Action": "Add",
        "Properties": {"Locale": "en-US"},
        "Rows": row_data
    }

    print(f"Sending request: {request}")
    response = requests.post(products_url, headers=headers, json=request)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")

    if response.status_code == 200:
        if response.text != "":
            return {"status": "success", "response": response.text}
        else:
            return {"status": "error", "response": f"Error parsing response: response is empty"}
    else:
        return {"status": "error", "response": response.text}
    
def appsheet_edit(data: dict|List[dict], table_name: str = None):
    print(Fore.RED + "[appsheet_edit] is being executed" + Style.RESET_ALL)
    print(f"Received data: {data}")
    print(f"Table name parameter: {table_name}")

    # Get table_name from parameter or data
    if table_name is None:
        table_name = data.pop("table_name", None)
    if table_name is None:
        return {"status": "error", "response": "table_name is required"}
    
    if isinstance(data, dict):
        data = [data]
    
    row_data = []
    for item in data:
        # Remove empty strings
        row = {k: v for k, v in item.items() if v != ""}
        row_data.append(row)

    products_url = f"https://api.appsheet.com/api/v2/apps/{appsheet_app_id}/tables/{table_name}/Action"
    headers = {
        "ApplicationAccessKey": appsheet_api_key,
        "Accept": "*/*",
    }

    request = {
        "Action": "Edit",
        "Properties": {"Locale": "en-US"},
        "Rows": row_data
    }

    print(f"Sending request: {request}")
    response = requests.post(products_url, headers=headers, json=request)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")

    if response.status_code == 200:
        try:
            return {"status": "success", "response": response.text}
        except Exception as e:
            return {"status": "error", "response": f"Error parsing response: {str(e)}"}
    else:
        return {"status": "error", "response": response.text}


if __name__ == "__main__":

    test_order = {
        'order_id': 'b8c74e0a', 
        'products': [
            {
                'ID_PRODUCTO': 1, 
                'TIPO': 'ESTANDAR', 
                'COLOR': 'EUCALIPTO', 
                'CANTIDAD': 2
            },
            {
                'ID_PRODUCTO': 1, 
                'TIPO': 'ESTANDAR', 
                'COLOR': 'FRESIA', 
                'CANTIDAD': 3
            }
        ]
    }
    response = add_products_to_order_api(test_order)
    print(response)

    """ # Test create_order function
    test_create_order = {
        'ID_CLIENTE': '996778',
        'TIPO_DE_ENTREGA': 'RETIRA EN FÁBRICA',
        'METODO_DE_PAGO': 'EFECTIVO',
        'DIRECCION': '',
        'NOTA': ''
    }
    response = create_order(test_create_order)
    print(response) """



"""

cliente id

tipo de entrega

VENDEDOR enum: ESTEBAN
FLORENCIA
NICOLAS
PATRICIA
FEDERICO
Prueba

OPCIONES list:

    - MONEDA enum: PESO, DOLAR  
    - PRODUCTOS PEDIDOS
        - ID PRODUCTO
        - CANTIDAD
        - opcional
            - PRECIO LISTA S/IVA: float
    - OPCIONALES: 
        - DESCUENTO GENERAL : float

        


CARGAR COMO PEDIDO: bool

"""