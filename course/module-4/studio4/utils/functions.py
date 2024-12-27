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
from llama_index.core.schema import TextNode
from pathlib import Path
from llama_index.core.storage import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from datetime import datetime
from tqdm import tqdm
from decimal import Decimal
import uuid

load_dotenv()

# Get the environment variables
host = os.getenv('MYSQL_DB_HOST')
user = os.getenv('MYSQL_DB_USER')
password = os.getenv('MYSQL_DB_PASSWORD')
database = os.getenv('MYSQL_SALES_DB_NAME')

appsheet_app_id = os.getenv('APPSHEET_APP_ID')
appsheet_api_key = os.getenv('APPSHEET_API_KEY')

# Construct the connection string
connection_string = f"mysql+pymysql://{user}:{password}@{host}/energia_global"
connection_string_ventas = f"mysql+pymysql://{user}:{password}@{host}/energia_global_ventas"

# Create the engine
engine = create_engine(connection_string)
engine_ventas = create_engine(connection_string_ventas)

metadata_obj = MetaData()
sql_database = SQLDatabase(engine)
sql_database_ventas = SQLDatabase(engine_ventas)
#table_node_mapping = SQLTableNodeMapping(sql_database)
#sql_retriever = SQLRetriever(sql_database)

# Main functions

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
    print(f"sending url {products_url}")
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



def index_all_tables(
    sql_databases: Dict[str, SQLDatabase] = {"energia_global": sql_database, "energia_global_ventas": sql_database_ventas}, 
    table_index_dir: str = "./table_indices"
) -> Dict[str, VectorStoreIndex]:
    """Index all tables."""

    table_names = [['CLIENTES', 'energia_global_ventas'], ['PRODUCTS', 'energia_global']]
    # no indexed: CAJA, CHEQUES, PRODUCTOS PEDIDOS, STOCK, CUENTAS CORRIENTES, PEDIDOS, CONTROL DE PRECIOS

    if not Path(table_index_dir).exists():
        os.makedirs(table_index_dir)

    vector_index_dict = {}
    
    for table_info in tqdm(table_names, desc="Indexing tables"):
        table_name = table_info[0]
        database_name = table_info[1]
        sql_database = sql_databases[database_name]
        engine = sql_database.engine
        
        print(f"\nIndexing rows in table: {table_name} from database: {database_name}")

        if not os.path.exists(f"{table_index_dir}/{table_name}"):
            with engine.connect() as conn:
                columns_query = (
                    f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'"
                    f"AND TABLE_SCHEMA = '{database_name}' "
                    f"ORDER BY ORDINAL_POSITION"
                )

                cursor = conn.execute(text(columns_query))
                result = cursor.fetchall()
                original_columns = [column[0] for column in result]
                # Create sanitized column names for metadata
                sanitized_columns = [col.replace(' ', '_') for col in original_columns]
                #print("Column order:", original_columns)  # Debug print

                cursor = conn.execute(text(f'SELECT * FROM `{table_name}`'))
                row_tups = [tuple(row) for row in cursor.fetchall()]
                #print("First row values:", row_tups[0])  # Debug print

            nodes = []
            for t in tqdm(row_tups, desc="Processing rows"):
                if row_tups.index(t) == 2000:
                    break

                processed_values = []
                for value in t:
                    if isinstance(value, (datetime, Decimal)):
                        processed_values.append(str(value))
                    elif value is None:
                        processed_values.append("")
                    else:
                        processed_values.append(str(value))

                nodes.append(TextNode(
                    text=str(tuple(processed_values)), 
                    metadata=dict(zip(sanitized_columns, processed_values))
                ))

            try:
                index = VectorStoreIndex(nodes)
            except TypeError as e:
                print(f"TypeError occurred while creating index: {e}")
                continue
            index.set_index_id("vector_index")
            index.storage_context.persist(f"{table_index_dir}/{table_name}")
        else:
            print('index already exists')
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{table_index_dir}/{table_name}"
            )
            index = load_index_from_storage(
                storage_context, index_id="vector_index")
            
        vector_index_dict[table_name] = index

    return vector_index_dict

def index_from_storage(
    sql_databases: Dict[str, SQLDatabase] = {"energia_global": sql_database, "energia_global_ventas": sql_database_ventas}, 
    table_index_dir: str = "./table_indices"
) -> Dict[str, VectorStoreIndex]:
    """Index all tables."""

    table_names = [['CLIENTES', 'energia_global_ventas'], ['PRODUCTS', 'energia_global']]
    # no indexed: CAJA, CHEQUES, PRODUCTOS PEDIDOS, STOCK, CUENTAS CORRIENTES, PEDIDOS, CONTROL DE PRECIOS

    if not Path(table_index_dir).exists():
        os.makedirs(table_index_dir)

    vector_index_dict = {}
    
    for table_info in tqdm(table_names, desc="Indexing tables"):
        table_name = table_info[0]
        database_name = table_info[1]
        sql_database = sql_databases[database_name]
        engine = sql_database.engine
        
        #print(f"\nIndexing rows in table: {table_name} from database: {database_name}")
        storage_context = StorageContext.from_defaults(
            persist_dir=f"{table_index_dir}/{table_name}"
        )
        index = load_index_from_storage(
            storage_context, index_id="vector_index")
            
        vector_index_dict[table_name] = index

    return vector_index_dict


def save_order(data: dict):
    print(Fore.RED + "[save_order] is being executed" + Style.RESET_ALL)

    if not data.get("ID_CLIENTE"):
        return {"status": "error", "response": "ID_CLIENTE is required"}
    if not data.get("OPCIONES"):
        return {"status": "error", "response": "OPCIONES are required"}
    
    try:
        # Generate IDs
        order_id = str(uuid.uuid4())[:8].upper()
        option_ids = [str(uuid.uuid4())[:8].upper() for _ in data["OPCIONES"]]


        # 2. Create all options
        options_data = []
        for opcion, option_id in zip(data["OPCIONES"], option_ids):
            options_data.append({
                "ID_KEY": option_id,
                "ID_PEDIDO": order_id,
                "MONEDA": opcion.get("MONEDA"),
                "DESCUENTO_GENERAL": opcion.get("DESCUENTO_GENERAL", 0)
            })
        
        if options_data:
            result = appsheet_add(options_data, "OPCIONES_PRESUPUESTOS")
            if result["status"] != "success":
                return {"status": "error", "response": result["response"]}

        # 3. Create all products
        products_data = []
        for opcion, option_id in zip(data["OPCIONES"], option_ids):
            for producto in opcion.get("PRODUCTOS_PEDIDOS", []):
                products_data.append({
                    "ID OPCION": option_id,
                    "ID PRODUCTO": producto.get("ID_PRODUCTO"),
                    "CANTIDAD": producto.get("CANTIDAD"),
                    "PRECIO LISTA S IVA": producto.get("PRECIO_LISTA_S_IVA")
                })

        if products_data:
            result = appsheet_add(products_data, "PRODUCTOS PEDIDOS")
            if result["status"] != "success":
                return {"status": "error", "response": result["response"]}
        

        # 1. Create main order
        order_data = {
            "ID_KEY": order_id,
            "ID_CLIENTE": data.get("ID_CLIENTE"),
            "VENDEDOR": data.get("VENDEDOR", ""),
            "TIPO DE ENTREGA": data.get("TIPO_DE_ENTREGA", ""),
            "DIRECCION": data.get("DIRECCION", ""),
            "CARGAR COMO PEDIDO": data.get("CARGAR_COMO_PEDIDO", "")
        }
        result = appsheet_add(order_data, "PEDIDOS")
        if result["status"] != "success":
            return {"status": "error", "response": result["response"]}

        # Mark as saved
        appsheet_edit({"ID_KEY": order_id, "GUARDADO": 1}, "PEDIDOS")

        return {"status": "success", "order_id": order_id}

    except Exception as e:
        return {"status": "error", "response": f"Error saving order: {str(e)}"}

if __name__ == "__main__":

    mock_order = {
    "ID_CLIENTE": "1",
    "TIPO_DE_ENTREGA": "CLIENTE",
    "VENDEDOR": "ESTEBAN",
    "OPCIONES": [
        {
            "MONEDA": "PESO",
            "PRODUCTOS_PEDIDOS": [
                {
                    "ID_PRODUCTO": 1001,
                    "CANTIDAD": 5,
                },
                {
                    "ID_PRODUCTO": 1002,
                    "CANTIDAD": 3,
                }
            ],
            "DESCUENTO_GENERAL": 10
        },
        {
            "MONEDA": "DOLAR",
            "PRODUCTOS_PEDIDOS": [
                {
                    "ID_PRODUCTO": 1001,
                    "CANTIDAD": 5
                }
            ],
            "DESCUENTO_GENERAL": 5
        }
    ],

    }

    response = save_order(mock_order)




    """ vector_index_dict = index_all_tables({"energia_global": sql_database, "energia_global_ventas": sql_database_ventas})

    test_retriever = vector_index_dict["PRODUCTS"].as_retriever(
    similarity_top_k=5
    )
    nodes = test_retriever.retrieve("EG-GEN-M-10000")
    for node in nodes:
        print("Content:", node.get_content())
        print("Metadata:", node.metadata)
        print("---") """

    """ 
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
    print(response) """

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