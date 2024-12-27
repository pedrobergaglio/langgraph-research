def filter_required_fields(schema: dict) -> dict:
    def is_optional(field_schema):
        return (isinstance(field_schema.get("type"), list) and 
                "null" in field_schema["type"])
    
    def filter_field(field_schema, depth=0):
        if not isinstance(field_schema, dict) or depth > 3:
            return None
            
        filtered_schema = field_schema.copy()
        
        # Check if field is optional
        if is_optional(field_schema):
            return None
            
        # Handle array items
        if "items" in field_schema:
            filtered_items = filter_field(field_schema["items"], depth + 1)
            if filtered_items is None:
                return None
            filtered_schema["items"] = filtered_items
            filtered_schema["type"] = "array"
            
        # Handle object properties
        if "properties" in field_schema:
            filtered_props = {}
            for prop_name, prop_schema in field_schema["properties"].items():
                filtered_prop = filter_field(prop_schema, depth + 1)
                if filtered_prop is not None:
                    filtered_props[prop_name] = filtered_prop
            
            if not filtered_props:
                return None
                
            filtered_schema["properties"] = filtered_props
            
            if "required" in field_schema:
                required = [r for r in field_schema["required"] 
                          if r in filtered_props]
                if required:
                    filtered_schema["required"] = required
                
        return filtered_schema

    filtered = filter_field(schema)
    return filtered if filtered is not None else {"type": "object", "properties": {}}

# test cases with child null-union fields

# test case 1
schema = {
            "title": "Order",
            "type": "object",
            "properties": {
            "ID_CLIENTE": {
                "type": "string",
                "description": "Client ID"
            },
            "TIPO_DE_ENTREGA": {
                "type": "string",
                "enum": ["CLIENTE", "RETIRA EN FÁBRICA", "OTRO"],
                "description": "Delivery type"
            },
            "VENDEDOR": {
                "type": "string",
                "enum": ["ESTEBAN", "FLORENCIA", "NICOLAS", "PATRICIA", "FEDERICO", "Prueba"],
                "description": "The user that is talking to you, ask his name, there's no default"
            },
            "OPCIONES": {
                "type": "array",
                "items": {
                "type": "object",
                "properties": {
                "MONEDA": {
                "type": "string",
                "enum": ["PESO", "DOLAR"],
                "description": "Currency. Salesperson must define which is the currency to use, There is no default"
                },
                "PRODUCTOS_PEDIDOS": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "PRECIO_LISTA_S_IVA": {
                    "type": ["float", "null"],
                    "description": "Optional list price without VAT, if not provided, it must be left as null because the system itself will calculate it"
                    }
                    },
                    "required": ["PRECIO_LISTA_S_IVA"]
                }
                },
                "DESCUENTO_GENERAL": {
                    "type": ["number", "null"],
                    "description": "General discount"
                }
                },
                "required": ["MONEDA", "PRODUCTOS_PEDIDOS", "DESCUENTO_GENERAL"]
                }
            },
            "CARGAR_COMO_PEDIDO": {
                "type": ["boolean", "null"],
                "description": "Defaults to false, because this is actually an estimate, not a confirmed order"
            }
            },
            "required": ["ID_CLIENTE", "TIPO_DE_ENTREGA", "VENDEDOR", "OPCIONES", "CARGAR_COMO_PEDIDO"]
            }

expected = {
    "type": "object",
    "properties": {}
}

print(filter_required_fields(schema))

# test case 2

schema = {
            "title": "Order",
            "type": "object",
            "properties": {
            "OPCIONES": {
                "type": "array",
                "items": {
                "type": "object",
                "properties": {
                "PRODUCTOS_PEDIDOS": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                    "PRECIO_LISTA_S_IVA": {
                    "type": ["float", "null"],
                    "description": "Optional list price without VAT, if not provided, it must be left as null because the system itself will calculate it"
                    }
                    },
                    "required": ["PRECIO_LISTA_S_IVA"]
                }
                }
                },
                "required": ["PRODUCTOS_PEDIDOS"]
                }
            },
            },
            "required": ["OPCIONES"]
            }

expected = {
    "type": "object",
    "properties": {}
}

print("")
print("")
print(filter_required_fields(schema))
print("")
print("")


# test case 3 with 3 levels

schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": ["string", "null"]
        },
        "age": {
            "type": ["integer", "null"]
        },
        "address": {
            "type": "object",
            "properties": {
                "street": {
                    "type": ["string", "null"]
                },
                "city": {
                    "type": ["string", "null"]
                },
                "country": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": ["string", "null"]
                        }
                    }
                }
            }
        }
    }
}

expected = {
    "type": "object",
    "properties": {}
}

print(filter_required_fields(schema))
