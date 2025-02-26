def convert_kindle_to_markdown(input_file, output_file):
    """
    Convierte un archivo exportado de subrayados y notas de Kindle a formato Markdown.

    Args:
        input_file (str): Ruta del archivo de entrada.
        output_file (str): Ruta del archivo de salida.
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    markdown_output = []
    current_block = []

    for line in lines:
        line = line.strip()

        if line == "==========":
            # Procesar el bloque acumulado
            if current_block:
                title_line = current_block[0]
                type_line = current_block[1]
                content_lines = current_block[2:]

                if "- La subrayado en la posición" in type_line:
                    markdown_output.append("\n".join(content_lines))
                elif "- La nota en la posición" in type_line:
                    note_content = "\n".join(content_lines)
                    markdown_output.append(f"> **_NOTE:_** {note_content.strip()}")

                markdown_output.append("---")
                current_block = []

        elif line:
            current_block.append(line)

    # Escribir el resultado en formato Markdown
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("\n\n".join(markdown_output).strip() + "\n")

# Uso del script
input_path = "My Clippings.txt"  # Reemplaza con la ruta real del archivo
output_path = "My Clippings.md"  # Reemplaza con la ruta deseada para el archivo de salida
convert_kindle_to_markdown(input_path, output_path)
