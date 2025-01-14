from flask import Flask, render_template
import os
import io
import base64
import sys
from subprocess import run

app = Flask(__name__)

# Ruta al directorio donde están tus archivos (Simulacion)
SIMULACION_DIR = os.path.join(os.path.dirname(__file__), 'Simulacion')

@app.route('/')
def list_scripts():
    try:
        # Listar los scripts disponibles
        files = [f for f in os.listdir(SIMULACION_DIR) if f.endswith('.py')]
        return render_template('list_scripts.html', files=files)
    except Exception as e:
        return str(e), 500
    
@app.route('/run/<script_name>')
def run_script(script_name):
    script_path = os.path.join(SIMULACION_DIR, script_name)

    # Verificar que el archivo existe y es un .py
    if not os.path.exists(script_path):
        return f"El archivo {script_name} no existe.", 404
    if not script_name.endswith('.py'):
        return "Solo se pueden ejecutar archivos Python (.py).", 400

    try:
        # Capturar la salida del print
        output_buffer = io.StringIO()
        sys.stdout = output_buffer  # Redirigir stdout a un buffer

        # Ejecutar el script de Python capturando gráficos
        globals_dict = {"output": "", "graphs": []}
        with open(script_path, 'r') as file:
            script_code = file.read()
        exec(script_code, globals_dict)

        # Recuperar la salida y los gráficos
        script_output = output_buffer.getvalue()  # Capturar lo que se imprimió
        globals_dict["output"] = script_output  # Almacenar la salida en "output"
        
        graph_list = globals_dict.get("graphs", [])

        return render_template(
            'run_script.html',
            script_name=script_name,
            result=script_output,
            graphs=graph_list
        )
    except Exception as e:
        return f"Error al ejecutar el script: {str(e)}", 500
    finally:
        sys.stdout = sys.__stdout__  # Restaurar stdout original


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

