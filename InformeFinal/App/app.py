### Librerias para manejo de imagenes y file system
import base64
import os
from urllib.parse import quote as urlquote
import datetime

### Se cargan las librerias para construir el APP
from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_daq as daq
from dash.dependencies import Input, Output

### Se cargan las librerias a utilizar para los modelos
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import plotly.express as px

### Herramientas de Tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from keras.metrics import mean_absolute_error

### Definicion de error
#loading dataframes
train_df = pd.read_csv('boneage-training-dataset.csv')
test_df = pd.read_csv('boneage-test-dataset.csv')

#appending file extension to id column for both training and testing dataframes
train_df['id'] = train_df['id'].apply(lambda x: str(x)+'.png')
test_df['Case ID'] = test_df['Case ID'].apply(lambda x: str(x)+'.png') 

#standard deviation of boneage
std_bone_age = train_df['boneage'].std()
#mean age is
mean_bone_age = train_df['boneage'].mean()

#models perform better when features are normalised to have zero mean and unity standard deviation
#using z score for the training
train_df['bone_age_z'] = (train_df['boneage'] - mean_bone_age)/(std_bone_age)

### Xception
def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 

### VGG16
boneage_div = 1.0
def mae_months(in_gt, in_pred):
    return mean_absolute_error(boneage_div*in_gt, boneage_div*in_pred)

# Cargar archivos
#loaded_test_X = np.load('dataX.npy')
#loaded_test_Y = np.load('dataY.npy')

### Configuraciones para el procesamiento de imagenes
img_size = 256
test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

### Directorios para guardar las imagenes
UPLOAD_DIRECTORY_TEST = "/project"
UPLOAD_DIRECTORY = "/project/app_uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

### Cargamos los modelos a utilizar
# Convolucion 2D - Keras
load_model = tf.keras.models.load_model('modelConvolution2DKeras.h5')

# Xception
loaded_model = tf.keras.models.load_model("modelo.h5", custom_objects={"mae_in_months": mae_in_months})
loaded_model.load_weights('best_model.h5')

# VGG16
loaded_model_vgg16 = tf.keras.models.load_model("model_vgg6.h5")

### Cargamos la data de los resultados del Validation
# Convolucion 2D - Keras
df1 = pd.read_csv('convolucion2dKeras.csv')
fig1 = px.scatter(df1, x="Edad real", y="Edad predicción")
fig1.update_traces(marker=dict(color='green'))
fig1.add_shape(type="line",
    xref="x", yref="y",
    x0=0, y0=0, x1=280,
    y1=280,
    line=dict(
        color="DarkOrange",
        width=3,
    ),
)

# Porcentaje de Accuracy Convolucion 2D - Keras
contador1 = 0
for i in range(len(df1['Edad predicción'])):
    if abs(df1['Edad predicción'][i] - df1['Edad real'][i]) < 35:
        contador1 = contador1 + 1

accuracy1 = round((contador1 * 100.0)/(len(df1['Edad predicción']) * 1.0), 2)

# Xception
df2 = pd.read_csv('xceptionKeras.csv')
fig2 = px.scatter(df2, x="Edad real", y="Edad predicción")
fig2.add_shape(type="line",
    xref="x", yref="y",
    x0=0, y0=0, x1=280,
    y1=280,
    line=dict(
        color="DarkOrange",
        width=3,
    ),
)

# Porcentaje de Accuracy Xception - Keras
contador2 = 0
for i in range(len(df2['Edad predicción'])):
    if abs(df2['Edad predicción'][i] - df2['Edad real'][i]) < 15:
        contador2 = contador2 + 1

accuracy2 = round((contador2 * 100.0)/(len(df2['Edad predicción']) * 1.0), 2)

# VGG16
df3 = pd.read_csv('predictionVGG16.csv')
fig3 = px.scatter(df3, x="Edad real", y="Edad predicción")
fig3.update_traces(marker=dict(color='red'))
fig3.add_shape(type="line",
    xref="x", yref="y",
    x0=0, y0=0, x1=280,
    y1=280,
    line=dict(
        color="DarkOrange",
        width=3,
    ),
)

# Porcentaje de Accuracy VGG16 - Keras
contador3 = 0
for i in range(len(df3['Edad predicción'])):
    if abs(df3['Edad predicción'][i] - df3['Edad real'][i]) < 35:
        contador3 = contador3 + 1

accuracy3 = round((contador3 * 100.0)/(len(df3['Edad predicción']) * 1.0), 2)


### Libreria para mejores letras y estilos
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=external_stylesheets)

### Ruta de descarga de archivos
@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

### Estructura del app HTML
app.layout = html.Div(
    [
        html.Center(
            children=html.H1("Predicción de Edad Ósea con imágenes de Radiografías"),
        ),
        html.Div(
            children = [
                html.Div(
                    children = html.Div(),
                    style = {
                        "width": "10%",
                    }
                ),
                html.Div(
                    children=[    
                        
                        html.Center(
                            children=[
                                html.H4("Cargar radiografía"),
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        ["Arrastra o click aqui para seleccionar un archivo a cargar."]
                                    ),
                                    style={
                                        "width": "30%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "30px",
                                    },
                                    multiple=True,
                                ),
                            ],            
                        ),
                    ],
                    style = {
                        "width": "80%",
                    }
                ),
                html.Div(
                    children = html.Div(),
                    style = {
                        "width": "10%",
                    }
                ),
            ],
            style = {
                "display": "flex",
            }
        ),
        html.Hr(),
        html.Div(
            children = [
                html.Div(
                    children = html.Div(),
                    style = {
                        "width": "10%",
                    }
                ),
                html.Div(
                    children = [
                        html.H4("Radiografía"),
                        html.Div(id='output-image-upload')
                    ],
                    style={
                        "width": "30%",
                    },  
                ),
                html.Div(
                    children = html.Div(),
                    style = {
                        "width": "10%",
                    }
                ),
                html.Div(
                    children = [
                        html.H4("Resultado"),
                        html.Ul(id="file-list")
                    ],
                    style={
                        "width": "30%",
                    },                    
                )
            ],
            style={
                "display": "flex"
            },
        ),
        html.Hr(),
        html.Div([
            daq.ToggleSwitch(
                id='my-toggle-switch',
                value=False,
                label='Mostrar el rendimiento de los algoritmos',
                labelPosition='top'
            ),
            html.Div(id='toggle-switch-output')
        ]),
        html.Div(
            children=[
                html.Div()
            ],
            style={
                "width": '100%',
                "height": "15px"
            }
        ),
        html.Div(
            children=[
                html.Div(
                    children = html.Div(),
                    style = {
                        "width": "8%",
                    }
                ),
                html.Div(
                    children = [
                        html.Img(
                            src='https://res.cloudinary.com/webuvg/image/upload/f_auto,q_auto,w_330,c_scale,fl_lossy,dpr_2.63/v1538753573/WEB/institucional/logo_uvg_negro.png',
                            style={
                                "width": "35%"
                            },
                        )
                    ],
                    style={
                        "width": "42%"
                    },
                ),
                html.Div(
                    children = [
                        html.Div(
                            'Maria Fernanda Estrada - 14198'
                        ),
                        html.Div(
                            'Rodrigo Samayoa - 17332'
                        ),
                        html.Div(
                            'Christopher Sandoval - 13660'
                        ),
                        html.Div(
                            'David Soto - 17551'
                        ),
                    ],
                    style={
                        "text-align": "right",
                        "width": "42%"
                    },
                ),
                html.Div(
                    children = html.Div(),
                    style = {
                        "width": "8%",
                    }
                ),
            ],
            style={
               "display": "flex" 
            }
        ),
    ],
    style={"max-width": "100%"},
)

### Metodo para guardar un archivo
def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

### Metodo para determinar los archivos que estan cargados
def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

### Metodo para generar links de descarga de archivos
def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

### Llamada Callback para calcular las edades oseas y poner la imagen en pantalla
@app.callback(
    [Output("file-list", "children"), Output("output-image-upload", "children")],
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
### Se actualiza la salida para construir el app
def update_output(uploaded_filenames, uploaded_file_contents):
    ### Podemos regenerar el resultado aqui
    ### Eliminar los archivos en el directorio
    for filename in os.listdir(UPLOAD_DIRECTORY):
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    ### Procesar con los algoritmos
    ### load_model.summary()

    ### Se guarda la imagen y se calcula la edad osea
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    ### Se actualiza el listado de archivos que existen
    files = uploaded_files()

    ### Si hay un archivo se calcula la edad
    if len(files) != 0:
        test_generator = test_data_generator.flow_from_directory(
            directory = UPLOAD_DIRECTORY_TEST,
            shuffle = False,
            class_mode = None,
            color_mode = 'rgb',
            target_size = (img_size,img_size))
        ### Convolucion 2D
        load_predictions = load_model.predict(test_generator)

        ### Xception
        y_pred = loaded_model.predict(test_generator)
        predicted = y_pred.flatten()
        predicted_months = mean_bone_age + std_bone_age*(predicted)

        ### VGG16
        vgg16_pred = loaded_model_vgg16.predict(test_generator)
        vgg16_predicted = vgg16_pred.flatten()
        vgg16_predicted_months = mean_bone_age + std_bone_age/2*(vgg16_predicted[0])

    ### Se revisa que no hayan archivos para poder mostrar la edad Osea
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()

    if len(files) == 0:
        return [html.Li("No hay resultados disponibles aún")], [html.Li("No hay imágenes disponibles aún")]
    else:
        if uploaded_file_contents is not None:
            children = [
                parse_contents(c, n) for c, n in
                zip(uploaded_file_contents, uploaded_filenames)
            ]

        tabla = dash_table.DataTable(
            id='table',
            columns=[
                {
                    'name': 'Modelo',
                    'id': 'Modelo'
                },
                {
                    'name': 'Edad ósea calculada',
                    'id': 'Resultado'
                },
            ],
            data=[
                {
                    'Modelo': 'Convolución 2D - Keras',
                    'Resultado': str(round(load_predictions[0][0],2)) + " meses"
                },
                {
                    'Modelo': 'Xception - Keras',
                    'Resultado': str(round(predicted_months[0],2)) + " meses"
                },
                {
                    'Modelo': 'VGG16 - Keras',
                    'Resultado': str(round(vgg16_predicted_months,2)) + " meses"
                },
            ],
            style_cell_conditional=[
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                } for c in ['Modelo']
            ],
            style_header={
                'backgroundColor': 'rgb(255, 165, 0)',
                'fontWeight': 'bold'
            }
        )

        tablaEspaciada = html.Div(
            children = [
                html.Div(
                    html.Div(),
                    style = {
                        "width": "10%",
                        "height": "30px",
                    }
                ),
                tabla
            ],
        )                       
        return [tablaEspaciada], children

### Parse de imagen y conversion a un objeto HTML
def parse_contents(contents, filename):
    return html.Div([
        html.H6(filename),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Div(
            children=html.Img(
                src=contents,
                style={
                    "width": "100%"
                },
            ),
        ),
    ])

### Llamada para hacer toogle de visibilidad de los resultados de rendimiento
@app.callback(
    dash.dependencies.Output('toggle-switch-output', 'children'),
    [dash.dependencies.Input('my-toggle-switch', 'value')])
### Se actualiza la salida para construir el app
def update_output_2(value):
    texto1 = "Convolución 2D - Keras: " +  str(accuracy1) + "% Accuracy"
    texto2 = "Xception - Keras: " + str(accuracy2) + "% Accuracy"
    texto3 = "VGG16 - Keras: " + str(accuracy3) + "% Accuracy"
    children = html.Div(
        children=[
            html.Div(
                children = [
                    html.Div(
                        children = html.Div(),
                        style = {
                            "width": "10%",
                        }
                    ),
                    html.Div(
                        children = [
                            html.H4("Rendimiento de los algoritmos con datos de prueba"),
                        ],
                        style={
                            "width": "50%",
                        },  
                    ),
                    html.Div(
                        children = html.Div(),
                        style = {
                            "width": "40%",
                        }
                    ),
                ],
                style={
                    "display": "flex"
                },
            ),
            html.Div(
                children = html.Div(),
                style = {
                    "width": "100%",
                    "height": "30px"
                }
            ),
            html.Div(
                children = [
                    html.Div(
                        children = html.Div(),
                        style = {
                            "width": "8%",
                        }
                    ),
                    html.Div(
                        children = [
                            html.Center(
                                html.H5(texto1)
                            ),
                            dcc.Graph(figure=fig1)
                        ],
                        style={
                            "width": "30%",
                        },  
                    ),
                    html.Div(
                        children = [
                            html.Center(
                                html.H5(texto2)
                            ),
                            dcc.Graph(figure=fig2)
                        ],
                        style={
                            "width": "30%",
                        },  
                    ),
                    html.Div(
                        children = [
                            html.Center(
                                html.H5(texto3)
                            ),
                            dcc.Graph(figure=fig3)
                        ],
                        style={
                            "width": "30%",
                        },  
                    ),
                ],
                style={
                    "display": "flex"
                },
            ),
            html.Hr(),
        ]
    )

    if value:
        return children
    return html.Hr()

if __name__ == "__main__":
    app.run_server(debug=True, port=8889)
