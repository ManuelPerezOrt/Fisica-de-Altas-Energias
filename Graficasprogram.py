import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
results_df = pd.read_csv('datos_totalh_bbbb.csv')

# Filtrar los datos por señal y fondo
signal_df = results_df[results_df['Td'] == 's']
background_df = results_df[results_df['Td'] == 'b']

# Definir las columnas a verificar (excluyendo la primera y la última columna)
columns_to_check = results_df.columns[1:-1]

# Función para crear histogramas y guardar la gráfica
def crear_histograma(df_signal, df_background, columna, xlabel, ylabel, title, filename=None):
    plt.hist(df_signal[columna], weights=df_signal['Evento'], bins=50, edgecolor='black', alpha=0.5, label='Signal', density=True)
    plt.hist(df_background[columna], weights=df_background['Evento'], bins=50, edgecolor='black', alpha=0.5, label='Background', density=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper right')
    if filename:
        plt.savefig(filename)
    plt.show()

while True:
    # Imprimir los nombres de las columnas
    print("Nombres de las columnas disponibles:")
    for columna in columns_to_check:
        print(columna)

    # Preguntar al usuario cuál gráfica quiere
    columna_seleccionada = input("Por favor, ingrese el nombre de la columna que desea graficar (o 'salir' para terminar): ")
    
    if columna_seleccionada.lower() == 'salir':
        break

    # Crear el histograma para la columna seleccionada sin límite y sin guardar
    crear_histograma(signal_df, background_df, columna_seleccionada, columna_seleccionada, 'Número de Eventos', f'{columna_seleccionada} vs Número de Eventos')

    # Preguntar al usuario si desea imponer un límite y guardar la gráfica
    imponer_limite = input("¿Desea imponer un límite y guardar la gráfica? (sí/no): ")
    
    if imponer_limite.lower() == 'sí':
        # Preguntar al usuario el límite para la columna seleccionada
        limite = float(input(f"Por favor, ingrese el límite para la columna {columna_seleccionada}: "))

        # Aplicar el filtro del límite a ambas tablas
        signal_df_filtrado = signal_df[signal_df[columna_seleccionada] <= limite]
        background_df_filtrado = background_df[background_df[columna_seleccionada] <= limite]

        # Preguntar al usuario el nombre del archivo para guardar la gráfica
        nombre_archivo = input("Por favor, ingrese el nombre del archivo para guardar la gráfica: ")
        nombre_archivo = nombre_archivo + ".jpg"

        # Crear el histograma para la columna seleccionada con el límite y guardar la gráfica
        crear_histograma(signal_df_filtrado, background_df_filtrado, columna_seleccionada, columna_seleccionada, 'Número de Eventos', f'{columna_seleccionada} vs Número de Eventos', nombre_archivo)

print("Ahora iniciara el entrenamiento con BDT")