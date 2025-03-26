import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
import seaborn as sns
import threading
import mimetypes
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import optuna
import pickle
import math
#import datatable as dt
import pandas as pd
from itertools import combinations
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from matplotlib.patches import Patch, Circle # Correcting the patch error
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

# Variables Globales
filtered_dfbg = None
filtered_dfsg = None
results_df = None
signal_df = None
background_df = None
df_combined = None
df_shuffled = None
columns_to_check = []
lista_num_names = []
lista_num_mod = []
lista_num = []
comb_pares_names = []
comb_trios_names = []
comb_cuartetos_names = []
combinaciones_pares = []
combinaciones_trios = []
combinaciones_cuartetos = []
vars_for_train = [] 

# Agregar tipos de archivos
mimetypes.add_type('lhco', '.lhco')
mimetypes.add_type('csv', '.csv')

#FUNCION PARA SELECCIONAR ARCHIVO SIGNAL
def select_signal_file():
    filepath = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv"), ("Archivos LHCO", "*.lhco")])
    if filepath:
        signal_listbox.insert(tk.END, filepath)
#FUNCION PARA SELECCIONAR ARCHIVO BACKGROUND
def add_background_file():
    filepath = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv"), ("Archivos LHCO", "*.lhco")])
    if filepath:
        background_listbox.insert(tk.END, filepath)
#FUNCION PARA ELIMINAR ARCHIVO SIGNAL SELECCIONADO
def remove_selected_signal():
    selected_indices = signal_listbox.curselection()
    for index in reversed(selected_indices):
        signal_listbox.delete(index)
#FUNCION PARA ELIMINAR ARCHIVO BACKGROUND SELECCIONADO       
def remove_selected_background():
    selected_indices = background_listbox.curselection()
    for index in reversed(selected_indices):
        background_listbox.delete(index)
#FUNCION PARA GENERAR ARCHIVO CSV DE SIGNAL
def generate_signal_csv():
    global filtered_dfsg
    signal_paths = signal_listbox.get(0, tk.END)

    if signal_paths:
        dfsg = pd.DataFrame()
        for i in signal_paths:
            mime_type, encoding = mimetypes.guess_type(i)
            if mime_type == 'lhco':  # Asumiendo que los archivos .lhco son tipo 'text/plain'
                data = pd.read_csv(i, sep=r'\s+')
            elif mime_type == 'csv':
                data = pd.read_csv(i)
            else:
                messagebox.showwarning("Advertencia", f"Tipo de archivo no soportado: {i}")
                continue  # Si el archivo no es LHCO o CSV, lo omite

            dfsg = pd.concat([dfsg, data], ignore_index=True)

        # Filtrar eventos con '# == 0'
        if '#' in dfsg.columns:
            mask = dfsg['#'] == 0
            dfsg.loc[mask, dfsg.columns != '#'] = 10.0
            filtered_dfsg = dfsg.copy()
        else:
            messagebox.showwarning("Advertencia", "La columna '#' no existe en los datos de SIGNAL")

        # Guardar el CSV
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            filtered_dfsg.to_csv(save_path, index=False)
            messagebox.showinfo("Éxito", f"Archivo de SIGNAL guardado en: {save_path}")
    else:
        messagebox.showwarning("Advertencia", "No se seleccionaron archivos de SIGNAL.")
#FUNCION PARA GENERAR ARCHIVO CSV DE BACKGROUND
def generate_background_csv():
    global filtered_dfbg
    background_paths = background_listbox.get(0, tk.END)

    if background_paths:
        dfbg = pd.DataFrame()
        for i in background_paths:
            mime_type, encoding = mimetypes.guess_type(i)
            if mime_type == 'lhco':  # Asumiendo que los archivos .lhco son tipo 'text/plain'
                data = pd.read_csv(i, sep=r'\s+')
            elif mime_type == 'csv':
                data = pd.read_csv(i)
            else:
                messagebox.showwarning("Advertencia", f"Tipo de archivo no soportado: {i}")
                continue  # Si el archivo no es LHCO o CSV, lo omite

            dfbg = pd.concat([dfbg, data], ignore_index=True)

        # Filtrar eventos con '# == 0'
        if '#' in dfbg.columns:
            mask = dfbg['#'] == 0
            dfbg.loc[mask, dfbg.columns != '#'] = 10.0
            filtered_dfbg = dfbg.copy()
        else:
            messagebox.showwarning("Advertencia", "La columna '#' no existe en los datos de BACKGROUND")

        # Guardar el CSV
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            filtered_dfbg.to_csv(save_path, index=False)
            messagebox.showinfo("Éxito", f"Archivo de BACKGROUND guardado en: {save_path}")
    else:
        messagebox.showwarning("Advertencia", "No se seleccionaron archivos de BACKGROUND.")

root = tk.Tk()
root.title("Interfaz de Análisis de Eventos")
root.geometry("550x800")

# Crear estilo de botones
style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 8), padding=3)

# Crear pestañas
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)
tab5 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Carga de archivos')
tab_control.add(tab2, text='Partículas a analizar')
tab_control.add(tab3, text='Cálculo y Análisis')
tab_control.add(tab4, text='Entrenamiento de Modelos')
tab_control.add(tab5, text='Significancia')

tab_control.pack(expand=1, fill='both')

# Contenido de la primera pestaña
signal_label = ttk.Label(tab1, text="Archivos SIGNAL:")
signal_label.pack(pady=5)

signal_frame = ttk.Frame(tab1)
signal_frame.pack(pady=5)

# Crear Listbox y Scrollbar para SIGNAL
signal_listbox = tk.Listbox(signal_frame, width=40, height=6)
signal_listbox.pack(side=tk.LEFT, padx=(0, 5))

signal_scrollbar = ttk.Scrollbar(signal_frame, orient=tk.VERTICAL, command=signal_listbox.yview)
signal_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

signal_listbox.config(yscrollcommand=signal_scrollbar.set)

signal_buttons_frame = ttk.Frame(signal_frame)
signal_buttons_frame.pack(side=tk.LEFT)

signal_button = ttk.Button(signal_buttons_frame, text="Buscar Archivo", command=select_signal_file)
signal_button.pack(pady=2)

remove_button = ttk.Button(signal_buttons_frame, text="Eliminar Archivo", command=remove_selected_signal)
remove_button.pack(pady=2)

signalcsv_button = ttk.Button(signal_buttons_frame, text="Generar Signal", command=generate_signal_csv)
signalcsv_button.pack(pady=2)

background_label = ttk.Label(tab1, text="Archivos BACKGROUND:")
background_label.pack(pady=5)

background_frame = ttk.Frame(tab1)
background_frame.pack(pady=5)

# Crear Listbox y Scrollbar para BACKGROUND
background_listbox = tk.Listbox(background_frame, width=40, height=6)
background_listbox.pack(side=tk.LEFT, padx=(0, 5))

background_scrollbar = ttk.Scrollbar(background_frame, orient=tk.VERTICAL, command=background_listbox.yview)
background_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

background_listbox.config(yscrollcommand=background_scrollbar.set)

background_buttons_frame = ttk.Frame(background_frame)
background_buttons_frame.pack(side=tk.LEFT)

background_button = ttk.Button(background_buttons_frame, text="Buscar Archivo", command=add_background_file)
background_button.pack(pady=2)

remove_button_bg = ttk.Button(background_buttons_frame, text="Eliminar Archivo", command=remove_selected_background)
remove_button_bg.pack(pady=2)

generatecsv_button = ttk.Button(background_buttons_frame, text="Generar Background", command=generate_background_csv)
generatecsv_button.pack(pady=2)

# Contenido de la segunda pestaña

# Diccionarios de partículas
particulas_dict = {
    'a': 0,
    'e-': 1,
    'mu-': 2,
    't': 3,
    'j': 4,
    'MET': 6,
    'e+': 5,
    'mu+': 7,
    'jb':8
}
particulas_dict_inv = {v: k for k, v in particulas_dict.items()}

# Lista para almacenar partículas
lista = []

#FUNCION PARA EXPANDIR Y CAMBIAR TUPLES
def expand_and_swap_tuples(tuples_list):
    expanded_list = []
    for t in tuples_list:
        for i in range(1, t[0] + 1):
            expanded_list.append((t[1], i))
    return expanded_list
#ESTA FUNCION SE UTILIZA PARA LOS NOMBRES
def tupla_a_cadena(tupla):
    if isinstance(tupla, tuple):
        return '(' + ', '.join(tupla_a_cadena(sub) for sub in tupla) + ')'
    else:
        return str(tupla)
#FUNCION PARA DETERMINAR EL VALOR DE X
def determinar_valor(x):
    if x == 6:
        return 0
    elif x in (1, 2):
        return -1
    elif x in (0, 3, 4):
        return 0
    elif x in (5, 7, 8):
        return 1
    else:
        return 0
#FUNCION PARA TRANSFORMAR TUPLES
def transformar_tuplas(tuplas):
    resultado = []
    for t in tuplas:
        particula = particulas_dict_inv[t[0]]
        if t[0] == 6:
            resultado.append((particula,))
        else:
            referencias = ['leading', 'subleading', 'tertiary', 'quaternary', 'quinary']
            referencia = referencias[t[1] - 1] if t[1] > 0 and t[1] <= len(referencias) else ''
            resultado.append((particula, referencia))
    return resultado
#FUNCION PARA AÑADIR PARTICULAS
def add_particle():
    try:
        cantidad = int(entry_quantity.get())
        particula = particle_choice.get()
        if cantidad <= 0:
            raise ValueError("La cantidad debe ser mayor a 0.")
        lista.append((cantidad, particula))
        lista_box.insert(tk.END, f"{cantidad} {particula}")
    except ValueError as e:
        messagebox.showerror("Error", f"Entrada inválida: {e}")
#FUNCION PARA ELIMINAR PARTICULAS SELECCIONADAS
def remove_selected_particle():
    selected_indices = lista_box.curselection()
    if not selected_indices:
        messagebox.showwarning("Advertencia", "No se seleccionó ninguna partícula.")
        return
    for index in reversed(selected_indices):
        lista.pop(index)
        lista_box.delete(index)
#FUNCION PARA ANALIZAR LAS PARTICULAS
def analyze_particles():
    global lista_num, lista_num_mod, lista_num_names, comb_pares_names, comb_trios_names, comb_cuartetos_names
    global combinaciones_pares, combinaciones_trios, combinaciones_cuartetos, comb_pares_listbox, comb_trios_listbox, comb_cuartetos_listbox

    lista_num = [(cantidad, particulas_dict[particula]) for cantidad, particula in lista]
    lista_num.append((1, 6))  # Agregar MET

    lista_num = expand_and_swap_tuples(lista_num)
    lista_num_names = transformar_tuplas(lista_num)
    lista_num = [(x, y, determinar_valor(x)) for (x, y) in lista_num]
    lista_num_mod = [(1 if x == 5 else 2 if x == 7 else x, y, z) for (x, y, z) in lista_num]    

    comb_pares_names = list(combinations(lista_num_names, 2))
    comb_trios_names = list(combinations(lista_num_names, 3))
    comb_cuartetos_names = list(combinations(lista_num_names, 4))

    combinaciones_pares = list(combinations(lista_num_mod, 2))
    combinaciones_trios = list(combinations(lista_num_mod, 3))
    combinaciones_cuartetos = list(combinations(lista_num_mod, 4))

    comb_pares_listbox.delete(0, tk.END)
    for comb in comb_pares_names:
        comb_pares_listbox.insert(tk.END, comb)

    comb_trios_listbox.delete(0, tk.END)
    for comb in comb_trios_names:
        comb_trios_listbox.insert(tk.END, comb)

    comb_cuartetos_listbox.delete(0, tk.END)
    for comb in comb_cuartetos_names:
        comb_cuartetos_listbox.insert(tk.END, comb)

#Función para sobrescribir la lista
def overwrite_pares():
    global comb_pares_names, combinaciones_pares
    selected_indices = comb_pares_listbox.curselection()
    
    # Obtener las combinaciones seleccionadas en formato nombres
    selected_names = [comb_pares_names[i] for i in selected_indices]
    if not selected_names:
        messagebox.showwarning("Advertencia", "No se ha seleccionado ninguna combinación.")
        return
    
    # Mapear las combinaciones seleccionadas de nombres a valores numéricos
    selected_combinations = [
        combinaciones_pares[i] for i in selected_indices
    ]
    
    # Sobrescribir las listas globales
    comb_pares_names = selected_names
    combinaciones_pares = selected_combinations

    # Refrescar el Listbox
    comb_pares_listbox.delete(0, tk.END)
    for name in comb_pares_names:
        comb_pares_listbox.insert(tk.END, name)

    messagebox.showinfo("Éxito", "La lista de combinaciones de pares ha sido sobrescrita correctamente.")

def overwrite_trios():
    global comb_trios_names, combinaciones_trios
    selected_indices = comb_trios_listbox.curselection()
    
    # Obtener las combinaciones seleccionadas en formato nombres
    selected_names = [comb_trios_names[i] for i in selected_indices]
    if not selected_names:
        messagebox.showwarning("Advertencia", "No se ha seleccionado ninguna combinación.")
        return
    
    # Mapear las combinaciones seleccionadas de nombres a valores numéricos
    selected_combinations = [
        combinaciones_trios[i] for i in selected_indices
    ]
    
    # Sobrescribir las listas globales
    comb_trios_names = selected_names
    combinaciones_trios = selected_combinations

    # Refrescar el Listbox
    comb_trios_listbox.delete(0, tk.END)
    for name in comb_trios_names:
        comb_trios_listbox.insert(tk.END, name)

    messagebox.showinfo("Éxito", "La lista de combinaciones de tríos ha sido sobrescrita correctamente.")

def overwrite_cuartetos():
    global comb_cuartetos_names, combinaciones_cuartetos
    selected_indices = comb_cuartetos_listbox.curselection()
    
    # Obtener las combinaciones seleccionadas en formato nombres
    selected_names = [comb_cuartetos_names[i] for i in selected_indices]
    if not selected_names:
        messagebox.showwarning("Advertencia", "No se ha seleccionado ninguna combinación.")
        return
    
    # Mapear las combinaciones seleccionadas de nombres a valores numéricos
    selected_combinations = [
        combinaciones_cuartetos[i] for i in selected_indices
    ]
    
    # Sobrescribir las listas globales
    comb_cuartetos_names = selected_names
    combinaciones_cuartetos = selected_combinations

    # Refrescar el Listbox
    comb_cuartetos_listbox.delete(0, tk.END)
    for name in comb_cuartetos_names:
        comb_cuartetos_listbox.insert(tk.END, name)

    messagebox.showinfo("Éxito", "La lista de combinaciones de cuartetos ha sido sobrescrita correctamente.")

tk.Label(tab2, text="Ingrese la cantidad y tipo de partícula del estado final que deseas analizar:").pack()

frame_input = tk.Frame(tab2)
frame_input.pack()

entry_quantity = tk.Entry(frame_input, width=10)
entry_quantity.pack(side=tk.LEFT)
particle_choice = tk.StringVar()
particle_choice.set("a")
option_menu = tk.OptionMenu(frame_input, particle_choice, *particulas_dict.keys())
option_menu.pack(side=tk.LEFT)

add_button = tk.Button(frame_input, text="Añadir Partícula", command=add_particle)
add_button.pack(side=tk.LEFT)

remove_button = tk.Button(frame_input, text="Eliminar Partícula", command=remove_selected_particle)
remove_button.pack(side=tk.LEFT)

frame_lista_box = tk.Frame(tab2)
frame_lista_box.pack()

scrollbar_lista = tk.Scrollbar(frame_lista_box, orient=tk.VERTICAL)
scrollbar_lista.pack(side=tk.RIGHT, fill=tk.Y)

lista_box = tk.Listbox(frame_lista_box, width=50, height=5, yscrollcommand=scrollbar_lista.set, selectmode=tk.MULTIPLE)
lista_box.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar_lista.config(command=lista_box.yview)

# Formatted explanation text
explanation_text = (
    "La lista final de partículas numeradas tiene el formato: [(x, y)] donde:\n"
    "- x  es el tipo de la partícula.\n"
    "- y  su posición energética.\n"
)
ttk.Label(tab2, text=explanation_text, justify="left").pack(pady=10)

# Botón para analizar
analyze_button = ttk.Button(tab2, text="Analizar", command=analyze_particles)
analyze_button.pack()

explanation_text_2 = (
    "Seleccióne las tuplas, trios y cuartetos de particulas que sean de su interes\n"
    "esto hara más rapido el cálculo si no se desean analizar todas.\n"
)
ttk.Label(tab2, text=explanation_text_2, justify="left").pack(pady=10)

# Crear un frame para las combinaciones de pares
frame_comb_pares = ttk.Frame(tab2, padding=10)
frame_comb_pares.pack(pady=5, fill=tk.X)

# Label para "Pares" centrado
ttk.Label(frame_comb_pares, text="Combinaciones de Pares:", anchor="center").pack(pady=5)

# Listbox, Scrollbar y Botón para "Pares"
listbox_pares_frame = ttk.Frame(frame_comb_pares)
listbox_pares_frame.pack(pady=5)

comb_pares_listbox = tk.Listbox(listbox_pares_frame, width=50, height=5, selectmode=tk.MULTIPLE)
comb_pares_listbox.pack(side=tk.LEFT)

comb_pares_scrollbar = ttk.Scrollbar(listbox_pares_frame, orient=tk.VERTICAL, command=comb_pares_listbox.yview)
comb_pares_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
comb_pares_listbox.config(yscrollcommand=comb_pares_scrollbar.set)

ttk.Button(listbox_pares_frame, text="Sobrescribir Lista", command=overwrite_pares).pack(side=tk.LEFT, padx=10)

# Crear un frame para las combinaciones de tríos
frame_comb_trios = ttk.Frame(tab2, padding=10)
frame_comb_trios.pack(pady=5, fill=tk.X)

# Label para "Tríos" centrado
ttk.Label(frame_comb_trios, text="Combinaciones de Tríos:", anchor="center").pack(pady=5)

# Listbox, Scrollbar y Botón para "Tríos"
listbox_trios_frame = ttk.Frame(frame_comb_trios)
listbox_trios_frame.pack(pady=5)

comb_trios_listbox = tk.Listbox(listbox_trios_frame, width=50, height=5, selectmode=tk.MULTIPLE)
comb_trios_listbox.pack(side=tk.LEFT)

comb_trios_scrollbar = ttk.Scrollbar(listbox_trios_frame, orient=tk.VERTICAL, command=comb_trios_listbox.yview)
comb_trios_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
comb_trios_listbox.config(yscrollcommand=comb_trios_scrollbar.set)

ttk.Button(listbox_trios_frame, text="Sobrescribir Lista", command=overwrite_trios).pack(side=tk.LEFT, padx=10)

# Crear un frame para las combinaciones de cuartetos
frame_comb_cuartetos = ttk.Frame(tab2, padding=10)
frame_comb_cuartetos.pack(pady=5, fill=tk.X)

# Label para "Cuartetos" centrado
ttk.Label(frame_comb_cuartetos, text="Combinaciones de Cuartetos:", anchor="center").pack(pady=5)

# Listbox, Scrollbar y Botón para "Cuartetos"
listbox_cuartetos_frame = ttk.Frame(frame_comb_cuartetos)
listbox_cuartetos_frame.pack(pady=5)

comb_cuartetos_listbox = tk.Listbox(listbox_cuartetos_frame, width=50, height=5, selectmode=tk.MULTIPLE)
comb_cuartetos_listbox.pack(side=tk.LEFT)

comb_cuartetos_scrollbar = ttk.Scrollbar(listbox_cuartetos_frame, orient=tk.VERTICAL, command=comb_cuartetos_listbox.yview)
comb_cuartetos_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
comb_cuartetos_listbox.config(yscrollcommand=comb_cuartetos_scrollbar.set)

ttk.Button(listbox_cuartetos_frame, text="Sobrescribir Lista", command=overwrite_cuartetos).pack(side=tk.LEFT, padx=10)

##### Aqui inicia la tercer pestaña

#FUNCION PARA FILTRAR LOS EVENTOS
def filtrar_eventos(df, num_list):
    try:
        event_indices = []
        current_event = []
        current_event_number = None
        num_list_first_elements = [t[0] for t in num_list]
        num_list_first_third_elements = [(t[0], t[2]) for t in num_list]
        total_rows = len(df)
        with tqdm(total=total_rows, desc="Filtrando eventos") as pbar:
            for i, row in df.iterrows():
                if row['#'] == 0:
                    if current_event:
                        event_typ_counts = [r['typ'] for r in current_event]
                        event_typ_ntrk_tuples = [(r['typ'], r['ntrk']) for r in current_event]
                        if all(event_typ_counts.count(num) >= num_list_first_elements.count(num) for num in set(num_list_first_elements)):
                            if all(event_typ_ntrk_tuples.count(tup) >= num_list_first_third_elements.count(tup) for tup in num_list_first_third_elements if tup[0] in [1, 2]):
                                event_indices.extend(current_event)
                    current_event = []
                    current_event_number = row['#']
                current_event.append(row)
                pbar.update(1)
            if current_event:
                event_typ_counts = [r['typ'] for r in current_event]
                event_typ_ntrk_tuples = [(r['typ'], r['ntrk']) for r in current_event]
                if all(event_typ_counts.count(num) >= num_list_first_elements.count(num) for num in set(num_list_first_elements)):
                    if all(event_typ_ntrk_tuples.count(tup) >= num_list_first_third_elements.count(tup) for tup in num_list_first_third_elements if tup[0] in [1, 2]):
                        event_indices.extend(current_event)
        return pd.DataFrame(event_indices)
    except Exception as e:
        print(f"Error durante el filtrado de eventos: {e}")
        return pd.DataFrame()
#FUNCION PARA PROCESAR LOS EVENTOS EN BLOQUES
def procesar_en_bloques(df, num_list, bloque_tamano=10000):
    total_filas = len(df)
    resultados = []
    inicio = 0

    while inicio < total_filas:
        fin = inicio + bloque_tamano

        # Asegúrate de no cortar eventos
        while fin < total_filas and df.iloc[fin]['#'] != 0:
            fin += 1

        bloque = df.iloc[inicio:fin]
        resultado_bloque = filtrar_eventos(bloque, num_list)
        resultados.append(resultado_bloque)
        inicio = fin

    return pd.concat(resultados, ignore_index=True)

### FUNCIONES PARA CALCULAR LOS EVENTOS

def Num_jets(evento):
    jets = evento[evento['typ'] == 4]
    njets = len(jets)
    return njets
#OBTENCIÓN DEL MOMENTUM VECTOR
def momentum_vector(pt, phi, eta):
    pt_x, pt_y, pt_z = (pt * np.cos(phi)), (pt * np.sin(phi)), pt * np.sinh(eta)
    return pt_x, pt_y, pt_z
#OBTENCIÓN DELTA R
def Deltar(evento,comb):
    prt1 = evento[evento['typ'] == comb[0][0]]
    prt2 = evento[evento['typ']== comb[1][0]]
    if not prt1.empty and not prt2.empty:
        # Obtener el pt del primer fotón y de la MET
        #print(posicion1)
        posicion1=comb[0][1]-1
        posicion2=comb[1][1]-1
        if comb[0][0] in [1, 2]:
            prt1 = prt1[prt1['ntrk'] == comb[0][2]]
        if comb[1][0] in [1, 2]:
            prt2 = prt2[prt2['ntrk'] == comb[1][2]]
        # Condición extra para typ 4
        if comb[0][0] == 4:
            if comb[0][2] == 0:
                prt1 = prt1[prt1['btag'] == comb[0][2]]
            elif comb[0][2] == 1:
                prt1 = prt1[prt1['btag'].isin([1, 2])] 
        if comb[1][0] == 4:
            if comb[1][2] == 0:
                prt2 = prt2[prt2['btag'] == comb[1][2]]
            elif comb[1][2] == 1:
                prt2 = prt2[prt2['btag'].isin([1, 2])]
        #print(prt1)
        eta_prt1 = prt1.iloc[posicion1]['eta']
        eta_prt2 = prt2.iloc[posicion2]['eta']
        phi_prt1 = prt1.iloc[posicion1]['phi']
        phi_prt2 = prt2.iloc[posicion2]['phi']
        return np.sqrt((eta_prt1-eta_prt2)**2 + (phi_prt1-phi_prt2)**2)
    return 0
#OBTENCIÓN PHI
def phi_part(evento, listapart):
    prt = evento[evento['typ'] == listapart[0]]
    
    # Condición extra para typ 1 o 2
    if listapart[0] in [1, 2]:
        prt = prt[prt['ntrk'] == listapart[2]]
    # Condición extra para typ 4
    if listapart[0] == 4:
        if listapart[2] == 0:
            prt = prt[prt['btag'] == listapart[2]]
        elif listapart[2] == 1:
            prt = prt[prt['btag'].isin([1, 2])]
    if not prt.empty:
        posicion = listapart[1] - 1
        phi_prt = prt.iloc[posicion]['phi']
        return phi_prt
    
    return 0
#OBTENCIÓN ETA
def eta_part(evento,listapart):
    prt=evento[evento['typ']==listapart[0]]
    if listapart[0] in [1, 2]:
        prt = prt[prt['ntrk'] == listapart[2]]
        # Condición extra para typ 4
    if listapart[0] == 4:
        if listapart[2] == 0:
            prt = prt[prt['btag'] == listapart[2]]
        elif listapart[2] == 1:
            prt = prt[prt['btag'].isin([1, 2])]
    if not prt.empty:
    	posicion=listapart[1]-1
    	eta_prt = prt.iloc[posicion]['eta']
    	return eta_prt
    return 0
#OBTENCIÓN PT
def pt_part(evento,listapart):
    prt=evento[evento['typ']==listapart[0]]
    if listapart[0] in [1, 2]:
        prt = prt[prt['ntrk'] == listapart[2]]
        # Condición extra para typ 4
    if listapart[0] == 4:
        if listapart[2] == 0:
            prt = prt[prt['btag'] == listapart[2]]
        elif listapart[2] == 1:
            prt = prt[prt['btag'].isin([1, 2])]
    if not prt.empty:
        posicion=listapart[1]-1
        pt_prt = prt.iloc[posicion]['pt']
        return pt_prt
    return 0
#MASA TRANSVERSAL
def m_trans(evento,comb):
    # Filtrar las partículas
    prt1 = evento[evento['typ'] == comb[0][0]]
    prt2 = evento[evento['typ']== comb[1][0]]
    if comb[0][0] in [1, 2]:
        prt1 = prt1[prt1['ntrk'] == comb[0][2]]
    if comb[1][0] in [1, 2]:
        prt2 = prt2[prt2['ntrk'] == comb[1][2]]
        # Condición extra para typ 4
    if comb[0][0] == 4:
            if comb[0][2] == 0:
                prt1 = prt1[prt1['btag'] == comb[0][2]]
            elif comb[0][2] == 1:
                prt1 = prt1[prt1['btag'].isin([1, 2])] 
    if comb[1][0] == 4:
            if comb[1][2] == 0:
                prt2 = prt2[prt2['btag'] == comb[1][2]]
            elif comb[1][2] == 1:
                prt2 = prt2[prt2['btag'].isin([1, 2])]
    # Asegurarse de que hay al menos un fotón y una MET en el evento
    if not prt1.empty and not prt2.empty:
        posicion1=comb[0][1]-1
        posicion2=comb[1][1]-1
        pt_prt1 = prt1.iloc[posicion1]['pt']
        pt_prt2 = prt2.iloc[posicion2]['pt']
        eta_prt1 = prt1.iloc[posicion1]['eta']
        eta_prt2 = prt2.iloc[posicion2]['eta']
        phi_prt1 = prt1.iloc[posicion1]['phi']
        phi_prt2 = prt2.iloc[posicion2]['phi']
        pt1_x,pt1_y,pt1_z=momentum_vector(pt_prt1, phi_prt1,eta_prt1 )
        pt2_x,pt2_y,pt2_z=momentum_vector(pt_prt2, phi_prt2,eta_prt2 )
        m_trans_sqrt=(np.sqrt(pt1_x**2 + pt1_y**2 ) + np.sqrt(pt2_x**2 + pt2_y**2 ))**2 -(pt1_x + pt2_x )**2 - (pt1_y + pt2_y )**2
        if m_trans_sqrt < 0:
            m_trans_sqrt=0
        # print(m_trans)
        m_trans=np.sqrt(m_trans_sqrt)
        return  m_trans
    return 0
#MASA INVARIANTE
def m_inv(evento, comb):
    # Filtrar las partículas
    prt = [evento[evento['typ'] == c[0]] for c in comb]
    
    # Condición extra para typ 1 o 2
    for i, c in enumerate(comb):
        if c[0] in [1, 2]:
            prt[i] = prt[i][prt[i]['ntrk'] == c[2]]
        # Condición extra para typ 4
    for i, c in enumerate(comb):
        if c[0] == 4:
            if c[2] == 0:
                prt[i] = prt[i][prt[i]['btag'] == c[2]]
            elif c[2] == 1:
                prt[i] = prt[i][prt[i]['btag'].isin([1, 2])]    
    
    if all(not p.empty for p in prt):
        posiciones = [c[1] - 1 for c in comb]
        pt = [p.iloc[pos]['pt'] for p, pos in zip(prt, posiciones)]
        eta = [p.iloc[pos]['eta'] for p, pos in zip(prt, posiciones)]
        phi = [p.iloc[pos]['phi'] for p, pos in zip(prt, posiciones)]
        
        momentum = [momentum_vector(pt[i], phi[i], eta[i]) for i in range(len(comb))]
        pt_x, pt_y, pt_z = zip(*momentum)
        
        m_in_squared = (
            (sum(np.sqrt(px**2 + py**2 + pz**2) for px, py, pz in zip(pt_x, pt_y, pt_z)))**2 -
            sum(px for px in pt_x)**2 -
            sum(py for py in pt_y)**2 -
            sum(pz for pz in pt_z)**2
        )
        
        # Verificar si m_in_squared es negativo
        if m_in_squared < 0:
            m_in_squared=0
        
        m_in = np.sqrt(m_in_squared)
        
        return m_in
    return 0
#FUNCION PARA CALCULAR LOS EVENTOS
def calculos_eventos(df, lista_num, combinaciones_pares, combinaciones_trios, combinaciones_cuartetos, batch_size=300):
    masainv = []
    masainv_trios = []
    masainv_cuartetos = []
    masatrans = []
    deltar = []
    no_jets = []
    pt = []
    phi = []
    eta = []

    total_batches = (len(df) + batch_size - 1) // batch_size  # Calcular el número total de lotes
    with tqdm(total=total_batches, desc="Calculando eventos") as pbar:
        start = 0
        while start < len(df):
            end = start + batch_size
            # Ajustar el final del lote para no cortar eventos a la mitad
            while end < len(df) and df.iloc[end]['#'] != 0:
                end += 1

            batch_df = df.iloc[start:end]
            current_event = []
            current_event_number = None

            for i, row in batch_df.iterrows():
                if row['#'] == 0:
                    if current_event:
                        event_df = pd.DataFrame(current_event)
                        no_jets.append(Num_jets(event_df))
                        for i in combinaciones_cuartetos:
                            masainv_cuartetos.append(m_inv(event_df, i))
                        for i in combinaciones_trios:
                            masainv_trios.append(m_inv(event_df, i))
                        for i in lista_num_mod:
                            pt.append(pt_part(event_df, i))
                            eta.append(eta_part(event_df, i))
                            phi.append(phi_part(event_df, i))
                        for i in combinaciones_pares:
                                masainv.append(m_inv(event_df, i))
                        for i in combinaciones_pares:
                            masatrans.append(m_trans(event_df, i))
                        for i in combinaciones_pares:
                            deltar.append(Deltar(event_df, i))
                    current_event = []
                    current_event_number = row['#']
                current_event.append(row)

            if current_event:
                event_df = pd.DataFrame(current_event)
                no_jets.append(Num_jets(event_df))
                for i in combinaciones_cuartetos:
                    masainv_cuartetos.append(m_inv(event_df, i))
                for i in combinaciones_trios:
                    masainv_trios.append(m_inv(event_df, i))
                for i in lista_num_mod:
                    pt.append(pt_part(event_df, i))
                    eta.append(eta_part(event_df, i))
                    phi.append(phi_part(event_df, i))
                for i in combinaciones_pares:
                        masainv.append(m_inv(event_df, i))
                for i in combinaciones_pares:
                        masatrans.append(m_trans(event_df, i))
                for i in combinaciones_pares:
                        deltar.append(Deltar(event_df, i))

            start = end
            pbar.update(1)  # Actualizar la barra de progreso
    masainv_trios = np.array(masainv_trios)
    if masainv_trios.size > 0:
        g = int(len(masainv_trios) / len(combinaciones_trios))
        masainv_trios = masainv_trios.reshape(g, -1)
    masainv_cuartetos = np.array(masainv_cuartetos)
    if masainv_cuartetos.size > 0:
        h = int(len(masainv_cuartetos) / len(combinaciones_cuartetos))
        masainv_cuartetos = masainv_cuartetos.reshape(h, -1)
    deltar = np.array(deltar)
    if deltar.size > 0:
        f = int(len(deltar) / len(combinaciones_pares))
        deltar = deltar.reshape(f, -1)
    phi = np.array(phi)
    if phi.size > 0:
        d = int(len(phi) / len(lista_num))
        phi = phi.reshape(d, -1)
    eta = np.array(eta)
    if eta.size > 0:
        e = int(len(eta) / len(lista_num))
        eta = eta.reshape(e, -1)
    pt = np.array(pt)
    if pt.size > 0:
        c = int(len(pt) / len(lista_num))
        pt = pt.reshape(c, -1)
    masainv = np.array(masainv)
    if masainv.size > 0:
        a = int(len(masainv) / len(combinaciones_pares))
        masainv = masainv.reshape(a, -1)
    masatrans = np.array(masatrans)
    if masatrans.size > 0:
        b = int(len(masatrans) / len(combinaciones_pares))
        masatrans = masatrans.reshape(b, -1)

    columtrios = []
    columcuartetos = []
    columpares = []
    columpares1 = []
    columpares2 = []
    colum = []
    colum1 = []
    colum2 = []
    for i in lista_num_names:
        cadena = tupla_a_cadena(i)
        colum.append('Pt ' + cadena)
        colum1.append('Eta ' + cadena)
        colum2.append('Phi ' + cadena)
    for i in comb_pares_names:
        cadena = tupla_a_cadena(i)
        columpares.append('m_inv ' + cadena)
    for i in comb_pares_names:
        cadena = tupla_a_cadena(i)
        columpares1.append('m_trans ' + cadena)
    for i in comb_pares_names:
        cadena = tupla_a_cadena(i)
        columpares2.append('deltaR ' + cadena)
    for i in comb_trios_names:
        cadena = tupla_a_cadena(i)
        columtrios.append('m_inv ' + cadena)
    for i in comb_cuartetos_names:
        cadena = tupla_a_cadena(i)
        columcuartetos.append('m_inv ' + cadena)

    # Crear DataFrames solo si los arrays no están vacíos
    csv_columtrios = pd.DataFrame(masainv_trios, columns=columtrios) if masainv_trios.size > 0 else pd.DataFrame()
    csv_columcuartetos = pd.DataFrame(masainv_cuartetos, columns=columcuartetos) if masainv_cuartetos.size > 0 else pd.DataFrame()
    csv_deltar = pd.DataFrame(deltar, columns=columpares2) if deltar.size > 0 else pd.DataFrame()
    csv_pt = pd.DataFrame(pt, columns=colum) if pt.size > 0 else pd.DataFrame()
    csv_eta = pd.DataFrame(eta, columns=colum1) if eta.size > 0 else pd.DataFrame()
    csv_phi = pd.DataFrame(phi, columns=colum2) if phi.size > 0 else pd.DataFrame()
    csv_minv = pd.DataFrame(masainv, columns=columpares) if masainv.size > 0 else pd.DataFrame()
    csv_mtrans = pd.DataFrame(masatrans, columns=columpares1) if masatrans.size > 0 else pd.DataFrame()
    # Concatenar solo los DataFrames que no están vacíos
    csv_combined = pd.concat([csv_phi, csv_eta, csv_pt, csv_minv, csv_mtrans, csv_deltar, csv_columtrios, csv_columcuartetos], axis=1)
    csv_combined["No_jets"] = no_jets
    #print(no_jets)
    return csv_combined

#FUNCION PARA FILTRAR LOS EVENTOS
def on_filtrar_eventos():
    global filtered_dfbg, filtered_dfsg

    # Preguntar al usuario si desea cargar un archivo filtrado en lugar de hacer el filtrado
    choice = messagebox.askyesno("Cargar archivo", "¿Deseas cargar un archivo filtrado en lugar de procesarlo?")
    
    if choice:  # Si elige "Sí", permite cargar los archivos
        file_bg = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], title="Cargar BG filtrado")
        file_sg = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], title="Cargar Signal filtrado")
        
        if file_bg and file_sg:
            try:
                filtered_dfbg = pd.read_csv(file_bg)
                filtered_dfsg = pd.read_csv(file_sg)
                messagebox.showinfo("Éxito", "Archivos filtrados cargados correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar los archivos: {e}")
        else:
            messagebox.showwarning("Advertencia", "No se seleccionaron ambos archivos. Se canceló la carga.")
        return  # Salir de la función si ya se cargaron los archivos

    # Si no se cargan archivos, proceder con el filtrado normal
    if filtered_dfbg is None or filtered_dfsg is None:
        messagebox.showerror("Error", "No se han cargado los DataFrames. Asegúrese de haberlos generado antes de filtrar.")
        return

    filtrar_btn.config(state=tk.DISABLED)  # Deshabilitar botón mientras se filtra
    messagebox.showinfo("Información", "Iniciando el proceso de filtrado...")

    def ejecutar_filtrado():
        global filtered_dfbg, filtered_dfsg

        try:
            file_sg = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar Signal filtrado")
            file_bg = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar BG filtrado")

            if not file_bg or not file_sg:
                messagebox.showerror("Error", "Debe seleccionar nombres para los archivos filtrados.")
                return

            # Filtrar en bloques
            filtered_dfbg = procesar_en_bloques(filtered_dfbg, lista_num_mod, bloque_tamano=100000)
            filtered_dfbg.to_csv(file_bg, index=False)

            filtered_dfsg = procesar_en_bloques(filtered_dfsg, lista_num_mod, bloque_tamano=100000)
            filtered_dfsg.to_csv(file_sg, index=False)

            messagebox.showinfo("Éxito", f"Filtrado completado.\nBG guardado en: {file_bg}\nSignal guardado en: {file_sg}")

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un problema al filtrar los eventos: {e}")

        finally:
            filtrar_btn.config(state=tk.NORMAL)  # Reactivar el botón al finalizar

    # Ejecutar en un hilo separado para no bloquear la interfaz
    hilo_filtrado = threading.Thread(target=ejecutar_filtrado)
    hilo_filtrado.start()
#FUNCION PARA INICIAR EL CALCULO
def on_iniciar_calculo():
    global Final_name, columns_to_check, df_combined

    # Preguntar al usuario si desea cargar un archivo de cálculo ya generado
    choice = messagebox.askyesno("Cargar archivo", "¿Deseas cargar un archivo de cálculo en lugar de generarlo?")
    
    if choice:  # Si elige "Sí", permite cargar el archivo
        Final_name = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], title="Cargar archivo de cálculo")
        
        if Final_name:
            try:
                df_combined = pd.read_csv(Final_name)
                columns_to_check = df_combined.columns.tolist()
                columna_menu["values"] = columns_to_check 
                messagebox.showinfo("Éxito", "Archivo de cálculo cargado correctamente.")
                return  # Salir de la función si ya se cargó el archivo
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")
        else:
            messagebox.showwarning("Advertencia", "No se seleccionó un archivo. Se canceló la carga.")
        return  

    # Si no se carga un archivo, proceder con el cálculo normal
    if filtered_dfsg is None or filtered_dfbg is None:
        messagebox.showerror("Error", "Debe filtrar los eventos antes de iniciar el cálculo.")
        return

    calcular_btn.config(state=tk.DISABLED)  # Deshabilitar el botón mientras se ejecuta el cálculo
    messagebox.showinfo("Información", "Iniciando el proceso de cálculo...")

    # Ejecutar el cálculo en un hilo separado
    hilo_calculo = threading.Thread(target=ejecutar_calculo, daemon=True)  # Agregar daemon=True
    hilo_calculo.start()
#FUNCION PARA EJECUTAR EL CALCULO
def ejecutar_calculo():
    global Final_name, columns_to_check, df_combined

    try:
        # Aplicar la función a ambos DataFrames
        csv_sig = calculos_eventos(filtered_dfsg, lista_num, combinaciones_pares, combinaciones_trios, combinaciones_cuartetos)
        csv_sig['Td'] = "s"

        csv_bg = calculos_eventos(filtered_dfbg, lista_num, combinaciones_pares, combinaciones_trios, combinaciones_cuartetos)
        csv_bg['Td'] = "b"

        # Pedir nombre para guardar el Signal
        name_sg = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar archivo de Signal")
        if name_sg:
            csv_sig.to_csv(name_sg, index=False)
            print(f"Se guardó el análisis para el Signal en: {name_sg}")

        # Pedir nombre para guardar el BG
        name_bg = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar archivo de Background")
        if name_bg:
            csv_bg.to_csv(name_bg, index=False)
            print(f"Se guardó el análisis para el BG en: {name_bg}")

        # Combinar ambos DataFrames
        df_combined = pd.concat([csv_bg, csv_sig], ignore_index=False)
        df_combined.reset_index(drop=True, inplace=True)
        df_combined.index += 1

        # Guardar los nombres de las columnas en columns_to_check
        columns_to_check = df_combined.columns.tolist()

        # Refrescar el combobox con las nuevas columnas
        columna_menu["values"] = columns_to_check 

        # Pedir nombre para guardar el archivo combinado
        Final_name = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar archivo combinado")
        if Final_name:
            df_combined = df_combined.rename_axis('Evento').reset_index()
            df_combined.to_csv(Final_name, index=False)
            print(f'El archivo combinado fue creado con el nombre: {Final_name}')
            messagebox.showinfo("Éxito", f"Archivo combinado guardado como:\n{Final_name}")
        else:
            print("No se guardó el archivo combinado.")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un problema en el cálculo: {e}")

    finally:
        calcular_btn.config(state=tk.NORMAL)  # Reactivar el botón al finalizar
#FUNCION PARA GENERAR LAS GRAFICAS
def generar_grafica():
    columna_seleccionada = columna_var.get().strip()
    
    if not columna_seleccionada:
        messagebox.showerror("Error", "Seleccione una columna.")
        return
    
    if df_combined is None or columna_seleccionada not in df_combined.columns:
        messagebox.showerror("Error", "No se encontró el archivo combinado o la columna no existe.")
        return

    # Filtrar valores nulos y verificar que la columna tenga más de un dato
    datos_validos = df_combined[columna_seleccionada].dropna()
    if datos_validos.empty or len(datos_validos) < 2:
        messagebox.showerror("Error", "No hay suficientes datos para generar el histograma.")
        return

    # Verificar si la columna es numérica
    if not pd.api.types.is_numeric_dtype(datos_validos):
        messagebox.showerror("Error", "La columna seleccionada no es numérica.")
        return

    # Obtener valores ingresados por el usuario (si existen)
    titulo_usuario = titulo_var.get().strip()
    rango_x_usuario = rango_x_var.get().strip()
    legend_usuario = legend_var.get().strip()  # Nombre de la leyenda

    # Determinar el título
    titulo = titulo_usuario if titulo_usuario else f"Distribución de {columna_seleccionada}"
    
    # Determinar el rango del eje X (si el usuario lo ingresó correctamente)
    try:
        if rango_x_usuario:
            x_min, x_max = map(float, rango_x_usuario.split(","))  # Convertir a dos números
            rango_x = (x_min, x_max)
        else:
            rango_x = None  # Mantener el valor predeterminado
    except ValueError:
        messagebox.showwarning("Advertencia", "Formato incorrecto para el rango en X. Use: min,max")
        rango_x = None  # Ignorar el rango si el formato no es válido

    # Determinar el nombre de la leyenda
    nombre_legend = legend_usuario if legend_usuario else "Td"

    # Crear histograma
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df_combined, x=columna_seleccionada, hue="Td", kde=True, bins=50, element="step", common_norm=False)

    plt.xlabel(columna_seleccionada)
    plt.ylabel("Número de Eventos")
    plt.title(titulo)

    # Aplicar el rango en X si fue definido correctamente
    if rango_x:
        plt.xlim(rango_x)

    plt.grid()

    # Modificar el nombre de la leyenda
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title=nombre_legend)  # Usa el nombre personalizado

    plt.show()

####

tk.Label(tab3, text="Sube o crea los archivos para el filtrado y cálculo de eventos").pack()

columna_var = tk.StringVar()
titulo_var = tk.StringVar()
rango_x_var = tk.StringVar()
legend_var = tk.StringVar()

#Botones y entradas pestaña 3
filtrar_btn = tk.Button(tab3, text="Filtrar Eventos", command=on_filtrar_eventos)
filtrar_btn.pack(pady=10)

calcular_btn = tk.Button(tab3, text="Iniciar Cálculo", command=on_iniciar_calculo)
calcular_btn.pack(pady=10)

# Selección de columna
tk.Label(tab3, text="Seleccione la columna que desea visualizar como histograma una vez\n" 
         " terminado el cálculo de las observables para visualizar las opciones:").pack()
columna_var = tk.StringVar()
columna_menu = ttk.Combobox(tab3, textvariable=columna_var, values=list(columns_to_check), width=80)
columna_menu.pack(pady=10)

# Crear etiquetas y entradas en la ventana

tk.Label(tab3, text="Título del Gráfico (Opcional):").pack(pady=4)
tk.Entry(tab3, textvariable=titulo_var).pack(pady=4)

tk.Label(tab3, text="Rango en X (Opcional, formato min,max):").pack(pady=4)
tk.Entry(tab3, textvariable=rango_x_var).pack(pady=4)

tk.Label(tab3, text="Nombre de la Leyenda (Opcional):").pack(pady=4)
tk.Entry(tab3, textvariable=legend_var).pack(pady=4)

# Botón para generar gráfica
btn_generar = tk.Button(tab3, text="Generar Histograma", command=generar_grafica)
btn_generar.pack(pady=10)

#### Pestaña 4 ####

def roc(test_x, test_y, train_x, train_y, model):
    """
    Presenta la curva ROC, que muestra la precisión del clasificador.
    Cuanto más cerca esté el área bajo la curva (AUC) de 1, mejor será el clasificador.
    """
    try:
        # Verificar si las entradas no están vacías
        if test_x is None or test_y is None or train_x is None or train_y is None:
            raise ValueError("Los conjuntos de prueba o entrenamiento no deben ser nulos.")

        if model is None:
            raise ValueError("El modelo no está definido o no está entrenado.")
        
        # Crear la figura para la curva ROC
        plt.figure(figsize=(10, 7))
        plt.title('ROC curve', fontsize=20)

        # Predicción en el conjunto de prueba
        model_predict = model.predict_proba(test_x)  # Obtener probabilidades
        if model_predict.shape[1] < 2:
            raise ValueError("El modelo debe generar probabilidades para ambas clases.")
        model_predict = model_predict[:, 1]
        auc_score = roc_auc_score(test_y, model_predict)  # Calcular AUC
        fpr, tpr, _ = roc_curve(test_y, model_predict)  # Calcular la curva ROC
        plt.plot(tpr, 1 - fpr, label=f'Test   {round(auc_score, 4)}', color='firebrick', linewidth=2)

        # Predicción en el conjunto de entrenamiento
        model_predict = model.predict_proba(train_x)
        model_predict = model_predict[:, 1]
        auc_score = roc_auc_score(train_y, model_predict)
        fpr, tpr, _ = roc_curve(train_y, model_predict)
        plt.plot(tpr, 1 - fpr, label=f'Train   {round(auc_score, 4)}', color='midnightblue', linewidth=2)
        print('Train : ', auc_score)

        # Estética del gráfico
        plt.legend(loc='best', fontsize=15)
        plt.ylabel('Purity', fontsize=15)
        plt.xlabel('Efficiency', fontsize=15)
        plt.ylim(0.0, 1.1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    except ValueError as ve:
        # Manejo de errores de validación
        messagebox.showerror("Error de Validación", str(ve))
    except FileNotFoundError:
        # Manejo de errores al guardar el archivo
        messagebox.showerror("Error", "No se pudo guardar la gráfica. Verifica el nombre o la ubicación.")
    except Exception as e:
        # Manejo de errores generales
        messagebox.showerror("Error", f"Ocurrió un error inesperado: {e}")

def plot_classifier_distributions(model, test, train, cols, print_params=False, params=None):
    global _ks_back
    global _ks_sign
    test_background = model.predict_proba(test.query('label==0')[cols])[:,1]
    test_signal     = model.predict_proba(test.query('label==1')[cols])[:,1]
    train_background= model.predict_proba(train.query('label==0')[cols])[:,1]
    train_signal    = model.predict_proba(train.query('label==1')[cols])[:,1]

    test_pred = model.predict_proba(test[cols])[:,1]
    train_pred= model.predict_proba(train[cols])[:,1]

    density = True
    fig, ax = plt.subplots(figsize=(10, 7))
    background_color = 'firebrick'
    opts = dict(
        range=[0,1],
        bins = 50,#previously its value was 25
        density = density
    )
    histtype1 = dict(
        histtype='step',
        linewidth=1,
        alpha=1,
    )

    ax.hist(train_background, **opts, **histtype1,
             facecolor=background_color,
             edgecolor=background_color,
             zorder=0)
    ax.hist(train_signal, **opts, **histtype1,
             facecolor='blue',
             edgecolor='blue',
             zorder=1000)

    hist_test_0 = np.histogram(test_background, **opts)
    hist_test_1 = np.histogram(test_signal, **opts)
    bins_mean = (hist_test_0[1][1:]+hist_test_0[1][:-1])/2
    bin_width = bins_mean[1]-bins_mean[0]
    area0 = bin_width*np.sum(test.label==0)
    area1 = bin_width*np.sum(test.label==1)

    opts2 = dict(
          capsize=3,
          ls='none',
          marker='P'
    )
   

    ax.errorbar(bins_mean, hist_test_0[0],  yerr = np.sqrt(hist_test_0[0]/area0), xerr=bin_width/2,
                 color=background_color, **opts2, zorder=100)
    ax.errorbar(bins_mean, hist_test_1[0],  yerr = np.sqrt(hist_test_1[0]/area1), xerr=bin_width/2,
                 color='midnightblue', **opts2, zorder=10000)

    # Aplicar el test KS
    ##statistic, p_value = ks_2samp(data1, data2)
    _ks_back = ks_2samp(train_background, test_background)[1]
    _ks_sign = ks_2samp(train_signal, test_signal)[1]

    auc_test  = roc_auc_score(test.label,test_pred )
    auc_train = roc_auc_score(train.label,train_pred)

    legend_elements = [Patch(facecolor='black', edgecolor='black', alpha=0.4,
                             label=f'Train : {round(auc_train,8)}'),
                      Line2D([0], [0], marker='|', color='black',
                             label=f'Test : {round(auc_test,8)}',
                              markersize=25, linewidth=1),
                       Circle((0.5, 0.5), radius=2, color='red',
                              label=f'Background (ks-pval) : {round(_ks_back,8)}',),
                       Circle((0.5, 0.5), 0.01, color='blue',
                              label=f'Signal (ks-pval) : {round(_ks_sign,8)}',),
                       ]

    ax.legend(
              #title='KS test',
              handles=legend_elements,
              #bbox_to_anchor=(0., 1.02, 1., .102),
              loc='upper center',
              ncol=2,
              #mode="expand",
              #borderaxespad=0.,
              frameon=True,
              fontsize=15)

    if print_params:
        ax.text(1.02, 1.02, params_to_string(model),
        transform=ax.transAxes,
      fontsize=13, ha='left', va='top')

    ax.set_yscale('log')
    ax.set_xlabel('BDT prediction',fontsize=20)
    plt.ylabel('Events',fontsize=20);
    ax.set_ylim(0.01, 100)
    plt.xticks(fontsize = 15); 
    plt.yticks(fontsize = 15); 

    #plt.savefig(os.path.join(dir_, 'LR_overtrain.pdf'), bbox_inches='tight')
    return fig, ax

def update_info():
    global df_shuffled

    # Verificar si 'df_shuffled' existe y no es None ni está vacío
    if df_shuffled is not None and not df_shuffled.empty:
        # Preguntar al usuario si quiere usar los datos existentes
        use_existing = messagebox.askyesno("Datos Existentes", "Ya existen datos cargados. ¿Deseas usar estos datos?")

        if use_existing:
            # Crear el texto que se mostrará directamente
            info_content = (f"Tamaño de los datos: {df_shuffled.shape}\n"
                            f"Número de eventos: {df_shuffled.shape[0]}\n"
                            f"Número de eventos de Signal: {len(df_shuffled[df_shuffled.Td == 's'])}\n"
                            f"Número de eventos de Background: {len(df_shuffled[df_shuffled.Td == 'b'])}\n"
                            f"Fracción de señal: {(len(df_shuffled[df_shuffled.Td == 's'])/(float(len(df_shuffled[df_shuffled.Td == 's']) + len(df_shuffled[df_shuffled.Td == 'b'])))*100):.2f}%")
            
            # Borrar contenido previo del cuadro de texto
            text_widget.delete('1.0', 'end')  # Borra desde la línea 1 hasta el final
            # Insertar el nuevo texto
            text_widget.insert('1.0', info_content)
            return  # Finalizar la función si el usuario decide usar los datos existentes

    # Si el usuario eligió no usar los datos existentes o no hay datos, proceder con la carga de datos
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        column_titles = df.iloc[0]
        df_shuffled = df.iloc[1:].sample(frac=1).reset_index(drop=True)
        df_shuffled.loc[-1] = column_titles
        df_shuffled.index = df_shuffled.index + 1
        df_shuffled = df_shuffled.sort_index()
        messagebox.showinfo("Carga de Datos", "Datos cargados exitosamente")

        # Crear el texto que se mostrará
        info_content = (f"Tamaño de los datos: {df_shuffled.shape}\n"
                        f"Número de eventos: {df_shuffled.shape[0]}\n"
                        f"Número de eventos de Signal: {len(df_shuffled[df_shuffled.Td == 's'])}\n"
                        f"Número de eventos de Background: {len(df_shuffled[df_shuffled.Td == 'b'])}\n"
                        f"Fracción de señal: {(len(df_shuffled[df_shuffled.Td == 's'])/(float(len(df_shuffled[df_shuffled.Td == 's']) + len(df_shuffled[df_shuffled.Td == 'b'])))*100):.2f}%")
        
        # Borrar contenido previo del cuadro de texto
        text_widget.delete('1.0', 'end')  # Borra desde la línea 1 hasta el final
        # Insertar el nuevo texto
        text_widget.insert('1.0', info_content)

def apply_filter():
    global df_shuffled

    try:
        # Convertir el valor ingresado en un número entero
        factor = int(factor_limite.get())

        # Obtener las cantidades de eventos de Signal y Background
        signal_count = len(df_shuffled[df_shuffled['Td'] == 's'])
        background_count = len(df_shuffled[df_shuffled['Td'] == 'b'])

        # Validar que el factor no sea mayor que los eventos disponibles
        if factor > signal_count:
            messagebox.showerror("Error", f"El factor ingresado ({factor}) es mayor al número de eventos de Signal disponibles ({signal_count}).")
            return  # Detener ejecución de la función

        if factor > background_count:
            messagebox.showerror("Error", f"El factor ingresado ({factor}) es mayor al número de eventos de Background disponibles ({background_count}).")
            return  # Detener ejecución de la función

        # Filtrar los eventos con base en el factor
        s_events = df_shuffled[df_shuffled['Td'] == 's'].head(factor)
        b_events = df_shuffled[df_shuffled['Td'] == 'b'].head(factor)
        df_shuffled = pd.concat([s_events, b_events], ignore_index=True)

        # Actualizar la interfaz y mostrar éxito
        info_content = (f"Tamaño de los datos: {df_shuffled.shape}\n"
                        f"Número de eventos: {df_shuffled.shape[0]}\n"
                        f"Número de eventos de Signal: {len(df_shuffled[df_shuffled.Td == 's'])}\n"
                        f"Número de eventos de Background: {len(df_shuffled[df_shuffled.Td == 'b'])}\n"
                        f"Fracción de señal: {(len(df_shuffled[df_shuffled.Td == 's'])/(float(len(df_shuffled[df_shuffled.Td == 's']) + len(df_shuffled[df_shuffled.Td == 'b'])))*100):.2f}%")
        
        # Borrar contenido previo del cuadro de texto
        text_widget.delete('1.0', 'end')  # Borra desde la línea 1 hasta el final
        # Insertar el nuevo texto
        text_widget.insert('1.0', info_content)

        messagebox.showinfo("Éxito", "¡Datos filtrados correctamente!")

    except ValueError:
        # Mostrar error si el valor no es válido
        messagebox.showerror("Error", "Por favor, ingrese un número entero válido para el factor.")

def process_data():
    global vars_for_train, df_4train, signal_features, signal_lab, bkgnd_features, bkgnd_labels, features_, label_

    try:
        # Verificar si 'df_shuffled' está cargado
        if df_shuffled is None or df_shuffled.empty:
            messagebox.showerror("Error", "No hay datos cargados para procesar.")
            return

        # Separar signal y background
        Sig_df = (df_shuffled[df_shuffled.Td == 's'])
        Bkg_df = (df_shuffled[df_shuffled.Td == 'b'])

        # Variables para entrenamiento
        vars_for_train = list(Sig_df.columns)  # Convertir en lista
        vars_for_train = [var for var in vars_for_train if var not in ["Td", "Evento"]]
        data4label = df_shuffled[vars_for_train]

        # Preparar subconjuntos
        signal4train = Sig_df[vars_for_train]
        bkg4train = Bkg_df[vars_for_train]

        # Calcular matriz de correlación
        correlations = signal4train.corr()

        # Seleccionar variables con alta correlación
        upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        to_drop_filtered = [column for column in to_drop if column in signal4train.columns]

        # Eliminar columnas correlacionadas
        signal4train = signal4train.drop(to_drop_filtered, axis=1)
        bkg4train = bkg4train.drop(to_drop_filtered, axis=1)

        # Actualizar vars_for_train eliminando las columnas correlacionadas
        vars_for_train = [var for var in vars_for_train if var not in to_drop]

        # Balancear conjuntos
        n_samples = min(signal4train.shape[0], bkg4train.shape[0])
        bkg4train = bkg4train.sample(n=n_samples, random_state=42)
        signal4train = signal4train.sample(n=n_samples, random_state=42)

        # Agregar etiquetas
        bkg4train = bkg4train.copy()
        signal4train = signal4train.copy()
        bkg4train.loc[:, 'signal/bkgnd'] = 0
        signal4train.loc[:, 'signal/bkgnd'] = 1

        # Concatenar los DataFrames
        df_4train = pd.concat([signal4train, bkg4train])

        # GENERAL DATA
        # Separar las etiquetas y las características
        features_ = df_4train.drop(['signal/bkgnd'], axis=1)  # train_x features = all minus (signal/bkgnd and masses)
        label_ = df_4train['signal/bkgnd']  # train_Y

        # Crear subconjuntos específicos para signal y background
        signal_features = signal4train.drop(['signal/bkgnd'], axis=1)  # signal_x
        signal_lab = signal4train['signal/bkgnd']  # signal_y

        bkgnd_features = bkg4train.drop(['signal/bkgnd'], axis=1)  # bkgnd_x
        bkgnd_labels = bkg4train['signal/bkgnd']  # bkgnd_y

        # Mensaje al usuario
        processed_info = (f"Datos procesados correctamente.\n"
                          f"Total de características usadas:\n {signal_features.shape[1]}\n"
                          f"Total de eventos para entrenamiento:\n {signal_features.shape[0] + bkgnd_features.shape[0]}\n"
                          f"Variables eliminadas por alta correlación:\n {to_drop}")
        messagebox.showinfo("Exito","Preprocesamiento Completado")

        # Actualizar contenido en el cuadro de texto (si existe)
        text_widget_2.delete('1.0', 'end')  # Borrar contenido previo
        text_widget_2.insert('1.0', processed_info)

        # Devolver datos procesados
        return signal_features, signal_lab, bkgnd_features, bkgnd_labels, features_, label_

    except Exception as e:
        # Manejo de errores
        messagebox.showerror("Error en el procesamiento", f"Ocurrió un error: {e}")

def train_and_optimize_model(signal_features, signal_lab, bkgnd_features, bkgnd_labels, size):
    global eval_set, test, train, cols, vars_for_train, modelv1
    global train_feat, train_lab, test_feat, test_lab

    try:
        # Dividir los datos en conjuntos de entrenamiento y prueba
        train_sig_feat, test_sig_feat, train_sig_lab, test_sig_lab = train_test_split(
            signal_features, signal_lab, test_size=size, random_state=1)
        train_bkg_feat, test_bkg_feat, train_bkg_lab, test_bkg_lab = train_test_split(
            bkgnd_features, bkgnd_labels, test_size=size, random_state=1)

        # Combinar los datos de prueba y entrenamiento
        test_feat = pd.concat([test_sig_feat, test_bkg_feat])  # test_x
        test_lab = pd.concat([test_sig_lab, test_bkg_lab])    # test_y
        train_feat = pd.concat([train_sig_feat, train_bkg_feat])  # train_x
        train_lab = pd.concat([train_sig_lab, train_bkg_lab])    # train_y

        # Crear conjuntos para evaluación
        eval_set = [(train_feat, train_lab), (test_feat, test_lab)]
        test = test_feat.assign(label=test_lab)
        train = train_feat.assign(label=train_lab)
        cols = vars_for_train

        # Variables auxiliares para hiperparámetros
        _ks_back = 0
        _ks_sign = 0

        while _ks_back < 0.05 or _ks_sign < 0.05:
            # Parámetros manuales iniciales
            manual_params = {
                'colsample_bylevel': 0.8129556523950925,
                'colsample_bynode': 0.6312324405171867,
                'colsample_bytree': 0.6479261529614907,
                'gamma': 6.0528983610080305,
                'learning_rate': 0.1438821307939924,
                'max_leaves': 15,               
                'max_depth': 5,
                'min_child_weight': 1.385895334160164,
                'reg_alpha': 6.454459356576733,
                'reg_lambda': 22.88928659795952,
                'n_estimators': 100
            }

            # Función de optimización usando Optuna
            def objective(trial):
                params = {
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.9),
                    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 0.7),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.7),
                    'gamma': trial.suggest_float('gamma', 5.5, 7),
                    'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.2),
                    'max_leaves': trial.suggest_int('max_leaves', 10, 20),
                    'max_depth': trial.suggest_int('max_depth', 4, 6),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 2.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 6, 7),
                    'reg_lambda': trial.suggest_float('reg_lambda', 22, 23),
                    'n_estimators': trial.suggest_int('n_estimators', 90, 120)
                }
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    tree_method='hist',
                    **params
                )
                model.fit(train_feat[cols], train_lab)
                preds = model.predict_proba(test_feat[cols])[:, 1]
                return roc_auc_score(test_lab, preds)

            # Optimización de hiperparámetros con Optuna
            study = optuna.create_study(direction='maximize')
            study.enqueue_trial(manual_params)
            study.optimize(objective, n_trials=50)

            # Resultados de los mejores hiperparámetros
            text_widget_3.delete('1.0', 'end')  # Borrar contenido previo
            text_widget_3.insert('1.0', f"Mejores hiperparámetros:\n{study.best_trial.params}\n\nMejor puntaje: {study.best_trial.value}")

            best_hyperparams = study.best_trial.params

            # Llenar valores nulos con la media del conjunto de entrenamiento
            train_feat[cols] = train_feat[cols].fillna(train_feat[cols].mean())
            test_feat[cols] = test_feat[cols].fillna(train_feat[cols].mean())

            # Crear conjunto de evaluación consistente
            eval_set = [(train_feat[cols], train_lab), (test_feat[cols], test_lab)]

            # Crear modelo con los mejores hiperparámetros
            modelv1 = xgb.XGBClassifier(
                objective='binary:logistic',
                tree_method='hist',
                n_jobs=5,
                max_leaves=best_hyperparams['max_leaves'],
                max_depth=best_hyperparams['max_depth'],
                learning_rate=best_hyperparams['learning_rate'],
                reg_alpha=best_hyperparams['reg_alpha'],
                reg_lambda=best_hyperparams['reg_lambda'],
                min_child_weight=best_hyperparams['min_child_weight'],
                colsample_bylevel=best_hyperparams['colsample_bylevel'],
                colsample_bynode=best_hyperparams['colsample_bynode'],
                colsample_bytree=best_hyperparams['colsample_bytree'],
                gamma=best_hyperparams['gamma'],
                n_estimators=best_hyperparams['n_estimators'],
                early_stopping_rounds=10
            )

            # Entrenar el modelo con los mejores hiperparámetros
            modelv1.fit(train_feat[cols], train_lab, eval_set=eval_set, verbose=True)
            return modelv1

    except Exception as e:
        # Manejo de errores
        messagebox.showerror("Error", f"Ocurrió un error durante el entrenamiento: {e}")
        return None

def carga_modelo():
    global modelv1

    # Cargar el modelo
    model_path = filedialog.askopenfilename(filetypes=[("Modelo XGBoost", "*.json")])
    if model_path:
        modelv1 = xgb.XGBClassifier()
        modelv1.load_model(model_path)
        messagebox.showinfo("Modelo Cargado", f"Modelo cargado exitosamente desde: {model_path}")

def generate_model_and_visuals(train_feat, train_lab, test_feat, test_lab, modelv1, eval_set):
    global Mymodel

    try:
        # Ajustar el modelo con el conjunto de evaluación y sin verbose
        modelv1.fit(train_feat[cols], train_lab, eval_set=eval_set, verbose=False)

        # Generar y guardar distribución del clasificador
        fig, ax = plot_classifier_distributions(modelv1, test=test, train=train, cols=cols, print_params=False)
        # Establecer título (opcional)
        # ax.set_title(r'Total sample size $\approx$ '+str(len(train) + len(test))+' optimized')

        # Mostrar métricas KS-pval si es relevante
        messagebox.showinfo("Metricas", f'Background(Ks-pval): {_ks_back}\n'
                            f'Signal(Ks-pval): {_ks_sign}')

        # Guardar la gráfica
        plt.show()

        # Generar y mostrar ROC
        roc(test_feat[cols], test_lab, train_feat[cols], train_lab, modelv1)

        # Mostrar gráfico de importancia de características
        messagebox.showinfo("Información","Se mostrará ahora una gráfica que organiza las variables por orden de importancia (F Score)")
        xgb.plot_importance(modelv1)
        plt.show()

        # Guardar el modelo entrenado
        model_path = filedialog.asksaveasfilename(defaultextension=".dat",
                                                  filetypes=[("Modelo XGBoost", "*.dat"), ("Todos los archivos", "*.*")])
        if model_path:
            pickle.dump(modelv1, open(model_path, "wb"))
            messagebox.showinfo("Modelo Guardado", f"Modelo guardado exitosamente en: {model_path}")

        # Cargar modelo para predecir
        Mymodel = pickle.load(open(model_path, "rb"))
        return Mymodel

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al generar el modelo o las gráficas: {e}")
        return None

def mycsvfile(pdf, myname):
    """
    Agregar predicciones del modelo y exportar CSV usando Dask.
    """
    try:
        # Convertir el DataFrame de pandas a Dask DataFrame
        ddf = dd.from_pandas(pdf, npartitions=4)  # Divide los datos en 4 particiones (ajustar según el tamaño del dataset)

        # Seleccionar las columnas necesarias para el modelo
        selected_vars = cols  # 'cols' contiene los nombres de las columnas necesarias
        datalabel = ddf[selected_vars].compute()  # Computar las particiones para obtener un pandas.DataFrame

        # Generar predicciones usando el modelo
        predict = Mymodel.predict_proba(datalabel)  # Usar el modelo previamente entrenado
        pdf['XGB'] = predict[:, 1]  # Agregar la nueva columna con las probabilidades

        # Convertir nuevamente a Dask DataFrame
        ddf = dd.from_pandas(pdf, npartitions=4)  # Útil si tienes datos grandes

        # Guardar el archivo como CSV
        ddf.to_csv(myname, single_file=True, index=False)  # Asegura que el archivo final esté en un solo CSV

        # Mensaje de éxito
        messagebox.showinfo("Exportación Completa", f"Archivo creado como: {myname}")

    except Exception as e:
        # Manejo de errores
        messagebox.showerror("Error", f"Ocurrió un error al exportar el archivo CSV: {e}")

def export_csv_gui():
    global final_csv
    # Se solicita al usuario que seleccione la ubicación y el nombre para guardar el CSV.
    final_csv = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
        title="Guardar archivo CSV"
    )
    if not final_csv:
        messagebox.showwarning("Cancelado", "No se seleccionó ningún archivo.")
        return

    # Llama a la función para exportar el CSV con el DataFrame df_shuffled
    mycsvfile(df_shuffled, final_csv)

factor_limite = tk.StringVar(value="100")
size = 0.8

tk.Label(tab4, text="Proporciona el archivo calculado para visulizar en el\n"
                    "siguiente cuadro de texto información acerca de él:").pack(pady=5)
tk.Button(tab4, text="Cargar Datos", command=update_info).pack()

# Configuración inicial del cuadro de texto
text_widget = tk.Text(tab4, height=5, width=40)
text_widget.pack(pady=5)

tk.Label(tab4, text="Elige el tamaño limite para los eventos signal/background:\n"
"ya que se recomienda sea del 50/50 (Opcional)").pack(pady=5)
tk.Entry(tab4, textvariable=factor_limite).pack(pady=5)
tk.Button(tab4, text="Actualizar Datos", command=apply_filter).pack()

tk.Button(tab4, text="Procesar Datos", command=process_data).pack(pady=5)

text_widget_2 = tk.Text(tab4, height=5, width=50)
text_widget_2.pack(pady=5)

tk.Label(tab4, text="Coloque la poporción de datos que desea usar\n"
                    "para el entrenamiento (se recomienda 0.8):").pack(pady=5)
tk.Entry(tab4, textvariable=size).pack(pady=10)

# Etiqueta en la parte superior
tk.Label(tab4, text="Entrena y Optimiza tu modelo o carga tu Modelo Entrenado:").pack(pady=5)

# Crear un frame para los botones
button_frame = tk.Frame(tab4)
button_frame.pack(pady=5)  # Empaquetar el frame

# Botón de Entrenar y Optimizar
tk.Button(button_frame, text="Entrenar y Optimizar Modelo", command=lambda: train_and_optimize_model(signal_features, signal_lab, bkgnd_features, bkgnd_labels, size)).pack(side="left", padx=5)

# Botón de Cargar Modelo
#tk.Button(button_frame, text="Cargar Modelo", command=carga_modelo).pack(side="left", padx=5)

text_widget_3 = tk.Text(tab4, height=5, width=40)
text_widget_3.pack(pady=5)

tk.Label(tab4, text="Genera el modelo y exporta los resultados:").pack(pady=5)

# Botón para generar el modelo y las visualizaciones
tk.Button(tab4, text="Generar Modelo y Gráficas", command=lambda: generate_model_and_visuals(
    train_feat, train_lab, test_feat, test_lab, modelv1, eval_set)).pack(pady=5)

# Botón para exportar CSV con las predicciones del modelo

tk.Label(tab4, text="Añadir la columna de clasificacion XGB al Df:").pack(pady=5)
tk.Button(tab4, text="Generar CSV", command=export_csv_gui).pack(pady=5)

# Configuración de la pestaña 5
####
tk.Label(tab5, text="Significancia Estadística").pack()

# Variables globales para almacenar los datos y resultados
df_shuffled = None
results_df = None
cols = None  # Se llenará al cargar el CSV

# Función para cargar y filtrar datos del CSV
def load_csv():
    global df_shuffled, cols
    file_path = filedialog.askopenfilename(title="Seleccione el archivo CSV", filetypes=[("Archivos CSV", "*.csv")])
    
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # Se asumen todas las columnas, excepto 'Td' y 'Evento' (puedes ajustarlo según necesites)
            cols = [c for c in df.columns if c not in ["Td", "Evento"]]
            # Obtener el factor desde la interfaz (convertido a entero)
            factor_val = int(factor_entry.get())
            # Filtrar: tomar los primeros 'factor_val' de cada clase ("s" y "b")
            s_events = df[df['Td'] == "s"].head(factor_val)
            b_events = df[df['Td'] == "b"].head(factor_val)
            df_shuffled = pd.concat([s_events, b_events], ignore_index=True)
            messagebox.showinfo("Carga exitosa", "Datos cargados y filtrados correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar o filtrar el CSV:\n{e}")
    else:
        messagebox.showwarning("Cancelado", "No se seleccionó archivo.")

# Función para calcular la significancia en un corte XGB específico
def compute_significance(mypdf, xgb_cut, XSs, XSb, factor_val):
    filtered_pdf = mypdf[mypdf['XGB'] > xgb_cut]
    Ns = len(filtered_pdf[filtered_pdf['Td'] == "s"])
    Nb = len(filtered_pdf[filtered_pdf['Td'] == "b"])
    pbTOfb = 1000  # Conversión de pb a fb
    IntLumi = 3000 # Luminosidad integrada
    alpha = XSs * pbTOfb * IntLumi / factor_val
    beta  = XSb * pbTOfb * IntLumi / factor_val
    try:
        Sig = (alpha * Ns) / math.sqrt((alpha * Ns) + (beta * Nb))
    except ZeroDivisionError:
        Sig = 0.0
    return Sig

# Función para calcular la significancia para un rango de cortes XGB
def calculate_significance_range():
    global results_df, df_shuffled
    try:
        if df_shuffled is None:
            messagebox.showerror("Error", "Primero debe cargar el archivo CSV.")
            return
        
        factor_val = int(factor_entry.get())
        XSs = float(xsignal_entry.get())
        XSb = float(xbackground_entry.get())
        
        # Si no existe la columna 'XGB', se la crea con valores aleatorios (como ejemplo)
        if 'XGB' not in df_shuffled.columns:
            np.random.seed(0)
            df_shuffled['XGB'] = np.random.rand(len(df_shuffled))
        
        Sig_vals = []
        XGB_vals = []
        xgb_val = 0.5
        for _ in range(499):
            sig_val = compute_significance(df_shuffled, xgb_val, XSs, XSb, factor_val)
            Sig_vals.append(sig_val)
            XGB_vals.append(xgb_val)
            xgb_val += 0.001
            
        results_df = pd.DataFrame({'XGB_cut': XGB_vals, 'Significance': Sig_vals})
        messagebox.showinfo("Cálculo Completado", "El cálculo de la significancia se realizó correctamente.")
    except Exception as e:
        messagebox.showerror("Error en Cálculo", f"Ocurrió un error durante el cálculo:\n{e}")

# Función para guardar los resultados en un CSV
def save_results():
    global results_df
    if results_df is None or results_df.empty:
        messagebox.showerror("Error", "No hay resultados para guardar. Calcule la significancia primero.")
        return
    file_name = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("Archivos CSV", "*.csv")],
        title="Guardar resultados de significancia"
    )
    if file_name:
        results_df.to_csv(file_name, index=False)
        messagebox.showinfo("Guardado", f"Resultados guardados en: {file_name}")
    else:
        messagebox.showwarning("Cancelado", "No se seleccionó archivo para guardar.")

# Función para mostrar la fila con la mayor significancia
def show_best_significance():
    global results_df
    if results_df is None or results_df.empty:
        messagebox.showerror("Error", "No hay datos de significancia para analizar. Calcule la significancia primero.")
        return
    best_row = results_df.loc[results_df['Significance'].idxmax()]
    best_tuple = tuple(best_row)
    messagebox.showinfo("Mejor Significancia", f"La mejor fila es:\n{best_tuple}")


# Botón para cargar CSV
frame_cargar = tk.Frame(tab5)
frame_cargar.pack(pady=5)

# Agregar el Label y el Button dentro de frame_cargar, ambos con side="left"
tk.Label(frame_cargar, text="Cargar CSV y aplicar filtros").pack(side="left", padx=5)
tk.Button(frame_cargar, text="Cargar CSV", command=load_csv).pack(side="left", padx=5)


# Entrada para Factor (cantidad de eventos)
frame_factor = tk.Frame(tab5)
frame_factor.pack(pady=5)
tk.Label(frame_factor, text="Factor (cantidad de eventos):").pack(side="left")
factor_entry = tk.Entry(frame_factor)
factor_entry.pack(side="left", padx=5)

# Entrada para XS Señal
frame_signal = tk.Frame(tab5)
frame_signal.pack(pady=5)
tk.Label(frame_signal, text="XS Señal (pb):").pack(side="left")
xsignal_entry = tk.Entry(frame_signal)
xsignal_entry.pack(side="left", padx=5)

# Entrada para XS Background
frame_background = tk.Frame(tab5)
frame_background.pack(pady=5)
tk.Label(frame_background, text="XS Background (pb):").pack(side="left")
xbackground_entry = tk.Entry(frame_background)
xbackground_entry.pack(side="left", padx=5)

# Botón para calcular la significancia
tk.Button(tab5, text="Calcular Significancia", command=calculate_significance_range).pack(pady=5)

# Botón para guardar los resultados
tk.Button(tab5, text="Guardar Resultados", command=save_results).pack(pady=5)

# Botón para mostrar la fila con la mejor significancia
tk.Button(tab5, text="Mostrar Mejor Significancia", command=show_best_significance).pack(pady=5)

root.mainloop()

