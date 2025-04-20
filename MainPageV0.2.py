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
from scipy.interpolate import griddata

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
    filepath = filedialog.askopenfilename(filetypes=[("LHCO Files", "*.lhco"), ("CSV Files", "*.csv")])
    if filepath:
        signal_listbox.insert(tk.END, filepath)
#FUNCION PARA SELECCIONAR ARCHIVO BACKGROUND
def add_background_file():
    filepath = filedialog.askopenfilename(filetypes=[("LHCO Files", "*.lhco"), ("CSV Files", "*.csv")])
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
                messagebox.showwarning("Warning", f"Unsupported file type: {i}")
                continue  # Si el archivo no es LHCO o CSV, lo omite

            dfsg = pd.concat([dfsg, data], ignore_index=True)

        # Filtrar eventos con '# == 0'
        if '#' in dfsg.columns:
            mask = dfsg['#'] == 0
            dfsg.loc[mask, dfsg.columns != '#'] = 10.0
            filtered_dfsg = dfsg.copy()
        else:
            messagebox.showwarning("Warning", "The column '#' does not exist in the SIGNAL data")

        # Guardar el CSV
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            filtered_dfsg.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"SIGNAL file saved at: {save_path}")
    else:
        messagebox.showwarning("Warning", "No SIGNAL files were selected.")
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
                messagebox.showwarning("Warning", f"Unsupported file type: {i}")
                continue  # Si el archivo no es LHCO o CSV, lo omite

            dfbg = pd.concat([dfbg, data], ignore_index=True)

        # Filtrar eventos con '# == 0'
        if '#' in dfbg.columns:
            mask = dfbg['#'] == 0
            dfbg.loc[mask, dfbg.columns != '#'] = 10.0
            filtered_dfbg = dfbg.copy()
        else:
            messagebox.showwarning("Warning", "The column '#' does not exist in the BACKGROUND data")

        # Guardar el CSV
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            filtered_dfbg.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"BACKGROUND file saved at: {save_path}")
    else:
        messagebox.showwarning("Warning", "No BACKGROUND files were selected.")

root = tk.Tk()
root.title("Event Analysis Interface")
root.geometry("550x800")

# Create button style
style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 8), padding=3)

# Create tabs
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)
tab5 = ttk.Frame(tab_control)
tab_control.add(tab1, text='File Upload')
tab_control.add(tab2, text='Particles to Analyze')
tab_control.add(tab3, text='Calculation and Analysis')
tab_control.add(tab4, text='Model Training')
tab_control.add(tab5, text='Significance')
tab_control.pack(expand=1, fill='both')

# ----------- Frame: Archivos de Señal ----------- #
frame1_signal = ttk.LabelFrame(tab1, text="SIGNAL Files", padding=10)
frame1_signal.pack(fill="x", padx=10, pady=10)

signal_row = ttk.Frame(frame1_signal)
signal_row.pack(pady=5)

# Listbox + Scrollbar
signal_listbox = tk.Listbox(signal_row, width=40, height=6)
signal_listbox.pack(side="left", padx=(0, 5))

signal_scrollbar = ttk.Scrollbar(signal_row, orient=tk.VERTICAL, command=signal_listbox.yview)
signal_scrollbar.pack(side="left", fill=tk.Y)

signal_listbox.config(yscrollcommand=signal_scrollbar.set)

# Botones para SIGNAL
signal_buttons = ttk.Frame(signal_row)
signal_buttons.pack(side="left", padx=5)

ttk.Button(signal_buttons, text="Browse File", command=select_signal_file).pack(pady=2)
ttk.Button(signal_buttons, text="Remove File", command=remove_selected_signal).pack(pady=2)
ttk.Button(signal_buttons, text="Generate Signal", command=generate_signal_csv).pack(pady=2)

# ----------- Frame: Archivos de Background ----------- #
frame1_background = ttk.LabelFrame(tab1, text="BACKGROUND Files", padding=10)
frame1_background.pack(fill="x", padx=10, pady=10)

background_row = ttk.Frame(frame1_background)
background_row.pack(pady=5)

# Listbox + Scrollbar
background_listbox = tk.Listbox(background_row, width=40, height=6)
background_listbox.pack(side="left", padx=(0, 5))

background_scrollbar = ttk.Scrollbar(background_row, orient=tk.VERTICAL, command=background_listbox.yview)
background_scrollbar.pack(side="left", fill=tk.Y)

background_listbox.config(yscrollcommand=background_scrollbar.set)

# Botones para BACKGROUND
background_buttons = ttk.Frame(background_row)
background_buttons.pack(side="left", padx=5)

ttk.Button(background_buttons, text="Browse File", command=add_background_file).pack(pady=2)
ttk.Button(background_buttons, text="Remove File", command=remove_selected_background).pack(pady=2)
ttk.Button(background_buttons, text="Generate Background", command=generate_background_csv).pack(pady=2)

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
            raise ValueError("Quantity must be greater than 0.")
        lista.append((cantidad, particula))
        lista_box.insert(tk.END, f"{cantidad} {particula}")
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input: {e}")
#FUNCION PARA ELIMINAR PARTICULAS SELECCIONADAS
def remove_selected_particle():
    selected_indices = lista_box.curselection()
    if not selected_indices:
        messagebox.showwarning("Warning", "No particle selected.")
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
        messagebox.showwarning("Warning", "No combination has been selected.")
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

    messagebox.showinfo("Success", "The pair combinations list has been successfully overwritten.")

def overwrite_trios():
    global comb_trios_names, combinaciones_trios
    selected_indices = comb_trios_listbox.curselection()
    
    # Obtener las combinaciones seleccionadas en formato nombres
    selected_names = [comb_trios_names[i] for i in selected_indices]
    if not selected_names:
        messagebox.showwarning("Warning", "No combination has been selected.")
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

    messagebox.showinfo("Success", "The trio combinations list has been successfully overwritten.")

def overwrite_cuartetos():
    global comb_cuartetos_names, combinaciones_cuartetos
    selected_indices = comb_cuartetos_listbox.curselection()
    
    # Obtener las combinaciones seleccionadas en formato nombres
    selected_names = [comb_cuartetos_names[i] for i in selected_indices]
    if not selected_names:
        messagebox.showwarning("Warning", "No combination has been selected.")
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

    messagebox.showinfo("Success", "The quartet combinations list has been successfully overwritten.")

# ----------- Frame: Entrada de partículas ----------- #
frame2_input = ttk.LabelFrame(tab2, text="Add Final State Particles", padding=10)
frame2_input.pack(fill="x", padx=10, pady=10)

ttk.Label(frame2_input, text="Enter the quantity and type of the final state particle you want to analyze:").pack(anchor="w", pady=(0, 5))

row_input = ttk.Frame(frame2_input)
row_input.pack(fill="x", pady=5)

entry_quantity = ttk.Entry(row_input, width=10)
entry_quantity.pack(side="left", padx=(0, 5))

particle_choice = tk.StringVar(value="a")
option_menu = ttk.OptionMenu(row_input, particle_choice, *particulas_dict.keys())
option_menu.pack(side="left", padx=(0, 5))

ttk.Button(row_input, text="Add Particle", command=add_particle).pack(side="left", padx=(0, 5))
ttk.Button(row_input, text="Remove Particle", command=remove_selected_particle).pack(side="left", padx=(0, 5))

# ----------- Frame: Lista de partículas agregadas ----------- #
frame2_lista = ttk.LabelFrame(tab2, text="Particle List", padding=10)
frame2_lista.pack(fill="x", padx=10, pady=10)

lista_frame = ttk.Frame(frame2_lista)
lista_frame.pack(fill="x")

scrollbar_lista = ttk.Scrollbar(lista_frame, orient=tk.VERTICAL)
scrollbar_lista.pack(side="right", fill=tk.Y)

lista_box = tk.Listbox(lista_frame, width=50, height=5, yscrollcommand=scrollbar_lista.set, selectmode=tk.MULTIPLE)
lista_box.pack(side="left", fill=tk.BOTH, expand=True)

scrollbar_lista.config(command=lista_box.yview)

# Texto explicativo
ttk.Label(tab2, text=(
    "The final list of numbered particles has the following format: [(x, y)] where:\n"
    "- x is the type of the particle.\n"
    "- y is its energy position.\n"
), justify="left").pack(padx=10, pady=(5, 10), anchor="w")

# Botón para analizar
ttk.Button(tab2, text="Analyze", command=analyze_particles).pack(pady=(0, 10))

# Segundo texto explicativo
ttk.Label(tab2, text=(
    "Select the tuples, triplets, and quartets of particles that are of interest to you.\n"
    "This will speed up the calculation if you don't want to analyze all of them."
), justify="left").pack(padx=10, pady=(0, 10), anchor="w")

# ----------- Frame: Combinaciones de Pares ----------- #
frame2_pares = ttk.LabelFrame(tab2, text="Pair Combinations", padding=10)
frame2_pares.pack(fill="x", padx=10, pady=5)

pares_frame = ttk.Frame(frame2_pares)
pares_frame.pack(fill="x")

comb_pares_listbox = tk.Listbox(pares_frame, width=50, height=5, selectmode=tk.MULTIPLE)
comb_pares_listbox.pack(side="left")

comb_pares_scrollbar = ttk.Scrollbar(pares_frame, orient=tk.VERTICAL, command=comb_pares_listbox.yview)
comb_pares_scrollbar.pack(side="left", fill=tk.Y)
comb_pares_listbox.config(yscrollcommand=comb_pares_scrollbar.set)

ttk.Button(pares_frame, text="Overwrite List", command=overwrite_pares).pack(side="left", padx=10)

# ----------- Frame: Combinaciones de Tríos ----------- #
frame2_trios = ttk.LabelFrame(tab2, text="Triplet Combinations", padding=10)
frame2_trios.pack(fill="x", padx=10, pady=5)

trios_frame = ttk.Frame(frame2_trios)
trios_frame.pack(fill="x")

comb_trios_listbox = tk.Listbox(trios_frame, width=50, height=5, selectmode=tk.MULTIPLE)
comb_trios_listbox.pack(side="left")

comb_trios_scrollbar = ttk.Scrollbar(trios_frame, orient=tk.VERTICAL, command=comb_trios_listbox.yview)
comb_trios_scrollbar.pack(side="left", fill=tk.Y)
comb_trios_listbox.config(yscrollcommand=comb_trios_scrollbar.set)

ttk.Button(trios_frame, text="Overwrite List", command=overwrite_trios).pack(side="left", padx=10)

# ----------- Frame: Combinaciones de Cuartetos ----------- #
frame2_cuartetos = ttk.LabelFrame(tab2, text="Quartet Combinations", padding=10)
frame2_cuartetos.pack(fill="x", padx=10, pady=5)

cuartetos_frame = ttk.Frame(frame2_cuartetos)
cuartetos_frame.pack(fill="x")

comb_cuartetos_listbox = tk.Listbox(cuartetos_frame, width=50, height=5, selectmode=tk.MULTIPLE)
comb_cuartetos_listbox.pack(side="left")

comb_cuartetos_scrollbar = ttk.Scrollbar(cuartetos_frame, orient=tk.VERTICAL, command=comb_cuartetos_listbox.yview)
comb_cuartetos_scrollbar.pack(side="left", fill=tk.Y)
comb_cuartetos_listbox.config(yscrollcommand=comb_cuartetos_scrollbar.set)

ttk.Button(cuartetos_frame, text="Overwrite List", command=overwrite_cuartetos).pack(side="left", padx=10)

##### Aqui inicia la tercer pestaña

#FUNCION PARA FILTRAR LOS EVENTOS
def filtrar_eventos(df, num_list):
    global progress_bar_f, progress_var_f

    # Mostrar barra al iniciar
    progress_bar_f["value"] = 0
    progress_bar_f.update()

    try:
        event_indices = []
        current_event = []
        current_event_number = None
        num_list_first_elements = [t[0] for t in num_list]
        num_list_first_third_elements = [(t[0], t[2]) for t in num_list]
        total_rows = len(df)

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

            # Actualizar barra de progreso
            progress = ((i + 1) / total_rows) * 100
            progress_var_f.set(progress)
            progress_bar_f.update()

        # Último evento
        if current_event:
            event_typ_counts = [r['typ'] for r in current_event]
            event_typ_ntrk_tuples = [(r['typ'], r['ntrk']) for r in current_event]
            if all(event_typ_counts.count(num) >= num_list_first_elements.count(num) for num in set(num_list_first_elements)):
                if all(event_typ_ntrk_tuples.count(tup) >= num_list_first_third_elements.count(tup) for tup in num_list_first_third_elements if tup[0] in [1, 2]):
                    event_indices.extend(current_event)

        # Ocultar barra al finalizar (opcional)
        progress_var_f.set(0)
        progress_bar_f.update()

        return pd.DataFrame(event_indices)

    except Exception as e:
        messagebox.showerror("Error", f"Error during event filtering: {e}")
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
#DELTA R
def Deltar(evento,comb):
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
    if not prt1.empty and not prt2.empty:
        posicion1 = comb[0][1] - 1
        posicion2 = comb[1][1] - 1
        if posicion1 < len(prt1) and posicion2 < len(prt2):
            eta_prt1 = prt1.iloc[posicion1]['eta']
            eta_prt2 = prt2.iloc[posicion2]['eta']
            phi_prt1 = prt1.iloc[posicion1]['phi']
            phi_prt2 = prt2.iloc[posicion2]['phi']
            
            delta_phi = abs(phi_prt1 - phi_prt2)
            delta_eta = abs(eta_prt1 - eta_prt2)
            
            if delta_phi > math.pi:
                delta_phi -= 2 * math.pi
            elif delta_phi < -math.pi:
                delta_phi += 2 * math.pi
            
            return np.sqrt(delta_eta**2 + delta_phi**2)
        else:
            return 0 
    return 0
#Ratio PT
def RatioPt(evento,comb):
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
    if not prt1.empty and not prt2.empty:
        # Obtener el pt del primer fotón y de la MET
        posicion1=comb[0][1]-1
        posicion2=comb[1][1]-1
        if posicion1 < len(prt1) and posicion2 < len(prt2):
        # Obtener el pt del primer fotón y de la MET
          pt_prt1 = prt1.iloc[posicion1]['pt']
          pt_prt2 = prt2.iloc[posicion2]['pt']
          return pt_prt1/pt_prt2
        else:
          return 0
    return 0
#ProductEta
def ProductEta(evento,comb):
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
    if not prt1.empty and not prt2.empty:
        # Obtener el pt del primer fotón y de la MET
        posicion1=comb[0][1]-1
        posicion2=comb[1][1]-1
        if posicion1 < len(prt1) and posicion2 < len(prt2):
          eta_prt1 = prt1.iloc[posicion1]['eta']
          eta_prt2 = prt2.iloc[posicion2]['eta']
          return eta_prt1*eta_prt2
        else:
          return 0
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
        if posicion < len(prt):
            phi_prt = prt.iloc[posicion]['phi']
            return phi_prt
        else:
            return 0 
    
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
        posicion = listapart[1] - 1
        if posicion < len(prt):
            eta_prt = prt.iloc[posicion]['eta']
            return eta_prt
        else:
            return 0 
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
        if posicion < len(prt):
            pt_prt = prt.iloc[posicion]['pt']
            return pt_prt
        else:
            return 0 
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
        if posicion1 < len(prt1) and posicion2 < len(prt2):
            pt_prt1 = prt1.iloc[posicion1]['pt']
            pt_prt2 = prt2.iloc[posicion2]['pt']
            eta_prt1 = prt1.iloc[posicion1]['eta']
            eta_prt2 = prt2.iloc[posicion2]['eta']
            phi_prt1 = prt1.iloc[posicion1]['phi']
            phi_prt2 = prt2.iloc[posicion2]['phi']
            
            pt1_x, pt1_y, pt1_z = momentum_vector(pt_prt1, phi_prt1, eta_prt1)
            pt2_x, pt2_y, pt2_z = momentum_vector(pt_prt2, phi_prt2, eta_prt2)
            
            m_trans_sqrt = (np.sqrt(pt1_x**2 + pt1_y**2) + np.sqrt(pt2_x**2 + pt2_y**2))**2 - (pt1_x + pt2_x)**2 - (pt1_y + pt2_y)**2
            
            if m_trans_sqrt < 0:
                m_trans_sqrt = 0
            
            m_trans = np.sqrt(m_trans_sqrt)
            return m_trans
        else:
            return 0  # O cualquier valor que consideres apropiado
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
        pt = []
        eta = []
        phi = []
        for p, pos in zip(prt, posiciones):
          if pos < len(p):
                pt.append(p.iloc[pos]['pt'])
                eta.append(p.iloc[pos]['eta'])
                phi.append(p.iloc[pos]['phi'])
          else:
                pt.append(0)
                eta.append(0)
                phi.append(0)
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
    global progress_bar_c, progress_var_c

    masainv = []
    masainv_trios = []
    masainv_cuartetos = []
    masatrans = []
    deltar = []
    no_jets = []
    pt = []
    phi = []
    eta = []
    X_eta = []
    Ratio_pt = []

    total_batches = (len(df) + batch_size - 1) // batch_size
    start = 0
    batch_index = 0

    # Mostrar barra al iniciar
    progress_bar_c["value"] = 0
    progress_bar_c.update()

    while start < len(df):
        end = start + batch_size
        while end < len(df) and df.iloc[end]['#'] != 0:
            end += 1

        batch_df = df.iloc[start:end]
        current_event = []
        current_event_number = None

        for _, row in batch_df.iterrows():
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
                        masatrans.append(m_trans(event_df, i))
                        deltar.append(Deltar(event_df, i))
                        Ratio_pt.append(RatioPt(event_df, i))
                        X_eta.append(ProductEta(event_df, i))
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
                masatrans.append(m_trans(event_df, i))
                deltar.append(Deltar(event_df, i))
                Ratio_pt.append(RatioPt(event_df, i))
                X_eta.append(ProductEta(event_df, i))

        # Actualizar barra de progreso
        batch_index += 1
        progress = (batch_index / total_batches) * 100
        progress_var_c.set(progress)
        progress_bar_c.update()

        start = end

    # Ocultar barra al terminar (opcional)
    progress_var_c.set(0)
    progress_bar_c.update()

    masainv_trios = np.array(masainv_trios)
    if masainv_trios.size > 0:
        a = int(len(masainv_trios) / len(combinaciones_trios))
        masainv_trios = masainv_trios.reshape(a, -1)
    masainv_cuartetos = np.array(masainv_cuartetos)
    if masainv_cuartetos.size > 0:
        a = int(len(masainv_cuartetos) / len(combinaciones_cuartetos))
        masainv_cuartetos = masainv_cuartetos.reshape(a, -1)
    deltar = np.array(deltar)
    if deltar.size > 0:
        a = int(len(deltar) / len(combinaciones_pares))
        deltar = deltar.reshape(a, -1)
    X_eta = np.array(X_eta)
    if X_eta.size > 0:
        a = int(len(X_eta) / len(combinaciones_pares))
        X_eta = X_eta.reshape(a, -1)
    Ratio_pt = np.array(Ratio_pt)
    if Ratio_pt.size > 0:
        a = int(len(Ratio_pt) / len(combinaciones_pares))
        Ratio_pt = Ratio_pt.reshape(a, -1)
    phi = np.array(phi)
    if phi.size > 0:
        a = int(len(phi) / len(lista_num))
        phi = phi.reshape(a, -1)
    eta = np.array(eta)
    if eta.size > 0:
        a = int(len(eta) / len(lista_num))
        eta = eta.reshape(a, -1)
    pt = np.array(pt)
    if pt.size > 0:
        a = int(len(pt) / len(lista_num))
        pt = pt.reshape(a, -1)
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
    columpares3 = []
    columpares4 = []
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
    for i in comb_pares_names:
        cadena = tupla_a_cadena(i)
        columpares3.append('PT1/PT2'+ cadena)
    for i in comb_pares_names:
        cadena = tupla_a_cadena(i)
        columpares4.append('Eta1*Eta2' + cadena)

    csv_columtrios = pd.DataFrame(masainv_trios, columns=columtrios) if masainv_trios.size > 0 else pd.DataFrame()
    csv_columcuartetos = pd.DataFrame(masainv_cuartetos, columns=columcuartetos) if masainv_cuartetos.size > 0 else pd.DataFrame()
    csv_deltar = pd.DataFrame(deltar, columns=columpares2) if deltar.size > 0 else pd.DataFrame()
    csv_pt = pd.DataFrame(pt, columns=colum) if pt.size > 0 else pd.DataFrame()
    csv_eta = pd.DataFrame(eta, columns=colum1) if eta.size > 0 else pd.DataFrame()
    csv_phi = pd.DataFrame(phi, columns=colum2) if phi.size > 0 else pd.DataFrame()
    csv_minv = pd.DataFrame(masainv, columns=columpares) if masainv.size > 0 else pd.DataFrame()
    csv_mtrans = pd.DataFrame(masatrans, columns=columpares1) if masatrans.size > 0 else pd.DataFrame()
    csv_ratiopt = pd.DataFrame(Ratio_pt, columns=columpares3) if Ratio_pt.size > 0 else pd.DataFrame()
    csv_prdeta = pd.DataFrame(X_eta, columns=columpares4) if X_eta.size > 0 else pd.DataFrame()
    # Concatenar solo los DataFrames que no están vacíos
    csv_combined = pd.concat([csv_phi, csv_eta, csv_pt, csv_minv, csv_mtrans, csv_deltar, csv_columtrios, csv_columcuartetos,csv_ratiopt,csv_prdeta], axis=1)
    csv_combined["No_jets"] = no_jets
    return csv_combined

#FUNCION PARA FILTRAR LOS EVENTOS
def on_filtrar_eventos():
    global filtered_dfsg

    # Preguntar al usuario si desea cargar un archivo filtrado en lugar de hacer el filtrado
    choice = messagebox.askyesno("Load SIGNAL file", "Do you want to load a filtered SIGNAL file instead of processing it?")
    
    if choice:  # Si elige "Sí", permite cargar los archivos
        file_sg = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], title="Load filtered SIGNAL")
        
        if file_sg:
            try:
                filtered_dfsg = pd.read_csv(file_sg)
                messagebox.showinfo("Success", "Filtered file loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load the file: {e}")
        return  # Salir de la función si ya se cargaron los archivos

    # Si no se cargan archivos, proceder con el filtrado normal
    if filtered_dfsg is None:
        messagebox.showerror("Error", "The DataFrames have not been loaded. Make sure they have been generated before filtering.")
        return

    filtrar_btn.config(state=tk.DISABLED)  # Deshabilitar botón mientras se filtra
    messagebox.showinfo("Information", "Starting the filtering process...")

    def ejecutar_filtrado():
        global filtered_dfsg

        try:
            file_sg = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save Filtered SIGNAL")
            if not file_sg:
                messagebox.showerror("Error", "You must select names for the filtered file.")
                return

            # Filtrar en bloques
            filtered_dfsg = procesar_en_bloques(filtered_dfsg, lista_num_mod, bloque_tamano=100000)
            filtered_dfsg.to_csv(file_sg, index=False)

            messagebox.showinfo("Success", f"Filtering completed.\nSignal saved at: {file_sg}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while filtering the events: {e}")

        finally:
            filtrar_btn.config(state=tk.NORMAL)  # Reactivar el botón al finalizar

    # Ejecutar en un hilo separado para no bloquear la interfaz
    hilo_filtrado = threading.Thread(target=ejecutar_filtrado)
    hilo_filtrado.start()
#FUNCION PARA INICIAR EL CALCULO
def on_iniciar_calculo():
    global Final_name, columns_to_check, df_combined

    # Preguntar al usuario si desea cargar un archivo de cálculo ya generado
    choice = messagebox.askyesno("Load file", "Do you want to load a calculation file instead of generating it?")
    
    if choice:  # Si elige "Sí", permite cargar el archivo
        Final_name = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], title="Load calculation file")
        
        if Final_name:
            try:
                df_combined = pd.read_csv(Final_name)
                
                # Filtrar solo columnas con datos numéricos
                columns_to_check = df_combined.select_dtypes(include='number').columns.tolist()

                if not columns_to_check:
                    messagebox.showerror("Error", "The file does not contain numeric columns.")
                    return

                columna_menu["values"] = columns_to_check 
                columna_var.set(columns_to_check[0])  # Opcional: seleccionar la primera por defecto
                messagebox.showinfo("Success", "Calculation file loaded successfully.")
                return
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load the file: {e}")
        else:
            messagebox.showwarning("Warning", "No file was selected. The load was canceled.")
        return

    # Si no se carga un archivo, proceder con el cálculo normal
    if filtered_dfsg is None or filtered_dfbg is None:
        messagebox.showerror("Error", "You must filter the events before starting the calculation.")
        return

    calcular_btn.config(state=tk.DISABLED)  # Deshabilitar el botón mientras se ejecuta el cálculo
    messagebox.showinfo("Information", "Starting the calculation process...")

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
        name_sg = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save SIGNAL file")
        if name_sg:
            csv_sig.to_csv(name_sg, index=False)

        # Pedir nombre para guardar el BG
        name_bg = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save BACKGROUND file")
        if name_bg:
            csv_bg.to_csv(name_bg, index=False)

        # Combinar ambos DataFrames
        df_combined = pd.concat([csv_bg, csv_sig], ignore_index=False)
        df_combined.reset_index(drop=True, inplace=True)
        df_combined.index += 1

        # Guardar los nombres de las columnas en columns_to_check
        columns_to_check = df_combined.select_dtypes(include='number').columns.tolist()

        # Refrescar el combobox con las nuevas columnas
        columna_menu["values"] = columns_to_check 
        columna_var.set(columns_to_check[0])  # Opcional: seleccionar la primera por defecto

        # Pedir nombre para guardar el archivo combinado
        Final_name = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save combined file")
        if Final_name:
            df_combined = df_combined.rename_axis('Event').reset_index()
            df_combined.to_csv(Final_name, index=False)
            messagebox.showinfo("Success", f"Combined file saved as:\n{Final_name}")
        else:
            messagebox.showinfo("The combined file was not saved.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during the calculation: {e}")

    finally:
        calcular_btn.config(state=tk.NORMAL)  # Reactivar el botón al finalizar

import plotly.graph_objects as go
from scipy.stats import gaussian_kde, norm, expon

def generar_grafica_v2():
    columna = columna_var.get().strip()
    if not columna or df_combined is None or columna not in df_combined.columns:
        messagebox.showerror("Error", "Check the selected column or file.")
        return

    datos = df_combined[columna].dropna()
    if datos.empty or len(datos) < 2:
        messagebox.showerror("Error", "Not enough data.")
        return

    # Entradas del usuario
    titulo = titulo_var.get().strip() or f"Distribution of {columna}"
    rango_x = rango_x_var.get().strip()
    leyenda = legend_var.get().strip() or "Type"
    bins_usuario = bins_var.get().strip()
    usar_log = log_var.get() if 'log_var' in globals() else False
    ajuste = ajuste_var.get() if 'ajuste_var' in globals() else "None"
    # Nuevas etiquetas de ejes
    etiqueta_x = xlabel_var.get().strip() or columna
    etiqueta_y = ylabel_var.get().strip() or "Events ( scaled to one )"

    try:
        x_range = tuple(map(float, rango_x.split(","))) if rango_x else None
    except:
        x_range = None

    try:
        bins = int(bins_usuario) if bins_usuario else 50
    except:
        bins = 50

    df_signal = df_combined[df_combined["Td"] == "s"]
    df_background = df_combined[df_combined["Td"] == "b"]

    fig = go.Figure()

    # Generar histograma normalizado (área = 1)
    def histo_con_area(df, name, color):
        hist, bin_edges = np.histogram(df[columna].dropna(), bins=bins, range=x_range, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=hist,
            name=name,
            marker_color=color,
            opacity=0.6,
            width=(bin_edges[1] - bin_edges[0]),
            hovertemplate=f"{name}<br>{columna}: %{{x}}<br>Events ( scaled to one ): %{{y:.3f}}<extra></extra>"
        ))
        return df[columna].dropna().values, bin_edges

    datos_b, bin_edges_b = histo_con_area(df_background, "Background", "rgba(0, 123, 255, 0.6)")
    datos_s, bin_edges_s = histo_con_area(df_signal, "Signal", "rgba(220, 53, 69, 0.6)")

    # Curvas de ajuste con misma área que el histograma
    def ajustar_y_graficar(datos, label, color):
        if len(datos) < 2:
            return

        x_vals = np.linspace(min(datos), max(datos), 500)

        if ajuste == "KDE":
            kde = gaussian_kde(datos)
            y_vals = kde(x_vals)

        elif ajuste == "Gaussiano":
            mu, sigma = norm.fit(datos)
            y_vals = norm.pdf(x_vals, mu, sigma)

        elif ajuste == "Exponencial":
            loc, scale = expon.fit(datos)
            y_vals = expon.pdf(x_vals, loc=loc, scale=scale)

        else:
            return

        # Normalizar la curva para que su integral = 1 (igual que histograma con density=True)
        area = np.trapz(y_vals, x_vals)
        y_vals /= area

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            name=f"{ajuste} {label}",
            line=dict(color=color.replace("0.6", "1.0"), dash="dash"),
            hovertemplate=f"{ajuste} {label}<br>{columna}: %{{x}}<br>Events ( scaled to one ): %{{y:.3f}}<extra></extra>"
        ))

    if ajuste != "None":
        ajustar_y_graficar(datos_b, "Background", "rgba(0, 123, 255, 0.6)")
        ajustar_y_graficar(datos_s, "Signal", "rgba(220, 53, 69, 0.6)")

    # Layout
    fig.update_layout(
        title={"text": f"$\\text{{{titulo}}}$", "x": 0.5},
        xaxis_title=f"$\\text{{{etiqueta_x}}}$",
        yaxis_title=f"$\\text{{{etiqueta_y}}}$",
        template="plotly_white",
        legend_title=leyenda,
        legend=dict(font=dict(size=12)),
    )

    if x_range:
        fig.update_xaxes(range=x_range)

    if usar_log:
        fig.update_yaxes(type="log")

    fig.show()

def generar_grafica():
    columna_seleccionada = columna_var.get().strip()

    if not columna_seleccionada:
        messagebox.showerror("Error", "Select a column.")
        return

    if df_combined is None or columna_seleccionada not in df_combined.columns:
        messagebox.showerror("Error", "The combined file was not found or the column does not exist.")
        return

    datos_validos = df_combined[columna_seleccionada].dropna()
    if datos_validos.empty or len(datos_validos) < 2:
        messagebox.showerror("Error", "There is not enough data to generate the histogram.")
        return

    # Valores del usuario
    titulo = titulo_var.get().strip() or f"Distribution of {columna_seleccionada}"
    legend_name = legend_var.get().strip() or "Event Type"
    rango_x_usuario = rango_x_var.get().strip()
    bins_usuario = bins_var.get().strip()  # Nuevo campo: número de bins

    # Rango en X
    try:
        if rango_x_usuario:
            x_min, x_max = map(float, rango_x_usuario.split(","))
            rango_x = (x_min, x_max)
        else:
            rango_x = None
    except ValueError:
        messagebox.showwarning("Warning", "Incorrect format for the X range. Use: min,max")
        rango_x = None

    # Bins
    try:
        bins = int(bins_usuario) if bins_usuario else 50
    except ValueError:
        messagebox.showwarning("Warning", "Bins must be a number. Using default = 50.")
        bins = 50

    # Estilo más profesional
    sns.set(style="whitegrid", palette="muted", font_scale=1.1)

    plt.figure(figsize=(10, 6))

    ax = sns.histplot(
        data=df_combined,
        x=columna_seleccionada,
        hue="Td",
        kde=True,
        bins=bins,
        element="step",
        common_norm=False,
        stat="density"  # Normalizado
    )

    plt.xlabel(columna_seleccionada)
    plt.ylabel("Events ( scaled to one )")
    plt.title(titulo)

    if rango_x:
        plt.xlim(rango_x)

    plt.grid(visible=True, linestyle="--", linewidth=0.5)

    # Leyenda detallada
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label == "s":
            new_labels.append("Signal")
        elif label == "b":
            new_labels.append("Background")
        else:
            new_labels.append(label)
    plt.legend(handles, new_labels, title=legend_name, loc='upper right')

    plt.tight_layout()
    plt.show()

# Entradas para personalización
titulo_var = tk.StringVar()
xlabel_var = tk.StringVar()
ylabel_var = tk.StringVar()
legend_var = tk.StringVar()
rango_x_var = tk.StringVar()
bins_var = tk.StringVar()
log_var = tk.BooleanVar()
ajuste_var = tk.StringVar()

# --- Encabezado principal ---
ttk.Label(tab3, text="Upload or create the SIGNAL files for event filtering and calculation").pack(pady=10)

# --- Frame: Botones principales y barras de progreso ---
frame_main_actions = ttk.LabelFrame(tab3, text="Event Processing", padding=10)
frame_main_actions.pack(pady=10, fill=tk.X)

filtrar_btn = ttk.Button(frame_main_actions, text="Filter Events", command=on_filtrar_eventos)
filtrar_btn.pack(pady=5)

progress_var_f = tk.DoubleVar()
progress_bar_f = ttk.Progressbar(frame_main_actions, variable=progress_var_f, maximum=100, length=400)
progress_bar_f.pack(pady=5)
progress_bar_f["value"] = 0
progress_bar_f.update()

calcular_btn = ttk.Button(frame_main_actions, text="Start Calculation", command=on_iniciar_calculo)
calcular_btn.pack(pady=5)

progress_var_c = tk.DoubleVar()
progress_bar_c = ttk.Progressbar(frame_main_actions, variable=progress_var_c, maximum=100, length=400)
progress_bar_c.pack(pady=5)
progress_bar_c["value"] = 0
progress_bar_c.update()

# --- Frame: Selección de columna ---
frame_columna = ttk.LabelFrame(tab3, text="Select Histogram Column", padding=10)
frame_columna.pack(pady=10, fill=tk.X)

ttk.Label(frame_columna, text="Select the column you want to display as a histogram once\n"
                              "the observables have been calculated:").pack(pady=5)
columna_var = tk.StringVar()
columna_menu = ttk.Combobox(frame_columna, textvariable=columna_var, values=list(columns_to_check), width=80)
columna_menu.pack(pady=5)

# --- Frame: Personalización del gráfico ---
frame_grafico = ttk.LabelFrame(tab3, text="Plot Customization", padding=10)
frame_grafico.pack(pady=10, fill=tk.X)

# Subframe: Título y leyenda
frame_titulos = ttk.Frame(frame_grafico)
frame_titulos.pack(pady=5, fill=tk.X)

ttk.Label(frame_titulos, text="Custom title:").pack(side=tk.LEFT, padx=5)
ttk.Entry(frame_titulos, textvariable=titulo_var, width=30).pack(side=tk.LEFT, padx=5)
ttk.Label(frame_titulos, text="Legend title:").pack(side=tk.LEFT, padx=5)
ttk.Entry(frame_titulos, textvariable=legend_var, width=30).pack(side=tk.LEFT, padx=5)

# Subframe: Ejes
frame_ejes = ttk.Frame(frame_grafico)
frame_ejes.pack(pady=5, fill=tk.X)

ttk.Label(frame_ejes, text="X-axis label:").pack(side=tk.LEFT, padx=5)
ttk.Entry(frame_ejes, textvariable=xlabel_var, width=30).pack(side=tk.LEFT, padx=5)
ttk.Label(frame_ejes, text="Y-axis label:").pack(side=tk.LEFT, padx=5)
ttk.Entry(frame_ejes, textvariable=ylabel_var, width=30).pack(side=tk.LEFT, padx=5)

# Subframe: Rango y Bins
frame_rango_bins = ttk.Frame(frame_grafico)
frame_rango_bins.pack(pady=5, fill=tk.X)

ttk.Label(frame_rango_bins, text="X Range (min,max):").pack(side=tk.LEFT, padx=5)
ttk.Entry(frame_rango_bins, textvariable=rango_x_var, width=25).pack(side=tk.LEFT, padx=5)
ttk.Label(frame_rango_bins, text="Number of bins:").pack(side=tk.LEFT, padx=5)
ttk.Entry(frame_rango_bins, textvariable=bins_var, width=30).pack(side=tk.LEFT, padx=5)

# Subframe: Ajuste y escala
frame_ajuste = ttk.Frame(frame_grafico)
frame_ajuste.pack(pady=5, fill=tk.X)

ttk.Checkbutton(frame_ajuste, text="Logarithmic Scale", variable=log_var).pack(side=tk.LEFT, padx=5)
ttk.Label(frame_ajuste, text="Fit type:").pack(side=tk.LEFT, padx=5)
ttk.Combobox(
    frame_ajuste,
    textvariable=ajuste_var,
    values=["None", "KDE", "Gaussian", "Exponential"],
    state="readonly",
    width=15
).pack(side=tk.LEFT, padx=5)

# --- Botón final para generar gráfica ---
ttk.Button(tab3, text="Generate Plot", command=generar_grafica_v2).pack(pady=10)

#### Pestaña 4 ####

def roc(test_x, test_y, train_x, train_y, model):
    """
    Presenta la curva ROC, que muestra la precisión del clasificador.
    Cuanto más cerca esté el área bajo la curva (AUC) de 1, mejor será el clasificador.
    """
    try:
        # Verificar si las entradas no están vacías
        if test_x is None or test_y is None or train_x is None or train_y is None:
            raise ValueError("Test or training sets must not be null.")

        if model is None:
            raise ValueError("Model is not defined or not trained.")
        
        # Crear la figura para la curva ROC
        plt.figure(figsize=(10, 7))
        plt.title('ROC curve', fontsize=20)

        # Predicción en el conjunto de prueba
        model_predict = model.predict_proba(test_x)  # Obtener probabilidades
        if model_predict.shape[1] < 2:
            raise ValueError("Model must output probabilities for both classes.")
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
        messagebox.showerror("Validation Error", str(ve))
    except FileNotFoundError:
        # Manejo de errores al guardar el archivo
        messagebox.showerror("Error", "Could not save the plot. Check the filename or location.")
    except Exception as e:
        # Manejo de errores generales
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

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
        use_existing = messagebox.askyesno("Existing Data", "Data has already been loaded. Do you want to use the existing data?")

        if use_existing:
            # Crear el texto que se mostrará directamente
            info_content = (f"Data size: {df_shuffled.shape}\n"
                            f"Number of events: {df_shuffled.shape[0]}\n"
                            f"Number of Signal events: {len(df_shuffled[df_shuffled.Td == 's'])}\n"
                            f"Number of Background events: {len(df_shuffled[df_shuffled.Td == 'b'])}\n"
                            f"Signal fraction: {(len(df_shuffled[df_shuffled.Td == 's'])/(float(len(df_shuffled[df_shuffled.Td == 's']) + len(df_shuffled[df_shuffled.Td == 'b'])))*100):.2f}%")
            
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
        messagebox.showinfo("Data Load", "Data loaded successfully.")

        # Crear el texto que se mostrará
        info_content = (f"Data size: {df_shuffled.shape}\n"
                        f"Number of events: {df_shuffled.shape[0]}\n"
                        f"Number of Signal events: {len(df_shuffled[df_shuffled.Td == 's'])}\n"
                        f"Number of Background events: {len(df_shuffled[df_shuffled.Td == 'b'])}\n"
                        f"Signal fraction: {(len(df_shuffled[df_shuffled.Td == 's'])/(float(len(df_shuffled[df_shuffled.Td == 's']) + len(df_shuffled[df_shuffled.Td == 'b'])))*100):.2f}%")
        
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
            messagebox.showerror("Error", f"The entered factor ({factor}) is greater than the number of available Signal events ({signal_count}).")
            return  # Detener ejecución de la función

        if factor > background_count:
            messagebox.showerror("Error", f"The entered factor ({factor}) is greater than the number of available Background events ({background_count}).")
            return  # Detener ejecución de la función

        # Filtrar los eventos con base en el factor
        s_events = df_shuffled[df_shuffled['Td'] == 's'].head(factor)
        b_events = df_shuffled[df_shuffled['Td'] == 'b'].head(factor)
        df_shuffled = pd.concat([s_events, b_events], ignore_index=True)

        # Actualizar la interfaz y mostrar éxito
        info_content = (f"Data size: {df_shuffled.shape}\n"
                        f"Number of events: {df_shuffled.shape[0]}\n"
                        f"Number of Signal events: {len(df_shuffled[df_shuffled.Td == 's'])}\n"
                        f"Number of Background events: {len(df_shuffled[df_shuffled.Td == 'b'])}\n"
                        f"Signal fraction: {(len(df_shuffled[df_shuffled.Td == 's'])/(float(len(df_shuffled[df_shuffled.Td == 's']) + len(df_shuffled[df_shuffled.Td == 'b'])))*100):.2f}%")
        
        # Borrar contenido previo del cuadro de texto
        text_widget.delete('1.0', 'end')  # Borra desde la línea 1 hasta el final
        # Insertar el nuevo texto
        text_widget.insert('1.0', info_content)

        messagebox.showinfo("Success", "Data filtered successfully!")

    except ValueError:
        # Mostrar error si el valor no es válido
        messagebox.showerror("Error", "Please enter a valid integer for the factor.")

def process_data():
    global vars_for_train, df_4train, signal_features, signal_lab, bkgnd_features, bkgnd_labels, features_, label_

    try:
        # Verificar si 'df_shuffled' está cargado
        if df_shuffled is None or df_shuffled.empty:
            messagebox.showerror("Error", "No data loaded to process.")
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
        processed_info = (f"Data processed successfully.\n"
                          f"Total features used:\n {signal_features.shape[1]}\n"
                          f"Total events for training:\n {signal_features.shape[0] + bkgnd_features.shape[0]}\n"
                          f"Variables dropped due to high correlation:\n {to_drop}")
        messagebox.showinfo("Success", "Preprocessing completed")

        # Actualizar contenido en el cuadro de texto (si existe)
        text_widget_2.delete('1.0', 'end')  # Borrar contenido previo
        text_widget_2.insert('1.0', processed_info)

        # Devolver datos procesados
        return signal_features, signal_lab, bkgnd_features, bkgnd_labels, features_, label_

    except Exception as e:
        # Manejo de errores
        messagebox.showerror("Processing Error", f"An error occurred: {e}")

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
        messagebox.showinfo("Export Complete", f"File successfully created as: {myname}")

    except Exception as e:
        # Manejo de errores
        messagebox.showerror("Error", f"An error occurred while exporting the CSV file: {e}")

import webbrowser
import pyperclip  # Asegúrate de tener instalado este paquete: pip install pyperclip

def wandb_login_gui():
    try:
        wandb.login(key=wandb_api_key.get(), relogin=True)

        api = wandb.Api()
        entity = api.default_entity
        project_url = f"https://wandb.ai/{entity}/{nameproyect.get()}"

        # Copiar al portapapeles
        pyperclip.copy(project_url)

        # Mostrar mensaje con opción de abrir el enlace
        msg = (
            "🚀 Login successful!\n\n"
            "🌐 The project URL has been copied to your clipboard.\n\n"
            "🔗 Do you want to open the project page in your browser?"
        )
        open_link = messagebox.askyesno("✅ Logged in to Weights & Biases", msg)

        if open_link:
            webbrowser.open(project_url)

    except Exception as e:
        messagebox.showerror('Error', f"❌ WandB login failed: {e}\n")

import wandb
from wandb.integration.xgboost import WandbCallback as WandbXGBCallback
import xgboost as xgb

def train_and_optimize_model(signal_features, signal_lab, bkgnd_features, bkgnd_labels, size):
    global eval_set, test, train, cols, vars_for_train, modelv1
    global train_feat, train_lab, test_feat, test_lab

    try:
        # 1. División de datos
        train_sig_feat, test_sig_feat, train_sig_lab, test_sig_lab = train_test_split(
            signal_features, signal_lab, test_size=size, random_state=1)
        train_bkg_feat, test_bkg_feat, train_bkg_lab, test_bkg_lab = train_test_split(
            bkgnd_features, bkgnd_labels, test_size=size, random_state=1)

        test_feat = pd.concat([test_sig_feat, test_bkg_feat])
        test_lab = pd.concat([test_sig_lab, test_bkg_lab])
        train_feat = pd.concat([train_sig_feat, train_bkg_feat])
        train_lab = pd.concat([train_sig_lab, train_bkg_lab])

        eval_set = [(train_feat, train_lab), (test_feat, test_lab)]
        test = test_feat.assign(label=test_lab)
        train = train_feat.assign(label=train_lab)
        cols = vars_for_train

        _ks_back = 0
        _ks_sign = 0

        while _ks_back < 0.05 or _ks_sign < 0.05:
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
                model = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist', **params)
                model.fit(train_feat[cols], train_lab)
                preds = model.predict_proba(test_feat[cols])[:, 1]
                return roc_auc_score(test_lab, preds)

            study = optuna.create_study(direction='maximize')
            study.enqueue_trial(manual_params)
            study.optimize(objective, n_trials=50)

            # Mostrar mejores hiperparámetros
            text_widget_3.delete('1.0', 'end')
            text_widget_3.insert('1.0', f"Best hyperparameters:\n{study.best_trial.params}\n\nBest score: {study.best_trial.value}")

            best_hyperparams = study.best_trial.params

            train_feat[cols] = train_feat[cols].fillna(train_feat[cols].mean())
            test_feat[cols] = test_feat[cols].fillna(train_feat[cols].mean())

            eval_set = [(train_feat[cols], train_lab), (test_feat[cols], test_lab)]

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
                early_stopping_rounds=10,
                eval_metric=['logloss', 'auc']
            )

            # ✅ Logging con WandB sin callback
            run = wandb.init(project=nameproyect.get(), name="XGB-run", reinit=True)

            modelv1.fit(
                train_feat[cols], train_lab,
                eval_set=eval_set,
                verbose=True
            )

            results = modelv1.evals_result()
            for epoch in range(len(results["validation_0"]["logloss"])):
                wandb.log({
                    "epoch": epoch,
                    "train-logloss": results["validation_0"]["logloss"][epoch],
                    "test-logloss": results["validation_1"]["logloss"][epoch]
                })

            wandb.finish()

            return modelv1

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")
        return None

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, auc,
    accuracy_score, roc_auc_score, f1_score
)

sns.set_style("whitegrid")

def generate_model_and_visuals(train_feat, train_lab, test_feat, test_lab, modelv1, eval_set):
    global Mymodel, final_csv

    try:
        wandb.init(project=nameproyect.get(), reinit=True)

        modelv1.fit(train_feat[cols], train_lab, eval_set=eval_set, verbose=False)

        # Métricas
        y_pred = modelv1.predict(test_feat[cols])
        y_prob = modelv1.predict_proba(test_feat[cols])[:, 1]
        acc = accuracy_score(test_lab, y_pred)
        f1 = f1_score(test_lab, y_pred)
        roc_score = roc_auc_score(test_lab, y_prob)
        messagebox.showinfo(
            "Model Evaluation Metrics",
            f"✅ Model metrics on test data:\n\n"
            f"• Accuracy: {acc:.4f}\n"
            f"• F1 Score: {f1:.4f}\n"
            f"• ROC AUC: {roc_score:.4f}"
        )

        # Guardar modelo
        model_artifact = wandb.Artifact(
            'xgboost-model',
            type='model',
            metadata={"model_params": modelv1.get_params()}
        )
        model_path = filedialog.asksaveasfilename(defaultextension=".dat",
                                                  filetypes=[("XGBoost Model", "*.dat"), ("All files", "*.*")])
        if model_path:
            with open(model_path, "wb") as f:
                pickle.dump(modelv1, f)
                model_artifact.add_file(model_path)
                wandb.log_artifact(model_artifact)

            messagebox.showinfo("Model Saved", f"Model successfully saved to:\n{model_path}")
            Mymodel = pickle.load(open(model_path, "rb"))

        wandb.finish()

        # Guardar CSV
        final_csv = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save CSV file"
        )
        if not final_csv:
            messagebox.showwarning("Cancelled", "No file was selected.")
            return

        mycsvfile(df_shuffled, final_csv)
        return Mymodel

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while generating the model:\n\n{e}")
        return None

# Función para mostrar la figura seleccionada
def show_selected_plot(plot_name):
    try:
        if Mymodel is None:
            messagebox.showwarning("Model not loaded", "No model is available. Please train or load a model first.")
            return

        if plot_name == "Score Distribution":
            fig, ax = plot_classifier_distributions(Mymodel, test=test, train=train, cols=cols, print_params=False)
            ax.set_title(r"$\text{Classifier Score Distribution}$", fontsize=16)
            ax.set_xlabel(r"$\text{Model Score}$", fontsize=14)
            ax.set_ylabel(r"$\text{Frequency}$", fontsize=14)
            plt.figure(fig.number)
            plt.tight_layout()
            plt.show()

        elif plot_name == "Feature Importance":
            fig, ax = plt.subplots(figsize=(10, 6))
            xgb.plot_importance(Mymodel, ax=ax)
            ax.set_title(r"$\text{Feature Importance (F Score)}$", fontsize=16)
            plt.tight_layout()
            plt.show()

        elif plot_name == "Confusion Matrix":
            cm = confusion_matrix(test_lab, Mymodel.predict(test_feat[cols]))
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            ax.set_title(r"$\text{Confusion Matrix}$", fontsize=16)
            plt.tight_layout()
            plt.show()

        elif plot_name == "Precision-Recall Curve":
            probs = Mymodel.predict_proba(test_feat[cols])[:, 1]
            precision, recall, _ = precision_recall_curve(test_lab, probs)
            pr_auc = auc(recall, precision)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label=fr"$\text{{AUC}} = {pr_auc:.2f}$", color="purple")
            ax.set_xlabel(r"$\text{Recall}$", fontsize=14)
            ax.set_ylabel(r"$\text{Precision}$", fontsize=14)
            ax.set_title(r"$\text{Precision-Recall Curve}$", fontsize=16)
            ax.legend()
            plt.tight_layout()
            plt.show()

        elif plot_name == "Score Histogram":
            probs = Mymodel.predict_proba(test_feat[cols])[:, 1]
            fig, ax = plt.subplots()
            sns.histplot(probs[test_lab == 0], kde=True, color="red", label="Background", stat="density", ax=ax)
            sns.histplot(probs[test_lab == 1], kde=True, color="blue", label="Signal", stat="density", ax=ax)
            ax.set_title(r"$\text{Score Distribution}$", fontsize=16)
            ax.set_xlabel(r"$\text{Model Score}$", fontsize=14)
            ax.legend()
            plt.tight_layout()
            plt.show()

        else:
            messagebox.showerror("Error", "Selected plot is not recognized.")

    except Exception as e:
        messagebox.showerror("Error", f"Could not display the plot:\n\n{e}")

def carga_modelo():
    global modelv1

    # Cargar el modelo
    model_path = filedialog.askopenfilename(filetypes=[("XGBoost Model", "*.json")])
    if model_path:
        modelv1 = xgb.XGBClassifier()
        modelv1.load_model(model_path)
        messagebox.showinfo("Model Loaded", f"Model successfully loaded from: {model_path}")

# Variables
nameproyect = tk.StringVar(value="my-xgb-project")
wandb_api_key = tk.StringVar()
factor_limite = tk.StringVar(value="100")
size = tk.DoubleVar(value=0.8)

# ----------- Frame: Carga de datos ----------- #
frame4_carga = ttk.LabelFrame(tab4, text="1. Load and Visualize Data", padding=10)
frame4_carga.pack(fill="x", padx=10, pady=10)

row_carga = ttk.Frame(frame4_carga)
row_carga.pack(pady=5)

ttk.Label(row_carga, text="Provide the processed file to visualize information about it:").pack(side="left", padx=5)
ttk.Button(row_carga, text="Load Data", command=update_info).pack(side="left", padx=5)
text_widget = tk.Text(frame4_carga, height=5, width=60)
text_widget.pack(pady=5)

# ----------- Frame: Filtrado y Preprocesamiento ----------- #
frame4_filter = ttk.LabelFrame(tab4, text="2. Filter and Process Data", padding=10)
frame4_filter.pack(fill="x", padx=10, pady=10)

ttk.Label(frame4_filter, text="Choose signal/background event size (optional, recommended 50/50):").pack(pady=5)

# Sub-frame horizontal para Entry + Botones en la misma línea
row_filter = ttk.Frame(frame4_filter)
row_filter.pack(pady=5)

ttk.Entry(row_filter, textvariable=factor_limite, width=10).pack(side="left", padx=5)
ttk.Button(row_filter, text="Update Data", command=apply_filter).pack(side="left", padx=5)
ttk.Button(row_filter, text="Process Data", command=process_data).pack(side="left", padx=5)

# Cuadro de texto para mostrar info de salida
text_widget_2 = tk.Text(frame4_filter, height=5, width=60)
text_widget_2.pack(pady=5)

# ----------- Frame: Proporción de entrenamiento ----------- #
frame4_split = ttk.LabelFrame(tab4, text="3. Train-Test Split", padding=10)
frame4_split.pack(fill="x", padx=10, pady=10)

row_split = ttk.Frame(frame4_split)
row_split.pack(pady=5)

ttk.Label(row_split, text="Enter training set proportion (e.g., 0.8):").pack(side="left", padx=5)
ttk.Entry(row_split, textvariable=size, width=10).pack(side="left", padx=5)

# ----------- Frame: WandB Login ----------- #
frame4_wandb = ttk.LabelFrame(tab4, text="4. Weights & Biases Login", padding=10)
frame4_wandb.pack(fill="x", padx=10, pady=10)

row_wandbname = ttk.Frame(frame4_wandb)
row_wandbname.pack(pady=5)

ttk.Label(row_wandbname, text="WandB project name:").pack(side="left", padx=5)
ttk.Entry(row_wandbname, textvariable=nameproyect, width=30).pack(side="left", padx=5)

row_wandb = ttk.Frame(frame4_wandb)
row_wandb.pack(pady=5)

ttk.Label(row_wandb, text="Enter your WandB API key:").pack(side="left", padx=5)
ttk.Entry(row_wandb, textvariable=wandb_api_key, width=30).pack(side="left", padx=5)
ttk.Button(row_wandb, text="Login to WandB", command=wandb_login_gui).pack(side="left", padx=5)

# ----------- Frame: Entrenamiento y optimización ----------- #
frame4_train = ttk.LabelFrame(tab4, text="5. Train and Optimize Model", padding=10)
frame4_train.pack(fill="x", padx=10, pady=10)

row_trmodel = ttk.Frame(frame4_train)
row_trmodel.pack(pady=5)

ttk.Button(row_trmodel, text="Train and Optimize Model",
          command=lambda: train_and_optimize_model(signal_features, signal_lab, bkgnd_features, bkgnd_labels, size.get())
).pack(side="left", padx=5)

ttk.Button(row_trmodel, text="Generate Model and Plots", command=lambda: generate_model_and_visuals(
    train_feat, train_lab, test_feat, test_lab, modelv1, eval_set)).pack(side="left", padx=5)

text_widget_3 = tk.Text(frame4_train, height=5, width=60)
text_widget_3.pack(pady=5)

# ----------- Frame: Visualización de gráficas ----------- #
frame4_graphs = ttk.LabelFrame(tab4, text="6. Visualize Generated Plots", padding=10)
frame4_graphs.pack(fill="x", padx=10, pady=10)

# Lista de nombres de gráficas disponibles
available_plots = [
    "Score Distribution",
    "Feature Importance",
    "Confusion Matrix",
    "Precision-Recall Curve",
    "Score Histogram"
]

# Variable para selección
selected_plot = tk.StringVar(value=available_plots[0])

# Menú desplegable
row_graph_select = ttk.Frame(frame4_graphs)
row_graph_select.pack(pady=5)

ttk.Label(row_graph_select, text="Select a plot to visualize:").pack(side="left", padx=5)
ttk.OptionMenu(row_graph_select, selected_plot, *available_plots).pack(side="left", padx=5)
ttk.Button(row_graph_select, text="Show Plot", command=lambda: show_selected_plot(selected_plot.get())).pack(side="left", padx=5)

# Configuración de la pestaña 5
####

# Variables globales
df_shuffled = None
results_df = None

# Función para cargar y filtrar datos del CSV
def load_csv():
    global df_shuffled
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            factor_val = int(factor_entry.get())
            # Filtrar: tomar los primeros 'factor_val' de cada clase ("s" y "b")
            s_events = df[df['Td'] == "s"].head(factor_val)
            b_events = df[df['Td'] == "b"].head(factor_val)
            df_shuffled = pd.concat([s_events, b_events], ignore_index=True)
            messagebox.showinfo("Successful Load", "Data loaded and filtered successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load or filter the CSV:\n{e}")
    else:
        messagebox.showwarning("Cancelled", "No file was selected.")

# Función para calcular la significancia
def compute_significance(mypdf, xgb_cut, IntLumi, XSs, XSb, factor_val):
    filtered_pdf = mypdf[mypdf['XGB'] > xgb_cut]
    Ns = len(filtered_pdf[filtered_pdf['Td'] == "s"])
    Nb = len(filtered_pdf[filtered_pdf['Td'] == "b"])
    pbTOfb = 1000  # Conversión de pb a fb
    alpha = XSs * pbTOfb * IntLumi / factor_val
    beta  = XSb * pbTOfb * IntLumi / factor_val
    try:
        Sig = (alpha * Ns) / math.sqrt((alpha * Ns) + (beta * Nb))
    except ZeroDivisionError:
        Sig = 0.0
    return Sig

# Función principal para calcular la significancia en un rango de luminosidad y cortes XGB
def calculate_significance_range():
    global results_df, df_shuffled
    try:
        if df_shuffled is None:
            messagebox.showerror("Error", "You must load the CSV file first.")
            return
        
        factor_val = int(factor_entry.get())
        XSs = float(xsignal_entry.get())
        XSb = float(xbackground_entry.get())
        
        # Si no existe la columna 'XGB', se la crea con valores aleatorios
        if 'XGB' not in df_shuffled.columns:
            np.random.seed(0)
            df_shuffled['XGB'] = np.random.rand(len(df_shuffled))
        
        Sig_vals = []
        XGB_vals = []
        Lumi_vals = []

        # Iterar sobre luminosidad y cortes XGB
        for IntLumi in range(300, 3100, 100):  # Luminosidad de 300 a 3000
            xgb_val = 0.5
            for _ in range(100):  # 100 pasos de cortes XGB
                sig_val = compute_significance(df_shuffled, xgb_val, IntLumi, XSs, XSb, factor_val)
                Sig_vals.append(sig_val)
                XGB_vals.append(xgb_val)
                Lumi_vals.append(IntLumi)
                xgb_val += 0.005
            
        results_df = pd.DataFrame({'Luminosity': Lumi_vals, 'XGB Cut': XGB_vals, 'Significance': Sig_vals})
        messagebox.showinfo("Calculation Completed", "Significance calculation completed successfully.")

        # Mostrar máxima significancia
        maxsig = max(Sig_vals)
        max_index = Sig_vals.index(maxsig)
        val_xgb = XGB_vals[max_index]
        val_lumi = Lumi_vals[max_index]
        messagebox.showinfo("Results", f"Maximum significance: {maxsig:.3f}\nXGB Cut: {val_xgb:.3f}\nLuminosity: {val_lumi}")

    except Exception as e:
        messagebox.showerror("Calculation Error", f"An error occurred during the calculation:\n{e}")

    if results_df is None or results_df.empty:
        messagebox.showerror("Error", "There are no results to save. Please calculate the significance first.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        results_df.to_csv(file_path, index=False)
        messagebox.showinfo("Saved", f"Results saved to: {file_path}")

# Función para graficar la significancia
def plot_significance():
    global results_df  # Asegúrate que esto está definido y lleno

    if results_df is None or results_df.empty:
        messagebox.showerror("Error", "No data available to plot.")
        return

    try:
        # Datos originales
        Lumi_vals = results_df['Luminosity'].values
        Sig_vals = results_df['Significance'].values
        XGB_vals = results_df['XGB Cut'].values

        # Crear malla de interpolación
        xi = np.linspace(min(Lumi_vals), max(Lumi_vals), 200)
        yi = np.linspace(min(Sig_vals), max(Sig_vals), 200)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolación de los valores
        zi = griddata((Lumi_vals, Sig_vals), XGB_vals, (xi, yi), method='cubic')

        # Crear figura
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(xi, yi, zi, levels=20, cmap='viridis')
        cbar = plt.colorbar(contour)
        cbar.set_label('XGB Cut', fontsize=14)
        plt.xlabel('Luminosity (fb$^{-1}$)', fontsize=14)
        plt.ylabel('Significance', fontsize=14)
        plt.title('Interpolated Significance Map', fontsize=16)
        plt.grid(True)

        # Mostrar la figura
        plt.tight_layout()
        plt.show()

        # Guardar
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
                                                 title="Save interpolated plot as...")
        if file_path:
            plt.savefig(file_path, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Plot saved to: {file_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate interpolated plot:\n\n{e}")

# ----------- Variables de Tab5 ----------- #
factor_limite_2 = tk.StringVar(value="100")

# ----------- Frame: Carga de CSV y Filtro ----------- #
frame5_carga = ttk.LabelFrame(tab5, text="1. Load and Filter Events", padding=10)
frame5_carga.pack(fill="x", padx=10, pady=10)

row_carga5 = ttk.Frame(frame5_carga)
row_carga5.pack(pady=5)

ttk.Label(row_carga5, text="Event size:").pack(side="left", padx=5)
factor_entry = ttk.Entry(row_carga5, textvariable=factor_limite_2, width=10).pack(side="left", padx=5)

ttk.Label(row_carga5, text="Load and Filter CSV Events").pack(side="left", padx=5)
ttk.Button(row_carga5, text="Load CSV", command=load_csv).pack(side="left", padx=5)

# ----------- Frame: Cross-Sections ----------- #
frame5_xs = ttk.LabelFrame(tab5, text="2. Cross-Sections", padding=10)
frame5_xs.pack(fill="x", padx=10, pady=10)

row_xs = ttk.Frame(frame5_xs)
row_xs.pack(pady=5)

ttk.Label(row_xs, text="XS Signal (pb):").pack(side="left", padx=5)
xsignal_entry = ttk.Entry(row_xs, width=10)
xsignal_entry.pack(side="left", padx=5)

ttk.Label(row_xs, text="XS Background (pb):").pack(side="left", padx=5)
xbackground_entry = ttk.Entry(row_xs, width=10)
xbackground_entry.pack(side="left", padx=5)

# ----------- Frame: Cálculo, Guardado y Gráficas ----------- #
frame5_actions = ttk.LabelFrame(tab5, text="3. Run, Save and Visualize Significance", padding=10)
frame5_actions.pack(fill="x", padx=10, pady=10)

row_actions = ttk.Frame(frame5_actions)
row_actions.pack(pady=5)

ttk.Button(row_actions, text="Calculate Significance", command=calculate_significance_range).pack(side="left", padx=5)
ttk.Button(row_actions, text="Plot Significance", command=plot_significance).pack(side="left", padx=5)

root.mainloop()

