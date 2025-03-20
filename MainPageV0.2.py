import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
import seaborn as sns
import threading
import mimetypes
import pandas as pd
from itertools import combinations
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Variables Globales
filtered_dfbg = None
filtered_dfsg = None
Final_names = []
results_df = None
signal_df = None
background_df = None
df_combined = None
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


# Agregar tipos de archivos
mimetypes.add_type('lhco', '.lhco')
mimetypes.add_type('csv', '.csv')

def select_signal_file():
    filepath = filedialog.askopenfilename(filetypes=[("LHCO or CSV", "*.lhco;*.csv")])
    if filepath:
        signal_listbox.insert(tk.END, filepath)

def add_background_file():
    filepath = filedialog.askopenfilename(filetypes=[("LHCO or CSV", "*.lhco;*.csv")])
    if filepath:
        background_listbox.insert(tk.END, filepath)

def remove_selected_signal():
    selected_indices = signal_listbox.curselection()
    for index in reversed(selected_indices):
        signal_listbox.delete(index)
        
def remove_selected_background():
    selected_indices = background_listbox.curselection()
    for index in reversed(selected_indices):
        background_listbox.delete(index)

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
root.geometry("500x400")

# Crear estilo de botones
style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 8), padding=3)

# Crear pestañas
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Carga de archivos')
tab_control.add(tab2, text='Partículas a analizar')
tab_control.add(tab3, text='Cálculo y Análisis')
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

# Funciones para la lógica del programa
def expand_and_swap_tuples(tuples_list):
    expanded_list = []
    for t in tuples_list:
        for i in range(1, t[0] + 1):
            expanded_list.append((t[1], i))
    return expanded_list

#Esta función se utiliza para los nombres 
def tupla_a_cadena(tupla):
    if isinstance(tupla, tuple):
        return '(' + ', '.join(tupla_a_cadena(sub) for sub in tupla) + ')'
    else:
        return str(tupla)

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

def add_particle():
    cantidad = int(entry_quantity.get())
    particula = particle_choice.get()
    lista.append((cantidad, particula))
    lista_box.insert(tk.END, f"{cantidad} {particula}")

def remove_selected_particle():
    selected_indices = lista_box.curselection()
    for index in reversed(selected_indices):
        lista.pop(index)
        lista_box.delete(index)

def analyze_particles():
    global lista_num, lista_num_mod, lista_num_names, comb_pares_names, comb_trios_names, comb_cuartetos_names
    global combinaciones_pares, combinaciones_trios, combinaciones_cuartetos

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


def overwrite_list(listbox, comb_list, comb_tuples):

    selected_indices = listbox.curselection()
    selected_combinations = [comb_list[i] for i in selected_indices]

    if not selected_combinations:
        messagebox.showwarning("Advertencia", "No se ha seleccionado ninguna combinación.")
        return

    # Refrescar la lista en la interfaz
    comb_list.clear()
    comb_list.extend(selected_combinations)
    comb_tuples.clear()
    comb_tuples.extend(selected_combinations)

    listbox.delete(0, tk.END)
    for comb in comb_list:
        listbox.insert(tk.END, comb)

    messagebox.showinfo("Éxito", "La lista ha sido sobrescrita con las selecciones realizadas.")

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
    "La lista final de partículas numeradas tiene el formato: [(x, y, z)] donde:\n"
    "- x  es el tipo de la partícula.\n"
    "- y  su posición energética.\n"
    "- z  su carga eléctrica."
)
ttk.Label(tab2, text=explanation_text, justify="left").pack(pady=10)

# Botón para analizar
analyze_button = ttk.Button(tab2, text="Analizar", command=analyze_particles)
analyze_button.pack()

# Crear listas desplazables
def create_scrollable_listbox(root, title):
    frame = ttk.Frame(root)
    frame.pack()
    ttk.Label(frame, text=title).pack()
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox = tk.Listbox(frame, width=50, height=5, yscrollcommand=scrollbar.set, selectmode=tk.MULTIPLE)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH)
    scrollbar.config(command=listbox.yview)
    return listbox, frame


comb_pares_listbox, frame_comb_pares = create_scrollable_listbox(tab2, "Combinaciones de pares:")
ttk.Button(frame_comb_pares, text="Sobrescribir Lista", command=lambda: overwrite_list(comb_pares_listbox, comb_pares_names, combinaciones_pares)).pack()

comb_trios_listbox, frame_comb_trios = create_scrollable_listbox(tab2, "Combinaciones de tríos:")
ttk.Button(frame_comb_trios, text="Sobrescribir Lista", command=lambda: overwrite_list(comb_trios_listbox, comb_trios_names, combinaciones_trios)).pack()

comb_cuartetos_listbox, frame_comb_cuartetos = create_scrollable_listbox(tab2, "Combinaciones de cuartetos:")
ttk.Button(frame_comb_cuartetos, text="Sobrescribir Lista", command=lambda: overwrite_list(comb_cuartetos_listbox, comb_cuartetos_names, combinaciones_cuartetos)).pack()

#Aqui inicia la tercer pestaña

tk.Label(tab3, text="Momento de Calcular").pack()
# Filtrado de eventos
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

# Funciones adicionales
def Num_jets(evento):
    jets = evento[evento['typ'] == 4]
    njets = len(jets)
    return njets

def momentum_vector(pt, phi, eta):
    pt_x, pt_y, pt_z = (pt * np.cos(phi)), (pt * np.sin(phi)), pt * np.sinh(eta)
    return pt_x, pt_y, pt_z

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
#OBTENCIÓN PT, ETA, PHI
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
    
    return None


def eta_part(evento, listapart):
    tipo = listapart[0]
    posicion = listapart[1] - 1  # Ajustamos a índice base 0
    criterio = listapart[2] if len(listapart) > 2 else None  # Evita errores si listapart tiene menos elementos

    # Filtrar por tipo de partícula
    prt = evento[evento['typ'] == tipo]

    # Aplicar filtros adicionales según tipo
    if tipo in [1, 2] and criterio is not None:
        prt = prt[prt['ntrk'] == criterio]
    elif tipo == 4 and criterio is not None:
        if criterio == 0:
            prt = prt[prt['btag'] == criterio]
        elif criterio == 1:
            prt = prt[prt['btag'].isin([1, 2])]

    # Verificar si hay elementos suficientes
    if len(prt) > posicion >= 0:
        try:
            return prt.iloc[posicion]['eta']
        except IndexError:
            return None  # Si la posición está fuera del rango
    return None


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
    return None
#MASA TRANSVERSA
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
    return None
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
    return None

def calculos_eventos(df, lista_num, combinaciones_pares, combinaciones_trios, combinaciones_cuartetos, batch_size=300):
    current_event = []
    current_event_number = None
    masainv = []
    masainv_trios = []
    masainv_cuartetos = []
    masatrans = []
    deltar = []
    no_jets = []
    pt = []
    phi = []
    eta = []
    columpares = []
    columpares1 = []
    columpares2 = []
    colum = []
    colum1 = []
    colum2 = []
    columtrios = []
    columcuartetos = []

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
                            masatrans.append(m_trans(event_df, i))
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
                    masatrans.append(m_trans(event_df, i))
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

    for i in lista_num_names:
        colum.append('Pt' + str(i))
        colum1.append('Eta' + str(i))
        colum2.append('Phi' + str(i))

    for i in comb_pares_names:
        columpares.append('m_inv' + str(i))
        columpares1.append('m_trans' + str(i))
        columpares2.append('deltaR' + str(i))

    for i in comb_trios_names:
        columtrios.append('m_inv' + str(i))
    for i in comb_cuartetos_names:
        columcuartetos.append('m_inv' + str(i))

    csv_columtrios = pd.DataFrame(masainv_trios, columns=columtrios) if masainv_trios.size > 0 else pd.DataFrame()
    csv_columcuartetos = pd.DataFrame(masainv_cuartetos, columns=columcuartetos) if masainv_cuartetos.size > 0 else pd.DataFrame()
    csv_deltar = pd.DataFrame(deltar, columns=columpares2) if deltar.size > 0 else pd.DataFrame()
    csv_pt = pd.DataFrame(pt, columns=colum) if pt.size > 0 else pd.DataFrame()
    csv_eta = pd.DataFrame(eta, columns=colum1) if eta.size > 0 else pd.DataFrame()
    csv_phi = pd.DataFrame(phi, columns=colum2) if phi.size > 0 else pd.DataFrame()
    csv_minv = pd.DataFrame(masainv, columns=columpares) if masainv.size > 0 else pd.DataFrame()
    csv_mtrans = pd.DataFrame(masatrans, columns=columpares1) if masatrans.size > 0 else pd.DataFrame()

    csv_combined = pd.concat([csv_phi, csv_eta, csv_pt, csv_minv, csv_mtrans, csv_deltar, csv_columtrios, csv_columcuartetos], axis=1)
    csv_combined["#_jets"] = no_jets

    return csv_combined


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
            file_bg = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar BG filtrado")
            file_sg = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Guardar Signal filtrado")

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

# Función para crear histogramas
def crear_histograma(df_signal, df_background, columna, xlabel, ylabel, title, filename=None):
    if df_signal is None or df_background is None:
        messagebox.showerror("Error", "Los datos aún no han sido cargados.")
        return

    primera_columna = df_signal.columns[0]

    plt.hist(df_signal[columna], weights=df_signal[primera_columna], bins=50, edgecolor='black', alpha=0.5, label='Signal', density=True)
    plt.hist(df_background[columna], weights=df_background[primera_columna], bins=50, edgecolor='black', alpha=0.5, label='Background', density=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper right')
    
    if filename:
        plt.savefig(filename)

    plt.show()

def generar_grafica():
    columna_seleccionada = columna_var.get()
    
    if not columna_seleccionada:
        messagebox.showerror("Error", "Seleccione una columna.")
        return
    
    if df_combined is None:
        messagebox.showerror("Error", "No se encontró el archivo combinado. Asegúrese de realizar los cálculos primero.")
        return

    # Crear histograma con hue="Td" para diferenciar Signal y Background
    plt.figure(figsize=(10, 6))
    sns.histplot(df_combined, x=columna_seleccionada, hue="Td", kde=True, bins=30, element="step", common_norm=False)
    
    plt.xlabel(columna_seleccionada)
    plt.ylabel("Número de Eventos")
    plt.title(f"Distribución de {columna_seleccionada}")
    
    plt.legend(title="Tipo", labels=["Background (b)", "Signal (s)"])
    plt.grid()
    plt.show()

    # Aplicar límite si está activado
    if limitar_var.get():
        try:
            limite = float(limite_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido para el límite.")
            return
        
        df_filtrado = df_combined[df_combined[columna_seleccionada] <= limite]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df_filtrado, x=columna_seleccionada, hue="Td", kde=True, bins=30, element="step", common_norm=False)

        plt.xlabel(columna_seleccionada)
        plt.ylabel("Número de Eventos")
        plt.title(titulo_entry.get() if titulo_entry.get() else f"Distribución de {columna_seleccionada} (Filtrado)")
        
        plt.legend(title="Tipo", labels=["Background (b)", "Signal (s)"])
        plt.grid()

        # Guardar imagen
        nombre_archivo = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPG files", "*.jpg")])
        if nombre_archivo:
            plt.savefig(nombre_archivo)
            messagebox.showinfo("Éxito", f"Imagen guardada en: {nombre_archivo}")
        
        plt.show()

#Botones y entradas pestaña 3
filtrar_btn = tk.Button(tab3, text="Filtrar Eventos", command=on_filtrar_eventos)
filtrar_btn.pack(pady=10)



calcular_btn = tk.Button(tab3, text="Iniciar Cálculo", command=on_iniciar_calculo)
calcular_btn.pack(pady=10)

# Selección de columna
tk.Label(tab3, text="Seleccione una columna:").pack()
columna_var = tk.StringVar()
columna_menu = ttk.Combobox(tab3, textvariable=columna_var, values=list(columns_to_check))
columna_menu.pack()

# Checkbox para límite
limitar_var = tk.BooleanVar()
limitar_check = tk.Checkbutton(tab3, text="Imponer un límite", variable=limitar_var)
limitar_check.pack()

# Entrada para límite
tk.Label(tab3, text="Límite (opcional):").pack()
limite_entry = tk.Entry(tab3)
limite_entry.pack()

# Entrada para título
tk.Label(tab3, text="Título de la gráfica:").pack()
titulo_entry = tk.Entry(tab3)
titulo_entry.pack()

# Botón para generar gráfica
btn_generar = tk.Button(tab3, text="Generar Gráfica", command=generar_grafica)
btn_generar.pack()

# Ejecutar la aplicación
root.mainloop()

