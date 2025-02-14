import tkinter as tk
from tkinter import messagebox, filedialog
from itertools import combinations
import pandas as pd

# Diccionarios de partículas
particulas_dict = {
    'photon': 0,
    'electron': 1,
    'muon': 2,
    'tau': 3,
    'jet': 4,
    'MET': 6,
    'positron': 5,
    'antimuon': 7
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

def determinar_valor(x):
    if x == 6:
        return 0
    elif x == 1:
        return -1
    elif x == 2:
        return -1
    elif x == 0:
        return 0
    elif x == 3:
        return 0
    elif x == 4:
        return 0
    elif x == 5:
        return 1
    elif x == 7:
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
            if t[1] == 1:
                referencia = 'leading'
            elif t[1] == 2:
                referencia = 'subleading'
            elif t[1] == 3:
                referencia = 'tertiary'
            elif t[1] == 4:
                referencia = 'quaternary'
            elif t[1] == 5:
                referencia = 'quinary'
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
    lista_num = [(cantidad, particulas_dict[particula]) for cantidad, particula in lista]
    lista_num.append((1, 6))

    lista_num = expand_and_swap_tuples(lista_num)
    
    lista_num_names = transformar_tuplas(lista_num)
    comb_pares_names = list(combinations(lista_num_names, 2))
    comb_trios_names = list(combinations(lista_num_names, 3))
    comb_cuartetos_names = list(combinations(lista_num_names, 4))
    
    lista_num = [(x, y, determinar_valor(x)) for (x, y) in lista_num]
    lista_num_mod = [(1 if x == 5 else 2 if x == 7 else x, y, z) for (x, y, z) in lista_num]
    
    global combinaciones_pares, combinaciones_trios, combinaciones_cuartetos
    combinaciones_pares = list(combinations(lista_num_mod, 2))
    combinaciones_trios = list(combinations(lista_num_mod, 3))
    combinaciones_cuartetos = list(combinations(lista_num_mod, 4))

    comb_pares_listbox.delete(0, tk.END)
    for comb in combinaciones_pares:
        comb_pares_listbox.insert(tk.END, comb)

    comb_trios_listbox.delete(0, tk.END)
    for comb in combinaciones_trios:
        comb_trios_listbox.insert(tk.END, comb)

    comb_cuartetos_listbox.delete(0, tk.END)
    for comb in combinaciones_cuartetos:
        comb_cuartetos_listbox.insert(tk.END, comb)

def overwrite_list(listbox, comb_list):
    selected_indices = listbox.curselection()
    selected_combinations = [comb_list[i] for i in selected_indices]
    comb_list.clear()
    comb_list.extend(selected_combinations)
    listbox.delete(0, tk.END)
    for comb in comb_list:
        listbox.insert(tk.END, comb)
    messagebox.showinfo("Éxito", "La lista ha sido sobrescrita con las selecciones realizadas.")

# Crear ventana
root = tk.Tk()
root.title("Análisis de Partículas")
root.geometry("600x800")

# Widgets
tk.Label(root, text="Ingrese la cantidad y tipo de partícula:").pack()

frame_input = tk.Frame(root)
frame_input.pack()

entry_quantity = tk.Entry(frame_input, width=10)
entry_quantity.pack(side=tk.LEFT)
particle_choice = tk.StringVar()
particle_choice.set("photon")
option_menu = tk.OptionMenu(frame_input, particle_choice, *particulas_dict.keys())
option_menu.pack(side=tk.LEFT)

add_button = tk.Button(frame_input, text="Añadir Partícula", command=add_particle)
add_button.pack(side=tk.LEFT)

remove_button = tk.Button(frame_input, text="Eliminar Selección", command=remove_selected_particle)
remove_button.pack(side=tk.LEFT)

frame_lista_box = tk.Frame(root)
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
tk.Label(root, text=explanation_text, justify="left").pack(pady=10)

analyze_button = tk.Button(root, text="Analizar", command=analyze_particles)
analyze_button.pack()

# Scrollable frames for combinations
def create_scrollable_listbox(root, title):
    frame = tk.Frame(root)
    frame.pack()
    tk.Label(frame, text=title).pack()
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox = tk.Listbox(frame, width=50, height=5, yscrollcommand=scrollbar.set, selectmode=tk.MULTIPLE)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH)
    scrollbar.config(command=listbox.yview)
    return listbox, frame

comb_pares_listbox, frame_comb_pares = create_scrollable_listbox(root, "Combinaciones de pares:")
tk.Button(frame_comb_pares, text="Sobrescribir Lista", command=lambda: overwrite_list(comb_pares_listbox, combinaciones_pares)).pack()

comb_trios_listbox, frame_comb_trios = create_scrollable_listbox(root, "Combinaciones de tríos:")
tk.Button(frame_comb_trios, text="Sobrescribir Lista", command=lambda: overwrite_list(comb_trios_listbox, combinaciones_trios)).pack()

comb_cuartetos_listbox, frame_comb_cuartetos = create_scrollable_listbox(root, "Combinaciones de cuartetos:")
tk.Button(frame_comb_cuartetos, text="Sobrescribir Lista", command=lambda: overwrite_list(comb_cuartetos_listbox, combinaciones_cuartetos)).pack()

# Ejecutar la aplicación
root.mainloop()
