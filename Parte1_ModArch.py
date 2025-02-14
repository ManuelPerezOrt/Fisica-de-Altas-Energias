import tkinter as tk
from tkinter import filedialog, messagebox
import mimetypes
import pandas as pd

# Agregar tipos de archivos
mimetypes.add_type('lhco', '.lhco')
mimetypes.add_type('csv', '.csv')

def select_signal_file():
    filepath = filedialog.askopenfilename(filetypes=[("LHCO or CSV", "*.lhco;*.csv")])
    if filepath:
        signal_entry.delete(0, tk.END)
        signal_entry.insert(0, filepath)

def add_background_file():
    filepath = filedialog.askopenfilename(filetypes=[("LHCO or CSV", "*.lhco;*.csv")])
    if filepath:
        background_listbox.insert(tk.END, filepath)

def remove_selected_background():
    selected_indices = background_listbox.curselection()
    for index in reversed(selected_indices):
        background_listbox.delete(index)

def generate_csv():
    signal_path = signal_entry.get()
    background_paths = background_listbox.get(0, tk.END)
    
    if not signal_path:
        messagebox.showerror("Error", "Debe seleccionar un archivo de señal.")
        return
    
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not save_path:
        return
    
    # Procesar BACKGROUND
    dfbg = pd.DataFrame()
    for i in background_paths:
        mime_type, encoding = mimetypes.guess_type(i)
        if mime_type == 'lhco':
            data = pd.read_csv(i, sep=r'\s+')
        elif mime_type == 'csv':
            data = pd.read_csv(i)
        dfbg = pd.concat([dfbg, data], ignore_index=True)
    
    # Procesar SIGNAL
    mime_type, encoding = mimetypes.guess_type(signal_path)
    if mime_type == 'lhco':
        dfsg = pd.read_csv(signal_path, sep=r'\s+')
    elif mime_type == 'csv':
        dfsg = pd.read_csv(signal_path)
    
    # Filtrar eventos con '# == 0'
    mask = dfbg['#'] == 0
    dfbg.loc[mask, dfbg.columns != '#'] = 10.0
    filtered_dfbg = dfbg.copy()
    
    mask = dfsg['#'] == 0
    dfsg.loc[mask, dfsg.columns != '#'] = 10.0
    filtered_dfsg = dfsg.copy()
    
    # Guardar CSV
    filtered_dfbg.to_csv(save_path, index=False)
    messagebox.showinfo("Éxito", f"Archivo guardado en: {save_path}")

# Crear ventana
root = tk.Tk()
root.title("Selector de Archivos")
root.geometry("500x400")

# Widgets
signal_label = tk.Label(root, text="Archivo SIGNAL:")
signal_label.pack()
signal_entry = tk.Entry(root, width=50)
signal_entry.pack()
signal_button = tk.Button(root, text="Seleccionar", command=select_signal_file)
signal_button.pack()

background_label = tk.Label(root, text="Archivos BACKGROUND:")
background_label.pack()
background_listbox = tk.Listbox(root, width=50, height=5)
background_listbox.pack()
background_button = tk.Button(root, text="Añadir", command=add_background_file)
background_button.pack()
remove_button = tk.Button(root, text="Eliminar Seleccionado", command=remove_selected_background)
remove_button.pack()

generate_button = tk.Button(root, text="Generar CSV", command=generate_csv)
generate_button.pack()

# Ejecutar la aplicación
root.mainloop()


