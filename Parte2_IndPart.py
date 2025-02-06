import tkinter as tk
from tkinter import messagebox
from itertools import combinations
import numpy as np

# Función para mostrar el cuadro de entrada
def ingresar_particulas():
    def agregar_particula():
        particula = entry_particula.get()
        if particula:
            lista_particulas.append(particula)
            listbox.insert(tk.END, particula)
            entry_particula.delete(0, tk.END)

    def procesar():
        if not lista_particulas:
            messagebox.showerror("Error", "Debe ingresar al menos una partícula.")
            return
        
        lista = [(int(e.split()[0]), e.split()[1]) for e in lista_particulas]
        lista_num = []
        for i in lista:
            lista2 = list(i)
            particle_dict = {'photon': 0, 'electron': 1, 'muon': 2, 'tau': 3, 'jet': 4}
            lista2[1] = particle_dict.get(lista2[1], 5)  # Si no es una partícula válida, se asigna un código 5
            lista_num.append(tuple(lista2))

        lista_num.append((1, 6))  # Para algún caso especial de MET
        num_list = [t[1] for t in lista_num]

        combinaciones_pares = list(combinations(lista_num, 2))
        combinaciones_trios = list(combinations(lista_num, 3))
        combinaciones_cuartetos = list(combinations(lista_num, 4))

        # Mostramos las combinaciones en el cuadro de texto
        resultado = f"Combinaciones de pares: {combinaciones_pares}\nCombinaciones de tríos: {combinaciones_trios}\nCombinaciones de cuartetos: {combinaciones_cuartetos}"
        text_resultado.config(state=tk.NORMAL)
        text_resultado.delete(1.0, tk.END)
        text_resultado.insert(tk.END, resultado)
        text_resultado.config(state=tk.DISABLED)

    # Ventana principal de ingreso de partículas
    ventana = tk.Tk()
    ventana.title("Ingreso de partículas")
    
    # Área de texto para mostrar los mensajes
    text_instrucciones = tk.Text(ventana, height=10, width=70)
    text_instrucciones.pack(pady=10)
    text_instrucciones.insert(tk.END, "Por favor, ingrese las partículas que desea analizar en el estado final.\n")
    text_instrucciones.insert(tk.END, "Primero, coloque el número de partículas (n) seguido del nombre de la partícula.\n")
    text_instrucciones.insert(tk.END, "Las partículas disponibles son: photon, electron, muon, tau, jet.\n\n")
    text_instrucciones.insert(tk.END, "Ejemplo: 2 photon\n")
    text_instrucciones.config(state=tk.DISABLED)

    entry_particula = tk.Entry(ventana)
    entry_particula.pack(pady=5)
    
    boton_agregar = tk.Button(ventana, text="Agregar partícula", command=agregar_particula)
    boton_agregar.pack(pady=5)
    
    listbox = tk.Listbox(ventana, height=10, width=40)
    listbox.pack(pady=10)

    boton_procesar = tk.Button(ventana, text="Procesar partículas", command=procesar)
    boton_procesar.pack(pady=10)

    lista_particulas = []

    # Área de texto para mostrar el resultado de las combinaciones
    text_resultado = tk.Text(ventana, height=10, width=70)
    text_resultado.pack(pady=10)
    text_resultado.config(state=tk.DISABLED)

    ventana.mainloop()

# Función para calcular el vector de momento
def momentum_vector(pt, phi, eta):
    pt_x, pt_y, pt_z = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return pt_x, pt_y, pt_z

# Función para calcular la distancia DeltaR
def Deltar(evento, comb):
    prt1 = evento[evento['typ'] == comb[0][0]]
    prt2 = evento[evento['typ'] == comb[1][0]]
    if not prt1.empty and not prt2.empty:
        eta_prt1, eta_prt2 = prt1.iloc[comb[0][1] - 1]['eta'], prt2.iloc[comb[1][1] - 1]['eta']
        phi_prt1, phi_prt2 = prt1.iloc[comb[0][1] - 1]['phi'], prt2.iloc[comb[1][1] - 1]['phi']
        return np.sqrt((eta_prt1 - eta_prt2) ** 2 + (phi_prt1 - phi_prt2) ** 2)
    return None

# Función para calcular la masa transversal
def m_trans(evento, comb):
    prt1 = evento[evento['typ'] == comb[0][0]]
    prt2 = evento[evento['typ'] == comb[1][0]]
    if not prt1.empty and not prt2.empty:
        pt1_x, pt1_y, pt1_z = momentum_vector(prt1.iloc[comb[0][1] - 1]['pt'], prt1.iloc[comb[0][1] - 1]['phi'], prt1.iloc[comb[0][1] - 1]['eta'])
        pt2_x, pt2_y, pt2_z = momentum_vector(prt2.iloc[comb[1][1] - 1]['pt'], prt2.iloc[comb[1][1] - 1]['phi'], prt2.iloc[comb[1][1] - 1]['eta'])
        m_trans = np.sqrt((np.sqrt(pt1_x ** 2 + pt1_y ** 2) + np.sqrt(pt2_x ** 2 + pt2_y ** 2)) ** 2 - (pt1_x + pt2_x) ** 2 - (pt1_y + pt2_y) ** 2)
        return m_trans
    return None

# Llamamos a la interfaz para ingresar partículas
ingresar_particulas()
