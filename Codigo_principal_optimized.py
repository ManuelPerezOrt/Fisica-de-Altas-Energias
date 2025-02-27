from itertools import combinations
import numpy as np
import pandas as pd
import csv
import mimetypes
import math
import sys
from tqdm import tqdm
import time
import datatable as dt
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna #to search best parameters for signal background plot
import sys
import scipy
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from xgboost.sklearn import XGBClassifier
from matplotlib.patches import Patch, Circle # Correcting the patch error
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import pickle
from datetime import datetime
#Solicitar instrucciones desde terminal

mimetypes.add_type('lhco', '.lhco')
mimetypes.add_type('csv', '.csv')

# Ruta del archivo LHCO
signal_path=input( 'Ingrese el path de su archivo SIGNAL, puede ser de los formatos .lhco o .csv: \n')

pathsback = []
evit=int(input("Si ya tiene un bg con la base de datos calculada coloque 1 y el path que colocara a continuacion sera ese, de caso contrario ingrese 0: ")) 

print('Ingrese el path de su archivo de BACKGROUND, puede ser de los formatos .lhco o .csv: (presione enter para continuar) \n')
while True:
	paths = input('Path: ')
	if paths == '':
        	break
	pathsback.append(paths)
#print(pathsback)
print('\n')
Final_name=input('Ingrese nombre del archivo .csv que será generado por este código: ')
Final_name = Final_name + '.csv'
dfbg = pd.DataFrame()
for i in pathsback:
    mime_type, encoding = mimetypes.guess_type(i)

    # Verificamos si el tipo de archivo es 'lhco'
    if mime_type == 'lhco':
        data=pd.read_csv(i,sep=r'\s+')
    if mime_type == 'csv':
    	data=pd.read_csv(i)

    # Concatenar las bases de datos
    dfbg = pd.concat([dfbg, data], ignore_index=True)

mask = dfbg['#'] == 0
# Actualiza todas las columnas excepto la columna '#'
dfbg.loc[mask, dfbg.columns != '#'] = 10.0
filtered_dfbg = pd.DataFrame(dfbg)

mime_type, encoding = mimetypes.guess_type(signal_path)
# Verificamos si el tipo de archivo es 'lhco'
if mime_type == 'lhco':
    dfsg=pd.read_csv(signal_path,sep=r'\s+')

print('\n')
if mime_type == 'csv':
    dfsg=pd.read_csv(signal_path)
mask = dfsg['#'] == 0
# Actualiza todas las columnas excepto la columna '#'
dfsg.loc[mask, dfsg.columns != '#'] = 10.0
filtered_dfsg = pd.DataFrame(dfsg)

print('\n')

print("Por favor, ingrese las partículas que desea analizar en el estado final.")
print("Primero, coloque el número de partículas (n) seguido del nombre de la partícula.")
print("Las partículas disponibles son: a(photon), e-(electron), e+(positron), mu+(antimuon), mu-(muon), t(tau), j(jet).\n")
print("Ejemplo: 2 a \n")

# Solicitar entrada del usuario
lista = []

while True:
    elemento = input("Partícula: ")

    if elemento.lower() == '':
        break

    lista.append(elemento)

lista = [(int(e.split()[0]), e.split()[1]) for e in lista]

# Convertir nombres de partículas a números
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
lista_num = [(cantidad, particulas_dict[particula]) for cantidad, particula in lista]
lista_num.append((1, 6))

def expand_and_swap_tuples(tuples_list):
    expanded_list = []
    for t in tuples_list:
        for i in range(1, t[0] + 1):
            expanded_list.append((t[1], i))
    return expanded_list

lista_num = expand_and_swap_tuples(lista_num)
#Esta función se utiliza para los nombres 
def tupla_a_cadena(tupla):
    if isinstance(tupla, tuple):
        return '(' + ', '.join(tupla_a_cadena(sub) for sub in tupla) + ')'
    else:
        return str(tupla)
# Función para determinar el tercer valor basado en x
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
    elif x == 8:
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
lista_num_names=transformar_tuplas(lista_num)
print(lista_num_names)
comb_pares_names = list(combinations(lista_num_names, 2))
comb_trios_names = list(combinations(lista_num_names, 3))
comb_cuartetos_names = list(combinations(lista_num_names, 4))
lista_num = [(x, y, determinar_valor(x)) for (x, y) in lista_num]
# Cambiar los valores de antimuon (7) a 2 y positron (5) a 1
lista_num_mod = [(4 if x == 8 else 1 if x == 5 else 2 if x == 7 else x, y, z) for (x, y, z) in lista_num]
#print(lista_num_mod)

# Obtener todas las combinaciones de 2, 3 y 4 elementos
combinaciones_pares = list(combinations(lista_num_mod, 2))
combinaciones_trios = list(combinations(lista_num_mod, 3))
combinaciones_cuartetos = list(combinations(lista_num_mod, 4))

#print("Combinaciones de pares:", combinaciones_pares)
#print("Combinaciones de tríos:", combinaciones_trios)
#print("Combinaciones de cuartetos:", combinaciones_cuartetos)


"""
#Filtrado de LHCO(se modificara para los jets b en algun momento)
"""
def filtrar_eventos(df, num_list, batch_size=300):
    event_indices = []
    current_event = []
    current_event_number = None
    num_list_first_elements = [t[0] for t in num_list]
    num_list_first_third_elements = [(t[0], t[2]) for t in num_list]

    start = 0
    total_batches = (len(df) + batch_size - 1) // batch_size  # Calcular el número total de lotes
    with tqdm(total=total_batches, desc="Filtrando eventos") as pbar:
     while start < len(df):
        end = start + batch_size
        # Ajustar el final del lote para no cortar eventos a la mitad
        while end < len(df) and df.iloc[end]['#'] != 0:
            end += 1

        batch_df = df.iloc[start:end]
        for i, row in batch_df.iterrows():
            if row['#'] == 0:
                if current_event:
                    event_typ_counts = [r['typ'] for r in current_event]
                    event_typ_ntrk_tuples = [(r['typ'], r['ntrk']) for r in current_event]
                    event_typ_btag_tuples = [(r['typ'], r['btag']) for r in current_event]
                    if all(event_typ_counts.count(num) >= num_list_first_elements.count(num) for num in set(num_list_first_elements)):
                        if all(event_typ_ntrk_tuples.count(tup) >= num_list_first_third_elements.count(tup) for tup in num_list_first_third_elements if tup[0] in [1, 2]):
                            if all(sum(1 for _, btag in event_typ_btag_tuples if _ == 4 and btag != 0) >= num_list_first_third_elements.count((4, 1)) for tup in num_list_first_third_elements if tup[0] == 4 and tup[1] == 1):
	                            event_indices.extend(current_event)
                current_event = []
                current_event_number = row['#']
            current_event.append(row)

        if current_event:
            event_typ_counts = [r['typ'] for r in current_event]
            event_typ_ntrk_tuples = [(r['typ'], r['ntrk']) for r in current_event]
            event_typ_btag_tuples = [(r['typ'], r['btag']) for r in current_event]
            if all(event_typ_counts.count(num) >= num_list_first_elements.count(num) for num in set(num_list_first_elements)):
                if all(event_typ_ntrk_tuples.count(tup) >= num_list_first_third_elements.count(tup) for tup in num_list_first_third_elements if tup[0] in [1, 2]):
                	if all(sum(1 for _, btag in event_typ_btag_tuples if _ == 4 and btag != 0) >= num_list_first_third_elements.count((4, 1)) for tup in num_list_first_third_elements if tup[0] == 4 and tup[1] == 1):
	                    event_indices.extend(current_event)

        start = end
        pbar.update(1)  # Actualizar la barra de progreso
    return pd.DataFrame(event_indices)
# Aplicar la función a ambos DataFrames
if evit != 1:
	filtered_dfbg = filtrar_eventos(filtered_dfbg, lista_num_mod)
	print("Se filtro el bg")
	#filtered_dfbg.to_csv("Filteredbg.csv", index=False)
filtered_dfsg = filtrar_eventos(filtered_dfsg, lista_num_mod)
#filtered_dfsg.to_csv("Filteredsignal.csv", index=False)

print("Ya se realizó el filtrado de ambos df")
print("Inicia el proceso del cálculo para el df")
#Funcion para no.jets
def Num_jets(evento):
    jets=evento[evento['typ']== 4]
    njets=len(jets)
    return njets
#Vector de momento
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
        columpares1.append('m_trans ' + cadena)
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

# Aplicar la función a ambos DataFrames
csv_sig = calculos_eventos(filtered_dfsg, lista_num, combinaciones_pares, combinaciones_trios, combinaciones_cuartetos)
csv_sig['Td'] = "s"
if evit != 1 :
	csv_bg = calculos_eventos(filtered_dfbg, lista_num, combinaciones_pares, combinaciones_trios, combinaciones_cuartetos)
	csv_bg['Td'] = "b"
	name_bg=input("Inserte el nombre con el que quiere guardar los resultados del bg , este le puede ser util para acelerar el proceso si desea analizar otro caso con el mismo bg: ")
	name_bg=name_bg + ".csv"
	csv_bg.to_csv(name_bg, index=False)
	print(f"Se guardo los analisis para el BG por si se utilizará en proximos calculos para que no sea necesario volver a calcular, se guardo con el nombre: {name_bg}")
if evit == 1 :
	csv_bg=filtered_dfbg
df_combined = pd.concat([csv_bg, csv_sig], ignore_index=False)

# Mantener la numeración de la primera columna
df_combined.reset_index(drop=True, inplace=True)
df_combined.index += 1
#print(df_combined)

# Guardar la base de datos combinada
df_combined = df_combined.rename_axis('Evento').reset_index()
df_combined.to_csv(Final_name, index=False)

print(f'El archivo fue creado con el nombre: {Final_name}')
print("Ahora se iniciará con el proceso de entrenamiento de BDT, el archivo final se sobreescribira sobre el que ya fue creado")
#INICIA PROCESO DE GRAFICADO
results_df = pd.read_csv(Final_name)
signal_df = results_df[results_df['Td'] == 's']
background_df = results_df[results_df['Td'] == 'b']
# Definir las columnas a verificar (excluyendo la primera y la última columna)
columns_to_check = df_combined.columns[1:-1]

# Función para crear histogramas y guardar la gráfica
def crear_histograma(df_signal, df_background, columna, xlabel, ylabel, title, filename=None):
        # Usar la primera columna para los pesos
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
    
    if imponer_limite.lower() == 'si':
        # Preguntar al usuario el límite para la columna seleccionada
        limite = float(input(f"Por favor, ingrese el límite para la columna {columna_seleccionada}: "))

        # Aplicar el filtro del límite a ambas tablas
        signal_df_filtrado = signal_df[signal_df[columna_seleccionada] <= limite]
        background_df_filtrado = background_df[background_df[columna_seleccionada] <= limite]

        # Preguntar al usuario el nombre del archivo para guardar la gráfica
        nombre_archivo = input("Por favor, ingrese el nombre del archivo para guardar la gráfica: ")
        nombre_archivo = nombre_archivo + ".jpg"

        # Crear el histograma para la columna seleccionada con el límite y guardar la gráfica
        name_graphic=input("Inserte el título que desea poner para la gráfica")
        crear_histograma(signal_df_filtrado, background_df_filtrado, columna_seleccionada, columna_seleccionada, 'N.Events', name_graphic, nombre_archivo)
print("INICIA PROCESO PARA BDT")
#INICIA PROCESO PARA BDT
# Separar la primera fila (títulos de las columnas)
column_titles = df_combined.iloc[0]

# Ordenar aleatoriamente las filas a excepción de la primera
df_shuffled = df_combined.iloc[1:].sample(frac=1).reset_index(drop=True)

# Volver a agregar la primera fila (títulos de las columnas)
df_shuffled.loc[-1] = column_titles
df_shuffled.index = df_shuffled.index + 1
df_shuffled = df_shuffled.sort_index()   #ESTE VA A LLEVAR LA INFORMACION JUNTA Y MEZCLADA
def roc(test_x,test_y,train_x,train_y, model):
    """"
    It presents the roc curve, which shows the accuracy of the classifier, 
    the closer the area 1, the better the classifier.   """
    plt.figure(figsize=(10,7))
    plt.title('ROC curve', fontsize=20)
    model_predict = model.predict_proba(test_x)
    model_predict = model_predict[:,1]
    auc_score = roc_auc_score(test_y, model_predict)
    fpr, tpr, _ = roc_curve(test_y, model_predict) #roc_curve(true binary labels, prediction scores)
    print('Test : ', auc_score)
    plt.plot(tpr, 1-fpr, label='Test   '+ str(round(auc_score, 4)), color='firebrick', linewidth=2)

    model_predict = model.predict_proba(train_x)
    model_predict = model_predict[:,1]
    auc_score = roc_auc_score(train_y, model_predict)
    fpr, tpr, _ = roc_curve(train_y, model_predict)
    plt.plot(tpr, 1-fpr, label='Train   ' + str(round(auc_score,4)) , color='midnightblue', linewidth=2)
    print('Train : ', auc_score)
    plt.legend(loc='best',fontsize=20)
    plt.ylabel('Purity', fontsize=20)
    plt.xlabel('Efficiency', fontsize=20)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.ylim(0.7,1.1)
    #plt.show()
    plt.grid=True
    plt.xticks(fontsize = 15); 
    plt.yticks(fontsize = 15); 
    nombre=input("Inserte el nombre con el que quiere guardar la imagen que contiene la curva roc obtenida para el modelo de entrenamiento: ")
    nombre= nombre + ".jpg"
    plt.savefig(nombre)

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

print('Size of data: {}'.format(df_shuffled.shape))
print('Number of events: {}'.format(df_shuffled.shape[0]))
#print('Number of columns: {}'.format(df_shuffled.shape[1]))

print ('\nList of features in dataset:')
for col in df_shuffled.columns:
    print(col)

print('Number of signal events: {}'.format(len(df_shuffled[df_shuffled.Td == 's'])))
print('Number of background events: {}'.format(len(df_shuffled[df_shuffled.Td == 'b'])))
print('Fraction signal: {}'.format(len(df_shuffled[df_shuffled.Td == 's'])/(float)(len(df_shuffled[df_shuffled.Td == 's']) + len(df_shuffled[df_shuffled.Td == 'b']))))

factor=int(input("Se recomienda tener un fraction signal=0.5, a cuanto de sera limitar la cantidad de datos tanto de bg como de signal: ")) 
# Get the 's' and 'b' 
s_events = df_shuffled[df_shuffled['Td'] == 's'].head(factor)  # specified number of signal events
b_events = df_shuffled[df_shuffled['Td'] == 'b'].head(factor)  #specified number of background events

# Combining filtered signal and background datasets for training
df_shuffled = pd.concat([s_events, b_events], ignore_index=True)

print('Number of signal events: {}'.format(len(df_shuffled[df_shuffled.Td == 's'])))
print('Number of background events: {}'.format(len(df_shuffled[df_shuffled.Td == 'b'])))
print('Fraction signal: {}'.format(len(df_shuffled[df_shuffled.Td == 's'])/(float)(len(df_shuffled[df_shuffled.Td == 's']) + len(df_shuffled[df_shuffled.Td == 'b']))))

Sig_df = (df_shuffled[df_shuffled.Td == 's'])
Bkg_df = (df_shuffled[df_shuffled.Td == 'b'])
vars_for_train = Sig_df.columns
vars_for_train = vars_for_train.drop(["Td", "Evento"])
print(vars_for_train)
data4label   = df_shuffled[vars_for_train]
signal4train = Sig_df[vars_for_train]
bkg4train    = Bkg_df[vars_for_train]
correlations = signal4train.corr()
# Checking for variables with high correlation
# Select upper triangle of correlation matrix
upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

#Filter the columns to drop, keeping present in the DataFrame
to_drop_filtered = [column for column in to_drop if column in signal4train.columns]

#Drop features without using inplace=True to avoid SettingWithCopyWarning
signal4train = signal4train.drop(to_drop_filtered, axis=1)

#same for bkg4train
bkg4train = bkg4train.drop(to_drop_filtered, axis=1)
print(f"Se borraran las siguientes columnas dadas su correlacion :  {to_drop}")
vars_for_train = list(vars_for_train)
for var in to_drop:
    vars_for_train.remove(var)
data4label   = df_shuffled[vars_for_train]
signal4train = Sig_df[vars_for_train]
bkg4train    = Bkg_df[vars_for_train]
n_samples = min(signal4train.shape[0], bkg4train.shape[0])
bkg4train = bkg4train.sample(n=n_samples, random_state=42)
#copy
bkg4train = bkg4train.copy()
signal4train = signal4train.copy()

#assignments
bkg4train.loc[:, 'signal/bkgnd'] = 0
signal4train.loc[:, 'signal/bkgnd'] = 1

# Concatenate the DataFrames
df_4train = pd.concat([signal4train, bkg4train])
#GENERAL DATA
# separate the labels and the features
features_ = df_4train.drop(['signal/bkgnd'], axis=1) #train_x features = all minus (signal/bkgnd and masses)
label_    = df_4train['signal/bkgnd']   #train_Y
#SIGNAL
signal_features = signal4train.drop(['signal/bkgnd'], axis=1) #signal_x
signal_label    = signal4train['signal/bkgnd'] #signal_y
#BKGND
bkgnd_features = bkg4train.drop(['signal/bkgnd'], axis=1) # bkgnd_x
bkgnd_labels   = bkg4train['signal/bkgnd'] # bkgnd_y
test=input("El entrenamiento se realizará preestablecidamente con una proporcion de 0.8 para el entrenamiento, ¿desea cambiarla?Si=1, No=0: ")
#SIGNAL
if test == "1" :
    size=input("Coloque el valor de la nueva proporcion que tendrá el entrenamiento (anteriormente era 0.8): ")
    size=float(size)
size=0.8
train_sig_feat, test_sig_feat, train_sig_lab, test_sig_lab = train_test_split(signal_features, signal_label,
                                                  test_size=size,
                                                  random_state=1)
#BKGND IZQ
train_bkg_feat, test_bkg_feat, train_bkg_lab, test_bkg_lab = train_test_split(bkgnd_features, bkgnd_labels,
                                                  test_size=size,
                                                  random_state=1)
test_feat = pd.concat([test_sig_feat, test_bkg_feat]) # test_x
test_lab = pd.concat([test_sig_lab, test_bkg_lab]) # test_y
train_feat = pd.concat([train_sig_feat, train_bkg_feat]) # train_x
train_lab = pd.concat([train_sig_lab, train_bkg_lab]) # train_y

eval_set = [(train_feat, train_lab), (test_feat, test_lab)]
test = test_feat.assign(label=test_lab)
train = train_feat.assign(label=train_lab)
cols = vars_for_train
#Calculo para los mejores hiperparametros
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
         'n_estimators':100
    }

    #Parameters close to manual parameters
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

# including manual parameters as initial trial
    study = optuna.create_study(direction='maximize')
    study.enqueue_trial(manual_params)  
    study.optimize(objective, n_trials=50) #optimizing

#Print results
    print("Best parameters:", study.best_trial.params)
    print("Best score:", study.best_trial.value)
# Guardar los mejores hiperparámetros
    best_hyperparams = study.best_trial.params

# Fill missing values in training and testing data with the mean of the training features
    train_feat[cols] = train_feat[cols].fillna(train_feat[cols].mean())
    test_feat[cols] = test_feat[cols].fillna(train_feat[cols].mean())

# Create evaluation set with consistent feature columns
    eval_set = [(train_feat[cols], train_lab), (test_feat[cols], test_lab)]

# Asignar automáticamente los mejores hiperparámetros en el modelo
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        **best_hyperparams
    )
    # Define and fit the XGBoost model with early stopping in the constructor
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
    
    # Fit the model with the corrected eval_set and verbose turned off
    modelv1.fit(train_feat[cols], train_lab, eval_set=eval_set, verbose=False)
    fig, ax = plot_classifier_distributions(modelv1, test=test, train=train, cols = cols, print_params=False)
    #ax.set_title(r'Total sample size $\approx$ '+str(len(train) + len(test))+' optimized')
    print(f'Background(Ks-pval): {_ks_back}')
    print(f'Signal(Ks-pval): {_ks_sign}')
name_dist=input("Inserte el nombre con que desea guardar la grafica que sera creada como el resultado del clasificador: ")
plt.savefig(name_dist + '.png')
#plt.savefig('classifier_distribution.pdf')
plt.show()

roc(test_feat[cols], test_lab, train_feat[cols], train_lab, modelv1)

plt.figure(figsize=(16, 12))
print("Se mostrará ahora una grafica que organiza por orden de importancia las variables que utilizo el modelo(F Score) ")
xgb.plot_importance(modelv1)
save=input("¿Desea guardar la imagen? si=1, no=0?: ")
if save == "1":
    name_save=input("Inserte el nombre con el que desea guardarla: ")
    plt.savefig(name_save + '.png')
plt.show()
pickle.dump(modelv1, open(f"model_xgbv1_opt.dat", "wb"))

Mymodel = pickle.load(open(f"model_xgbv1_opt.dat", "rb")) 
def mycsvfile(pdf, myname):
    """ this function will embedded the variable xgb in our dataset. """
    selected_vars = cols
    datalabel = pdf[selected_vars]
    
    ##
    predict = Mymodel.predict_proba(datalabel)
    
    ##
    pdf['XGB'] = predict[:,1]
    
    ## DataTable
    table_df = dt.Frame(pdf)
    table_df.to_csv(myname)
mycsvfile(df_shuffled,Final_name)
print(f'El archivo fue creado con el nombre: {Final_name}')

#INICIA EL CALCULO DE SIGNIFICANCIA
df_shuffled=pd.read_csv(Final_name)
s_events = df_shuffled[df_shuffled['Td'] == "s"].head(factor)  # specified number of signal events
b_events = df_shuffled[df_shuffled['Td'] == "b"].head(factor)  #specified number of background events

# Combining filtered signal and background datasets for training
df_shuffled = pd.concat([s_events, b_events], ignore_index=True)

XSsignal = float(input("Ingrese la cross section de la señal en pb: ")) # Sección eficaz de la señal
XSbackground = float(input("Ingrese la cross section del background en pb: "))  # Sección eficaz del background
def CalSig(mypdf, xgbcut,XSs,XSb):
    mypdf = mypdf[mypdf['XGB'] > xgbcut]  # Filtrar los datos

    Ns_csv = len(mypdf[mypdf.Td == "s"])
    Nb_csv = len(mypdf[mypdf.Td == "b"])
    fraction_csv_s = Ns_csv / factor
    fraction_csv_b = Nb_csv / factor
    pbTOfb = 1000  # factor conversión de pb a fb
    IntLumi = 3000  # Luminosidad integrada
    alpha = XSs * pbTOfb * IntLumi / factor # Factor de escalamiento de los eventos generados a eventos calculados de la señal
    beta = XSb * pbTOfb * IntLumi / factor  # Factor de escalamiento de los eventos generados a eventos calculados del background

    try:
        Sig = (alpha * Ns_csv) / (math.sqrt((alpha * Ns_csv) + (beta * Nb_csv)))
    except ZeroDivisionError:
        print("División por cero detectada. Continuando con el siguiente cálculo.")
        Sig = float(0)  # O cualquier valor que consideres apropiado para manejar el error
    print('Number of signal events: {}'.format(len(mypdf[mypdf.Td == 's'])))
    print('Number of background events: {}'.format(len(mypdf[mypdf.Td == 'b'])))
    print(f'La significancia obtenida es :{Sig},con un corte sobre XGB en :{xgbcut}')
    return Sig

def main():
    global results_df
    data = df_shuffled  # Cargar los datos desde un archivo CSV

    Sigval = []
    XGBval = []
    xgbi = 0.5
    for jj in range(499):
        Sigval.append(CalSig(data, xgbi,XSsignal,XSbackground))
        XGBval.append(xgbi)
        xgbi = xgbi + 0.001

    # Crear un DataFrame con los resultados
    results_df = pd.DataFrame({
        'XGB_cut': XGBval,
        'Significance': Sigval
    })

    # Guardar el DataFrame en un archivo CSV
    sig_nom=input("Ingrese el nombre que ahora guardara los calculos de la significancia: ")
    results_df.to_csv(sig_nom + '.csv', index=False)
    #print(f"Resultados guardados en 'significance_results.csv")

if __name__ == '__main__':
    main()
# Encontrar la fila con la significancia más alta
max_significance_row = results_df.loc[results_df['Significance'].idxmax()]

# Convertir la fila a una tupla
max_significance_tuple = tuple(max_significance_row)
print(max_significance_tuple)

