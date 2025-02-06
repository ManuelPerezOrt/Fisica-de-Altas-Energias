from itertools import combinations
import numpy as np
import pandas as pd
import csv
import mimetypes
import sys
from tqdm import tqdm
import time

#Solicitar instrucciones desde terminal

mimetypes.add_type('lhco', '.lhco')
mimetypes.add_type('csv', '.csv')
# Ruta del archivo LHCO
signal_path=input( 'Ingrese el path de su archivo SIGNAL, puede ser de los formatos .lhco o .csv: \n')

pathsback = []
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
#data.loc[data['#'] == 0, :] = float('10')
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
#data.loc[data['#'] == 0, :] = float('10')
filtered_dfsg = pd.DataFrame(dfsg)

#print(filtered_dfbg)
#print(filtered_dfsg)

print('\n')

print("Por favor, ingrese las partículas que desea analizar en el estado final.")
print("Primero, coloque el número de partículas (n) seguido del nombre de la partícula.")
print("Las partículas disponibles son: photon, electron, muon, tau, jet.\n")
print("Ejemplo: 2 photon\n")

# Solicitar entrada del usuario

lista = []

while True:
    elemento = input("Partícula: ")

    if elemento.lower() == '':
        break

    lista.append(elemento)
lista = [(int(e.split()[0]), e.split()[1]) for e in lista]
print(lista)
cantidad=[]
particulas=[]
num_list = []
lista_num=[]
for i in lista:
        print (i[1])
        lista2=list(i)
        if lista2[1] == 'photon':
            lista2[1] = 0
        if lista2[1] == 'electron':
            lista2[1] = 1
        if lista2[1] == 'muon':
            lista2[1] = 2
        if lista2[1] == 'tau':
            lista2[1] = 3
        if lista2[1] == 'jet':
            lista2[1] = 4
        lista_num.append(tuple(lista2))
lista_num.append((1,6))
#print(num_list)
def expand_and_swap_tuples(tuples_list):
    expanded_list = []
    for t in tuples_list:
        for i in range(1, t[0] + 1):
            expanded_list.append((t[1], i))
    return expanded_list
lista_num=expand_and_swap_tuples(lista_num)
num_list = [t[0] for t in lista_num]

print(lista_num)
# Obtener todas las combinaciones de 2 elementos
combinaciones_pares = list(combinations(lista_num, 2))
combinaciones_trios = list(combinations(lista_num, 3))
combinaciones_cuartetos=list(combinations(lista_num, 4))
#lista = pd.DataFrame(num_list, columns=['typ'])
#print(lista)

"""
#Filtrado de LHCO
"""
def filtrar_eventos(df, num_list):
    event_indices = []
    current_event = []
    current_event_number = None

    for i, row in df.iterrows():
        if row['#'] == 0:
            if current_event:
                event_typ_counts = [r['typ'] for r in current_event]
                if all(event_typ_counts.count(num) >= num_list.count(num) for num in set(num_list)):
                    event_indices.extend(current_event)
                current_event = []
            current_event_number = row['#']
        current_event.append(row)

    if current_event:
        event_typ_counts = [r['typ'] for r in current_event]
        if all(event_typ_counts.count(num) >= num_list.count(num) for num in set(num_list)):
            event_indices.extend(current_event)

    return pd.DataFrame(event_indices)

# Aplicar la función a ambos DataFrames
filtered_dfbg = filtrar_eventos(filtered_dfbg, num_list)
filtered_dfsg = filtrar_eventos(filtered_dfsg, num_list)
print(filtered_dfbg)
print(filtered_dfsg)
#Funcion para no.jets
def Num_jets(evento):
    jets=evento[evento['typ']==4]
    njets=len(jets)
    return njets
#Vector de momento
def momentum_vector(pt, phi, eta):
     pt_x, pt_y, pt_z = (pt * np.cos(phi)), (pt * np.sin(phi)), pt * np.sinh(eta)
     return pt_x, pt_y, pt_z
#DeltaR
def Deltar(evento,comb):
    prt1 = evento[evento['typ'] == comb[0][0]]
    prt2 = evento[evento['typ']== comb[1][0]]
    if not prt1.empty and not prt2.empty:
        # Obtener el pt del primer fotón y de la MET
        #print(posicion1)
        posicion1=comb[0][1]-1
        posicion2=comb[1][1]-1
        eta_prt1 = prt1.iloc[posicion1]['eta']
        eta_prt2 = prt2.iloc[posicion2]['eta']
        phi_prt1 = prt1.iloc[posicion1]['phi']
        phi_prt2 = prt2.iloc[posicion2]['phi']
        return np.sqrt((eta_prt1-eta_prt2)**2 + (phi_prt1-phi_prt2)**2)
#OBTENCIÓN PT, ETA, PHI
def phi_part(evento,listapart):
    prt=evento[evento['typ']==listapart[0]]
    posicion=listapart[1]-1
    phi_prt = prt.iloc[posicion]['phi']
    return phi_prt

def eta_part(evento,listapart):
    prt=evento[evento['typ']==listapart[0]]
    posicion=listapart[1]-1
    eta_prt = prt.iloc[posicion]['eta']
    return eta_prt

def pt_part(evento,listapart):
    prt=evento[evento['typ']==listapart[0]]
    posicion=listapart[1]-1
    pt_prt = prt.iloc[posicion]['pt']
    return pt_prt

#MASA TRANSVERSA
def m_trans(evento,comb):
    # Filtrar las partículas
    prt1 = evento[evento['typ'] == comb[0][0]]
    prt2 = evento[evento['typ']== comb[1][0]]
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
        m_trans=np.sqrt(
                    (np.sqrt(pt1_x**2 + pt1_y**2 ) + np.sqrt(pt2_x**2 + pt2_y**2 ))**2 -
                    (pt1_x + pt2_x )**2 - (pt1_y + pt2_y )**2
                )
        # print(m_trans)
        return  m_trans
    return None
#MASA INVARIANTE
def m_inv(evento, comb):
    # Filtrar las partículas
    prt = [evento[evento['typ'] == c[0]] for c in comb]
    
    if all(not p.empty for p in prt):
        posiciones = [c[1] - 1 for c in comb]
        pt = [p.iloc[pos]['pt'] for p, pos in zip(prt, posiciones)]
        eta = [p.iloc[pos]['eta'] for p, pos in zip(prt, posiciones)]
        phi = [p.iloc[pos]['phi'] for p, pos in zip(prt, posiciones)]
        
        momentum = [momentum_vector(pt[i], phi[i], eta[i]) for i in range(len(comb))]
        pt_x, pt_y, pt_z = zip(*momentum)
        
        m_in = np.sqrt(
            (sum(np.sqrt(px**2 + py**2 + pz**2) for px, py, pz in zip(pt_x, pt_y, pt_z)))**2 -
            sum(px for px in pt_x)**2 -
            sum(py for py in pt_y)**2 -
            sum(pz for pz in pt_z)**2
        )
        return m_in
    return None
def calculos_eventos(df, lista_num, combinaciones_pares, combinaciones_trios, combinaciones_cuartetos):
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

    for i, row in df.iterrows():
        if row['#'] == 0:
            if current_event:
                event_df = pd.DataFrame(current_event)
                no_jets.append(Num_jets(event_df))
                for i in combinaciones_cuartetos:
                    masainv_cuartetos.append(m_inv(event_df, i))
                for i in combinaciones_trios:
                    masainv_trios.append(m_inv(event_df, i))
                for i in lista_num:
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
        for i in lista_num:
            pt.append(pt_part(event_df, i))
            eta.append(eta_part(event_df, i))
            phi.append(phi_part(event_df, i))
        for i in combinaciones_pares:
            masainv.append(m_inv(event_df, i))
            masatrans.append(m_trans(event_df, i))
            deltar.append(Deltar(event_df, i))

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

    for i in lista_num:
        colum.append('Pt' + str(i))
        colum1.append('Eta' + str(i))
        colum2.append('Phi' + str(i))

    for i in combinaciones_pares:
        columpares.append('m_inv' + str(i))
        columpares1.append('m_trans' + str(i))
        columpares2.append('deltaR' + str(i))

    for i in combinaciones_trios:
        columtrios.append('m_inv' + str(i))
    for i in combinaciones_cuartetos:
        columcuartetos.append('m_inv' + str(i))

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
    csv_combined["#_jets"] = no_jets

    return csv_combined
# Aplicar la función a ambos DataFrames
csv_bg = calculos_eventos(filtered_dfbg, lista_num, combinaciones_pares, combinaciones_trios, combinaciones_cuartetos)
csv_bg['Td'] = "b"

csv_sig = calculos_eventos(filtered_dfsg, lista_num, combinaciones_pares, combinaciones_trios, combinaciones_cuartetos)
csv_sig['Td'] = "s"

df_combined = pd.concat([csv_bg, csv_sig], ignore_index=True)

# Mantener la numeración de la primera columna
df_combined.iloc[:, 0] = range(1, len(df_combined) + 1)

# Guardar la base de datos combinada
df_combined.to_csv(Final_name, index=True)

print(f'El archivo fue creado con el nombre: {Final_name}')

#INICIA PROCESO PARA BDT
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
    nombre=input("Inserte el nombre con el que quiere guardar la imagen que contiene la curva roc obtenida para el modelo de entrenamiento")
    nombre= nombre + ".jpg"
    plt.savefig(nombre)

def plot_classifier_distributions(model, test, train, cols, print_params=False, params=None):

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
print('Number of columns: {}'.format(df_shuffled.shape[1]))

print ('\nList of features in dataset:')
for col in df_shuffled.columns:
    print(col)

print('Number of signal events: {}'.format(len(df_shuffled[df_shuffled.Td == 's'])))
print('Number of background events: {}'.format(len(df_shuffled[df_shuffled.Td == 'b'])))
print('Fraction signal: {}'.format(len(df_shuffled[df_shuffled.Td == 's'])/(float)(len(df_shuffled[df_shuffled.Td == 's']) + len(df_shuffled[df_shuffled.Td == 'b']))))

factor=int(input("Se recomienda tener un fraction signal=0.5, a cuanto de sera limitar la cantidad de datos tanto de bg como de signal")) 
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
cols_to_delete = ['Td']
vars_for_train = vars_for_train.drop("Td")
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
test=input("El entrenamiento se realizará preestablecidamente con una proporcion de 0.8 para el entrenamiento, ¿desea cambiarla?Si=1, No=0")
#SIGNAL
if test == "1" :
    size=input("Coloque el valor de la nueva proporcion que tendrá el entrenamiento (anteriormente era 0.8)")
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
manual_params = {
    'colsample_bylevel': 0.8129556523950925,
     'colsample_bynode': 0.6312324405171867,
     'colsample_bytree': 0.6479261529614907,
     'gamma': 6.0528983610080305,
     'learning_rate': 0.1438821307939924,
     #'max_depth': 5,
     #'max_leaves': 15,
     'max_leaves': 15,               
     'max_depth': 5,
     'min_child_weight': 1.385895334160164,
     #'min_child_weight': 1,
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

#Inicia los resultados del modelo 
fig, ax = plot_classifier_distributions(modelv1, test=test, train=train, cols = cols, print_params=False)
#ax.set_title(r'Total sample size $\approx$ '+str(len(train) + len(test))+' optimized')
name_dist=input("Inserte el nombre con que desea guardar la grafica que sera creada como el resultado del clasificador")

plt.savefig(name_dist + '.png')
#plt.savefig('classifier_distribution.pdf')
plt.show()

roc(test_feat[cols], test_lab, train_feat[cols], train_lab, modelv1)

plt.figure(figsize=(16, 12))
print("Se mostrará ahora una grafica que organiza por orden de importancia las variables que utilizo el modelo(F Score)")
xgb.plot_importance(modelv1)
save=input("¿Desea guardar la imagen? si=1, no=0?")
if save == "1":
    name_save=input("Inserte el nombre con el que desea guardarla")
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

