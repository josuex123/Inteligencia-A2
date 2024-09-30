import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import tkinter as tk
from tkinter import messagebox

# Diccionario de datos simulado
datos = {
    'Tamaño (m2)': [70, 85, 120, 150, 110, 65, 200, 175, 95, 140],
    'Habitaciones': [3, 4, 4, 5, 4, 2, 6, 5, 3, 4],
    'Distancia al centro (km)': [5, 3, 8, 12, 7, 4, 10, 15, 6, 9],
    'Precio alto (1: Sí, 0: No)': [0, 1, 1, 1, 1, 0, 1, 1, 0, 1]
}

# Convertir el diccionario a un DataFrame de pandas
df = pd.DataFrame(datos)

# Separar las características (X) del objetivo (y)
X = df[['Tamaño (m2)', 'Habitaciones', 'Distancia al centro (km)']]
y = df['Precio alto (1: Sí, 0: No)']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear el clasificador de árbol de decisión
clasificador = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo con los datos de entrenamiento
clasificador.fit(X_entrenamiento, y_entrenamiento)

# Predecir con los datos de prueba
y_prediccion = clasificador.predict(X_prueba)

# Evaluar el modelo
precision = accuracy_score(y_prueba, y_prediccion)
print(f'Precisión: {precision * 100:.2f}%')

# Visualizar el árbol de decisión
plt.figure(figsize=(12, 8))
tree.plot_tree(clasificador, 
               feature_names=X.columns, 
               class_names=['No', 'Sí'], 
               filled=True, 
               impurity=False, 
               rounded=True, 
               proportion=False,
               fontsize=14) 

for text_obj in plt.gcf().findobj(plt.Text):
    if 'samples' in text_obj.get_text():
        text_obj.set_text(text_obj.get_text().replace('samples', 'Casos'))
    if 'value' in text_obj.get_text():
        text_obj.set_text(text_obj.get_text().replace('value', 'Conteo'))
    if 'class' in text_obj.get_text():
        text_obj.set_text(text_obj.get_text().replace('class', 'Decisión'))

# Etiquetas explicativas
plt.title("Árbol de Decisión para Predecir Precio de Casas", fontsize=14)
plt.text(0.5, 0.95, "Casos = número de ejemplos en el nodo", fontsize=12, ha="center", va="center")
plt.text(0.5, 0.9, "Conteo = cuántos ejemplos en cada categoría", fontsize=12, ha="center", va="center")
plt.text(0.5, 0.85, "Decisión = predicción del nodo", fontsize=12, ha="center", va="center")

plt.show()

# Ejmplos de casas
nuevas_casas = pd.DataFrame(np.array([[100, 3, 8], [80, 4, 5]]), columns=['Tamaño (m2)', 'Habitaciones', 'Distancia al centro (km)'])
predicciones = clasificador.predict(nuevas_casas)

# función para mostrar las predicciones 
def mostrar_predicciones():
    pred_texto = f'Predicciones para nuevas casas: {predicciones}'
    messagebox.showinfo("Predicciones", pred_texto)

ventana = tk.Tk()
ventana.title("Predicciones de precios de casas")
ventana.geometry("300x200")

boton_mostrar = tk.Button(ventana, text="Mostrar predicciones", command=mostrar_predicciones)
boton_mostrar.pack(pady=20)

ventana.mainloop()
