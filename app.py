import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Cargar el dataset desde el mismo directorio
df = pd.read_excel('EJVDIP.xlsx')

# Preprocesamiento: Separar cursos individuales y entrenar modelo SVD
data = []
for index, row in df.iterrows():
    cursos_individuales = eval(row['Cursos'])  # Convertir la cadena de cursos a lista
    for curso in cursos_individuales:  # <== Aquí está la corrección
        data.append((row['Ejecutivo de Venta'], curso, row['VolumenVentas']))

df_cursos = pd.DataFrame(data, columns=['ejecutivo', 'curso', 'ventas'])

# Crear el dataset para Surprise
reader = Reader(rating_scale=(df_cursos['ventas'].min(), df_cursos['ventas'].max()))
surprise_data = Dataset.load_from_df(df_cursos[['ejecutivo', 'curso', 'ventas']], reader)

# Dividir en conjunto de entrenamiento y prueba
trainset, testset = train_test_split(surprise_data, test_size=0.2)

# Entrenar el modelo SVD
algo = SVD()
algo.fit(trainset)

# Evaluar el modelo (opcional, para ver el rendimiento)
predictions = algo.test(testset)
st.write(f"RMSE del modelo: {accuracy.rmse(predictions)}")

# Función para recomendar cursos
def recomendar_cursos_por_rendimiento(ejecutivo_id, top_n=3):
    perfil = df[df['Ejecutivo de Venta'] == ejecutivo_id].iloc[0]
    
    ventas_ejecutivo = perfil['VolumenVentas']
    ventas_media = df['VolumenVentas'].mean()
    
    if ventas_ejecutivo >= ventas_media:
        grupo_comparar = df[df['VolumenVentas'] < ventas_media]
    else:
        grupo_comparar = df[df['VolumenVentas'] >= ventas_media]

    cursos_tomados_grupo = set(grupo_comparar['Cursos'].explode().apply(eval).explode())
    cursos_tomados_ejecutivo = set(eval(perfil['Cursos']))
    cursos_recomendables = cursos_tomados_grupo - cursos_tomados_ejecutivo

    recomendaciones = []
    for curso in cursos_recomendables:
        pred = algo.predict(ejecutivo_id, curso)
        recomendaciones.append((curso, pred.est))
    
    recomendaciones.sort(key=lambda x: x[1], reverse=True)
    return recomendaciones[:top_n]

# Configurar la aplicación en Streamlit
st.title("Recomendación de Cursos para Ejecutivos de Venta")

# Input: ID del Ejecutivo
ejecutivo_id = st.text_input("Introduce el ID del Ejecutivo de Venta")

if ejecutivo_id:
    # Convertir a entero
    ejecutivo_id = int(ejecutivo_id)
    
    # Obtener perfil del ejecutivo
    perfil = df[df['Ejecutivo de Venta'] == ejecutivo_id].iloc[0]
    
    # Mostrar información del ejecutivo
    st.write("### Información del Ejecutivo")
    st.write(f"**Nombre**: {perfil['Ejecutivo de Venta Descripción']}")
    st.write(f"**Sexo**: {perfil['Descripción del Sexo']}")
    st.write(f"**Estado**: {perfil['Division Personal Descripción']}")
    st.write(f"**Ventas**: {perfil['VolumenVentas']}")
    
    # Recomendar cursos
    recomendaciones = recomendar_cursos_por_rendimiento(ejecutivo_id)
    
    st.write("### Cursos Recomendados")
    for curso, score in recomendaciones:
        st.write(f"**Curso**: {curso}, **Score**: {score:.2f}")
    
    # Explicación de la recomendación
    st.write("### Justificación de la Recomendación")
    if perfil['VolumenVentas'] >= df['VolumenVentas'].mean():
        st.write("Estos cursos se recomiendan porque han demostrado mejorar las ventas en ejecutivos con rendimiento inferior.")
    else:
        st.write("Estos cursos se recomiendan porque han sido exitosos en ejecutivos con mejor rendimiento.")
