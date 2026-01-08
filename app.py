import streamlit as st
import pandas as pd
import numpy as np

# 1. Configuración de la página (Título y diseño)
st.set_page_config(page_title="Simulador de Mantenimiento", page_icon="⚙️")

st.title("🏭 Centro de Control de Mantenimiento Predictivo")
st.markdown("### Simulador de Probabilidad de Falla en Tiempo Real")

# 2. Sidebar (La barra lateral para los controles del Ingeniero/Operador)
st.sidebar.header("Parámetros del Proceso")

def obtener_input_usuario():
    # Deslizadores para las variables físicas (Simulando sensores)
    air_temp = st.sidebar.slider("Temperatura Aire [K]", 295.0, 305.0, 300.0)
    process_temp = st.sidebar.slider("Temperatura Proceso [K]", 305.0, 315.0, 310.0)
    rpm = st.sidebar.slider("Velocidad de Rotación [RPM]", 1100, 2900, 1500)
    torque = st.sidebar.slider("Torque [Nm]", 3.0, 80.0, 40.0)
    tool_wear = st.sidebar.slider("Desgaste Herramienta [min]", 0, 250, 0)
    
    # Guardamos los datos en un diccionario
    datos_usuario = {
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }
    features = pd.DataFrame(datos_usuario, index=[0])
    return features

# Capturamos lo que mueve el usuario
input_df = obtener_input_usuario()

# 3. Panel Principal (Resultados)
st.subheader("Estado Actual de la Máquina")

# Mostramos los datos que eligió el usuario
st.table(input_df)

# --- AQUÍ CONECTARÍAS TU MODELO REAL ---
# Por ahora, simulamos una predicción con lógica simple para que veas la app funcionar
# (Más adelante cargaremos tu modelo 'random forest' real aquí)
if st.button('Ejecutar Diagnóstico'):
    
    # Simulación simple: Si Torque * RPM es muy alto, falla
    potencia_ficticia = input_df['Torque [Nm]'] * input_df['Rotational speed [rpm]']
    probabilidad_falla = 0 # Inicializamos
    
    if potencia_ficticia.iloc[0] > 180000 or input_df['Tool wear [min]'].iloc[0] > 200:
        st.error("🚨 ALERTA CRÍTICA: Alta probabilidad de falla inminente.")
        st.metric(label="Probabilidad de Falla", value="85%", delta="Alto Riesgo")
        st.warning("Recomendación: Detener línea y revisar herramienta.")
    else:
        st.success("✅ ESTADO NORMAL: Operación segura.")
        st.metric(label="Probabilidad de Falla", value="2%", delta="-Bajo Riesgo")

st.markdown("---")
st.caption("Desarrollado por Ismael Benjamin Sosa - Ingeniero Industrial & Data Analyst")