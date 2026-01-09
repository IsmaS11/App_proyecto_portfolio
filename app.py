import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Predicción de Fallas Industriales",
    page_icon="🏭",
    layout="wide"
)

# --- 1. CARGA DE MODELOS ---
@st.cache_resource
def cargar_modelos():
    try:
        # Asegúrate de que el nombre del archivo sea EXACTAMENTE el que tienes en la carpeta
        return joblib.load('best_models_xgb_mantenimiento.pkl')
    except FileNotFoundError:
        return None

modelos = cargar_modelos()

# Título Principal
st.title("🏭 Panel de Control de Mantenimiento Prescriptivo")
st.markdown("""
Esta herramienta simula las condiciones operativas y calcula la probabilidad de 
fallas específicas basada en modelos de Machine Learning entrenados.
""")

# --- 2. BARRA LATERAL (INPUTS CRUDOS) ---
st.sidebar.header("🎛️ Parámetros de Operación")

def obtener_datos_usuario():
    # Selector de Calidad (Type)
    tipo_letra = st.sidebar.selectbox("Calidad del Producto (Type)", ["L (Low)", "M (Medium)", "H (High)"])
    
    # Mapeo: L=0, M=1, H=3
    mapeo_tipo = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 3}
    tipo_valor = mapeo_tipo[tipo_letra]

    # Sliders
    air_temp = st.sidebar.slider("Temperatura Aire [K] (air_temp_k)", 295.0, 305.0, 300.0)
    process_temp = st.sidebar.slider("Temperatura Proceso [K] (process_temp_k)", 305.0, 315.0, 310.0)
    rpm = st.sidebar.slider("Velocidad Rotación [RPM] (rotational_speed_rpm)", 1100, 2900, 1500)
    torque = st.sidebar.slider("Torque [Nm] (torque_nm)", 3.0, 80.0, 40.0)
    wear = st.sidebar.slider("Desgaste Herramienta [min] (tool_wear_min)", 0, 250, 0)
    
    # DataFrame inicial
    datos_crudos = {
        'type_encoded': tipo_valor, # Ya lo guardamos con el nombre correcto
        'air_temp_k': air_temp,
        'process_temp_k': process_temp,
        'rotational_speed_rpm': rpm,
        'torque_nm': torque,
        'tool_wear_min': wear
    }
    
    return pd.DataFrame(datos_crudos, index=[0])

# Obtenemos el input base
df_input = obtener_datos_usuario()

# --- 3. MOTOR DE INGENIERÍA DE CARACTERÍSTICAS (VISUALIZACIÓN) ---
st.subheader("📊 Variables Calculadas en Tiempo Real")

# Calculamos variables extra SOLO para mostrar al usuario (Ingeniería)
# Hacemos una copia para no afectar la visualización si luego filtramos
df_visual = df_input.copy()
df_visual['temp_delta'] = df_visual['process_temp_k'] - df_visual['air_temp_k']
df_visual['power'] = df_visual['torque_nm'] * df_visual['rotational_speed_rpm'] * (2 * np.pi / 60)
df_visual['wear_torque_product'] = df_visual['tool_wear_min'] * df_visual['torque_nm']

# Mostramos la tabla completa con cálculos
st.dataframe(df_visual.style.format("{:.2f}"))

# --- 4. PANEL DE PREDICCIONES ---
st.divider()
st.subheader("🔍 Diagnóstico de Fallas Específicas")

if st.button("Ejecutar Análisis de Riesgo"):
    if modelos is None:
        st.error("⚠️ Error: No se encontró el archivo .pkl")
    else:
        # 1. Definimos las llaves del diccionario de modelos
        fallas_a_evaluar = [
            'Falla_Desgaste (TWF)',
            'Falla_Calor (HDF)',
            'Falla_Potencia (PWF)',
            'Falla_Sobrecarga (OSF)'
        ]
        
        # 2. PREPARACIÓN DE DATOS (PARCHE CRÍTICO)
        # Definimos las columnas que el modelo "viejo" espera
        columnas_modelo = [
            'air_temp_k', 
            'process_temp_k', 
            'rotational_speed_rpm', 
            'torque_nm', 
            'tool_wear_min', 
            'type_encoded'
        ]
        
        # --- CORRECCIÓN DEL ERROR "NameError" ---
        # Primero creamos la copia
        df_para_modelo = df_input.copy()
        
        # Luego filtramos dejando solo las 6 columnas originales
        df_para_modelo = df_para_modelo[columnas_modelo]

        # Variables para resumen
        hay_falla_general = False
        max_probabilidad = 0.0
        mensaje_falla = ""

        # 3. BUCLE DE PREDICCIÓN
        cols = st.columns(len(fallas_a_evaluar))
        
        for i, nombre_falla_key in enumerate(fallas_a_evaluar):
            with cols[i]:
                try:
                    modelo_actual = modelos[nombre_falla_key]
                    
                    # Predecimos usando el DataFrame limpio (6 columnas)
                    probabilidad = modelo_actual.predict_proba(df_para_modelo)[0][1]
                    
                    # --- CORRECCIÓN DEL ERROR "probability" vs "probabilidad" ---
                    
                    # Lógica de resumen
                    if probabilidad > 0.5:
                        hay_falla_general = True
                        mensaje_falla = nombre_falla_key
                    
                    if probabilidad > max_probabilidad:
                        max_probabilidad = probabilidad
                    
                    # Visualización
                    titulo_corto = nombre_falla_key.split('(')[0].strip()
                    st.metric(label=titulo_corto, value=f"{probabilidad:.1%}")
                    
                    if probabilidad > 0.5:
                        st.error("🚨 FALLA")
                    else:
                        st.success("✅ OK")
                        
                except KeyError:
                    st.warning(f"Falta modelo: {nombre_falla_key}")
                except Exception as e:
                    st.error(f"Error: {e}")

        # 4. RESUMEN FINAL
        st.divider()
        if hay_falla_general:
            st.error(f"🚨 ALARMA DE PLANTA: Se recomienda detener la máquina. Causa probable: {mensaje_falla}")
        else:
            st.success(f"✅ MÁQUINA OPERATIVA: Ningún modelo detecta riesgo crítico (Máx riesgo: {max_probabilidad:.1%})")

st.markdown("---")
st.caption("Sistema de Mantenimiento Inteligente - Portfolio de Ingeniería")
st.caption("Desarrollado por Ismael Benjamin Sosa - Ingeniero Industrial & Data Analyst")