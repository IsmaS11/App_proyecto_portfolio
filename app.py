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
        # Asegúrate de que este archivo contenga el diccionario con las claves correctas
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
    # Selector de Calidad (Type) con el mapeo que indicaste
    tipo_letra = st.sidebar.selectbox("Calidad del Producto (Type)", ["L (Low)", "M (Medium)", "H (High)"])
    
    # Mapeo específico solicitado: L=0, M=1, H=3
    mapeo_tipo = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 3}
    tipo_valor = mapeo_tipo[tipo_letra]

    # Sliders con los rangos y nombres solicitados
    air_temp = st.sidebar.slider("Temperatura Aire [K] (air_temp_k)", 295.0, 305.0, 300.0)
    process_temp = st.sidebar.slider("Temperatura Proceso [K] (process_temp_k)", 305.0, 315.0, 310.0)
    rpm = st.sidebar.slider("Velocidad Rotación [RPM] (rotational_speed_rpm)", 1100, 2900, 1500)
    torque = st.sidebar.slider("Torque [Nm] (torque_nm)", 3.0, 80.0, 40.0)
    wear = st.sidebar.slider("Desgaste Herramienta [min] (tool_wear_min)", 0, 250, 0)
    
    # Creamos el DataFrame inicial con los datos crudos
    datos_crudos = {
        'type_encoded': tipo_valor,
        'air_temp_k': air_temp,
        'process_temp_k': process_temp,
        'rotational_speed_rpm': rpm,
        'torque_nm': torque,
        'tool_wear_min': wear
    }
    
    return pd.DataFrame(datos_crudos, index=[0])

# Obtenemos el input base
df_input = obtener_datos_usuario()

# --- 3. MOTOR DE INGENIERÍA DE CARACTERÍSTICAS (TRANSFORMACIONES) ---
# Aquí aplicamos EXACTAMENTE las mismas fórmulas que usaste en el entrenamiento
st.subheader("📊 Variables Calculadas en Tiempo Real")

# A. Delta de Temperatura
df_input['temp_delta'] = df_input['process_temp_k'] - df_input['air_temp_k']

# B. Potencia (Power)
# Nota: np.pi requiere importar numpy
df_input['power'] = df_input['torque_nm'] * df_input['rotational_speed_rpm'] * (2 * np.pi / 60)

# C. Producto Desgaste x Torque
df_input['wear_torque_product'] = df_input['tool_wear_min'] * df_input['torque_nm']

# Mostramos al usuario lo que la IA está "viendo" (incluyendo las calculadas)
st.dataframe(df_input.style.format("{:.2f}"))

# --- 4. PANEL DE PREDICCIONES ---
st.divider()
st.subheader("🔍 Diagnóstico de Fallas Específicas")

if st.button("Ejecutar Análisis de Riesgo"):
    if modelos is None:
        st.error("⚠️ Error: No se encontró el archivo .pkl")
    else:
        # 1. DEFINIMOS LAS LLAVES EXACTAS QUE TIENE TU DICCIONARIO
        # (Copiadas tal cual salieron de tu script de inspección)
        fallas_a_evaluar = [
            'Falla_Desgaste (TWF)',
            'Falla_Calor (HDF)',
            'Falla_Potencia (PWF)',
            'Falla_Sobrecarga (OSF)'
        ]
        
        # Variables para calcular el estado general "virtual"
        hay_falla_general = False
        max_probabilidad = 0.0
        mensaje_falla = ""

        # 2. CREAMOS LAS COLUMNAS PARA MOSTRAR LOS RESULTADOS
        cols = st.columns(len(fallas_a_evaluar))
        
        for i, nombre_falla_key in enumerate(fallas_a_evaluar):
            with cols[i]:
                try:
                    # Obtenemos el modelo usando la LLAVE EXACTA
                    modelo_actual = modelos[nombre_falla_key]
                    
                    # Predecimos
                    probabilidad = modelo_actual.predict_proba(df_input)[0][1]
                    
                    # Actualizamos el estado general (Lógica: Si falla uno, falla la máquina)
                    if probability > 0.5:
                        hay_falla_general = True
                        mensaje_falla = nombre_falla_key
                    
                    if probability > max_probabilidad:
                        max_probabilidad = probability
                    
                    # VISUALIZACIÓN INDIVIDUAL
                    # Usamos un nombre más corto para el título visual (quitamos el paréntesis si quieres)
                    titulo_corto = nombre_falla_key.split('(')[0].strip()
                    st.metric(label=titulo_corto, value=f"{probabilidad:.1%}")
                    
                    if probability > 0.5:
                        st.error("🚨 FALLA")
                    else:
                        st.success("✅ OK")
                        
                except KeyError:
                    st.warning(f"Clave '{nombre_falla_key}' no encontrada.")
                except Exception as e:
                    st.error(f"Error: {e}")

        # 3. RESUMEN GENERAL (INFERIDO)
        st.divider()
        if hay_falla_general:
            st.error(f"🚨 ALARMA DE PLANTA: Se recomienda detener la máquina. Causa probable: {mensaje_falla}")
        else:
            st.success(f"✅ MÁQUINA OPERATIVA: Ningún modelo detecta riesgo crítico (Máx riesgo: {max_probabilidad:.1%})")

st.markdown("---")
st.caption("Sistema de Mantenimiento Inteligente - Portfolio de Ingeniería")
st.caption("Desarrollado por Ismael Benjamin Sosa - Ingeniero Industrial & Data Analyst")