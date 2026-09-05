# Copyright (c) 2026 Nombre del autor
# SPDX-License-Identifier: MIT
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go


st.set_page_config(layout="wide")
st.title("Visualización matemática de datos")

DATA_FOLDER = "data"

MAIN_WIDTH = 300
PARALLEL_WIDTH = 600
HEIGHT = 300

# ============================================================
# FUNCIONES MATEMATICAS
# ============================================================

def estandarizar(s):
    if s.empty:
        return s
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return s * 0
    return (s - s.mean()) / std

def dominio_expandido(s):
    minimo = s.min()
    maximo = s.max()
    rango = maximo - minimo
    if rango == 0:
        rango = 1
    margen = rango * 0.05
    return [minimo - margen, maximo + margen]

def dominio_simetrico(df, x, y):

    m = max(
        abs(df[x].min()),
        abs(df[x].max()),
        abs(df[y].min()),
        abs(df[y].max())
    )

    if m == 0:
        m = 1

    return [-m, m]

# ============================================================
# CIRCUNFERENCIA UNIDAD
# ============================================================

def semicircunferencias(color):

    t1 = np.linspace(0, np.pi, 300)
    t2 = np.linspace(np.pi, 2*np.pi, 300)

    df1 = pd.DataFrame({
        "x": np.cos(t1),
        "y": np.sin(t1)
    })

    df2 = pd.DataFrame({
        "x": np.cos(t2),
        "y": np.sin(t2)
    })

    return (
        alt.Chart(df1).mark_line(color=color).encode(x="x:Q", y="y:Q")
        +
        alt.Chart(df2).mark_line(color=color).encode(x="x:Q", y="y:Q")
    )

# ============================================================
# VECTOR CON FLECHA
# ============================================================

def vector_con_flecha(x, y):

    linea_df = pd.DataFrame({
        "x":[0,x],
        "y":[0,y]
    })

    punta_df = pd.DataFrame({
        "x":[x],
        "y":[y],
        "angle":[np.degrees(np.arctan2(y,x))]
    })

    linea = alt.Chart(linea_df).mark_line(strokeWidth=3).encode(
        x="x:Q",
        y="y:Q"
    )

    punta = alt.Chart(punta_df).mark_point(
        shape="triangle",
        size=150
    ).encode(
        x="x:Q",
        y="y:Q",
        angle="angle:Q"
    )

    return linea + punta

# ============================================================
# COORDENADAS PARALELAS RAPIDAS
# ============================================================

def paralelas_rapidas(df, columnas):

    valores = df[columnas].values

    n, k = valores.shape

    x = np.tile(columnas, n)
    y = valores.reshape(-1)

    ids = np.repeat(df["__id__"].values, k)

    df_long = pd.DataFrame({
        "Variable": x,
        "Valor": y,
        "__id__": ids
    })

    df_long["Variable"] = df_long["Variable"].astype(str)
    df_long["Valor"] = pd.to_numeric(df_long["Valor"], errors="coerce")
    df_long["__id__"] = df_long["__id__"].astype(str)

    return df_long

# ============================================================
# ESTADISTICAS
# ============================================================

def estadisticas_originales(df, cols):

    datos = {}

    for c in cols:

        if c in df.columns:

            s = pd.to_numeric(df[c], errors="coerce").dropna()

            if not s.empty:

                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)

                datos[c] = {
                    "Media": s.mean(),
                    "Mediana": s.median(),
                    "Desv_Tipica": s.std(ddof=0),
                    "Minimo": s.min(),
                    "P25": q1,
                    "P50": s.quantile(0.5),
                    "P75": q3,
                    "Maximo": s.max(),
                    "Rango_Intercuartil": q3-q1,
                    "Coeficiente de asimetría": s.skew(),
                    "Curtosis":s.kurt()
                }

    return pd.DataFrame(datos)

# ============================================================
# cREAR GRÁFICO VACÍO
# ============================================================

def empty_chart(width, height, mensaje="Sin datos"):
    return alt.Chart(
        pd.DataFrame({"x":[0], "y":[0]})
    ).mark_text(
        size=14,
        color="gray"
    ).encode(
        text=alt.value(mensaje)
    ).properties(
        width=width,
        height=height
    )

# ============================================================
# SELECCIÓN POR BRUSH
# ============================================================

def filtrar_por_brush(df, brush_state, x_field, y_field):

    if brush_state is None:
        return pd.Series([False]*len(df), index=df.index)

    if "x" not in brush_state or "y" not in brush_state:
        return pd.Series([False]*len(df), index=df.index)

    x0, x1 = brush_state["x"]
    y0, y1 = brush_state["y"]

    return (
        (df[x_field] >= x0) & (df[x_field] <= x1) &
        (df[y_field] >= y0) & (df[y_field] <= y1)
    )

# ============================================================
# PIPELINE DE TRANSFORMACIONES
# ============================================================

def construir_df_render(df_base, modo, col_x, col_y, columnas_mv, sistema_pesos):

    df = df_base.copy()

    if df.empty:
        return df, col_x, col_y, None, []

    x_field = col_x
    y_field = col_y
    dominio = None
    vectores = []

    # --------------------------------------------------------
    # ESTANDARIZADA
    # --------------------------------------------------------
    if modo == "Representación estandarizada":

        df["_x"] = estandarizar(df[col_x])
        df["_y"] = estandarizar(df[col_y])

        dominio = dominio_simetrico(df, x_field, y_field)

    # --------------------------------------------------------
    # MULTIVARIABLE
    # --------------------------------------------------------
    elif modo == "Representación multivariable":

        df_z = df[columnas_mv].apply(estandarizar)

        x_proj = 0
        y_proj = 0

        for c in df_z.columns:

            w1 = st.session_state.get(f"w1_{c}", 0.0)
            w2 = st.session_state.get(f"w2_{c}", 0.0)

            if sistema_pesos == "Cartesiano":
                vx, vy = w1, w2
            else:
                vx = w1 * np.cos(w2 * np.pi)
                vy = w1 * np.sin(w2 * np.pi)

            x_proj += df_z[c] * vx
            y_proj += df_z[c] * vy

            vectores.append((vx, vy))

        df["_x"] = x_proj
        df["_y"] = y_proj

        dominio = dominio_simetrico(df, x_field, y_field)

    elif "X_proj" in df.columns and "Y_proj" in df.columns:

        # caso PCA / LDA
        df["_x"] = df["X_proj"]
        df["_y"] = df["Y_proj"]

        dominio = dominio_simetrico(df, x_field, y_field)
    
    # --------------------------------------------------------
    # REAL / EXPANDIDA
    # --------------------------------------------------------
    else:
        df["_x"] = df[col_x]
        df["_y"] = df[col_y]

    x_field = "_x"
    y_field = "_y"

    return df, x_field, y_field, dominio, vectores


# ============================================================
# CARGA ARCHIVOS
# ============================================================

if not os.path.exists(DATA_FOLDER):
    st.sidebar.error("No existe carpeta data")
    st.stop()

archivos = sorted([
    f for f in os.listdir(DATA_FOLDER)
    if f.endswith(".data")
])

if not archivos:
    st.sidebar.error("No hay archivos .data")
    st.stop()

archivo = st.sidebar.selectbox(
    "Archivo",
    archivos,
    index=0
)

# DETECTAR CAMBIO DE ARCHIVO

archivo_prev = st.session_state.get("archivo_actual", None)

if archivo_prev is not None and archivo != archivo_prev:

    # RESET COMPLETO
    for k in list(st.session_state.keys()):
        del st.session_state[k]

    # guardar nuevo archivo
    st.session_state.archivo_actual = archivo

    st.rerun()

# INICIALIZAR SOLO SI NO EXISTE
if "archivo_actual" not in st.session_state:
    st.session_state.archivo_actual = archivo

df_original = pd.read_csv(
    os.path.join(DATA_FOLDER, archivo),
    sep=";"
)

df_original["__id__"] = df_original.index.astype(str)
df_original["_id_order"] = np.arange(len(df_original))

st.session_state["proyeccion_valida"] = True
# ============================================================
# INICIALIZAR CHECKBOX
# ============================================================

if "df_editado" not in st.session_state:
    df_tmp = df_original.copy()
    df_tmp["Seleccionar"] = True
    st.session_state.df_editado = df_tmp

# si cambia dataset, reiniciar
if len(st.session_state.df_editado) != len(df_original):
    df_tmp = df_original.copy()
    df_tmp["Seleccionar"] = True
    st.session_state.df_editado = df_tmp

# SELECCIÓN GLOBAL PLOTLY
if "plotly_selected_ids" not in st.session_state:
    st.session_state.plotly_selected_ids = []

columnas = df_original.columns.tolist()
columnas_num = df_original.select_dtypes(include="number").columns.tolist()

hay_numericas = len(columnas_num) > 0

for c in columnas_num:

    if f"w1_store_{c}" not in st.session_state:
        st.session_state[f"w1_store_{c}"] = 0.0

    if f"w2_store_{c}" not in st.session_state:
        st.session_state[f"w2_store_{c}"] = 0.0
# ============================================================
# SIDEBAR
# ============================================================

tabs = st.sidebar.tabs(["1","2","3"])

with tabs[0]:

    st.write("Representación")
    modo = st.selectbox(
        "Modo",
        [
            "Representación real",
            "Representación expandida",
            "Representación estandarizada",
            "Representación multivariable"
        ]
    )

    tipo = st.selectbox(
        "Tipo grafico",
        ["Dispersión"]
    )

    if hay_numericas:
        col_x = st.selectbox("Eje X", columnas_num, index=0)
        col_y = st.selectbox("Eje Y", columnas_num, index=0)
    else:
        col_x = None
        col_y = None
        st.warning("No hay columnas numéricas. Mostrando gráfico vacío.")
    
with tabs[1]:

    st.write("Visualización")
    st.selectbox("Columna de color", columnas, key="col_color")
    st.selectbox(
        "Esquema de color",
        [
            "Automático",
            "viridis","plasma","inferno","magma",
            "cividis","turbo","spectral",
            "category10","tableau10","tableau20",
        ],
        key="esquema_color",
    )
    st.color_picker("Color circunferencia unidad", key="color_circunferencia")
    st.checkbox("Intercambiar tabla estadísticas", key="stats_transpose")
    
    # grosor lineas paralelas
    stroke_paralelas = st.slider(
        "Grosor lineas paralelas",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1
    )

with tabs[2]:

    st.write("Multivariable")
    # filtrar y poner solo columnas con letras
    columnas_defecto = [
        c for c in columnas_num
        if len(str(c)) > 0 and str(c)[0].isalpha()
    ]
    columnas_mv = st.multiselect(
        "Dimensiones",
        columnas_num,
        default=columnas_defecto[:min(5,len(columnas_num))],
        key="columnas_mv"
    )
    for c in columnas_mv:

        if f"w1_store_{c}" not in st.session_state:
            st.session_state[f"w1_store_{c}"] = 0.0

        if f"w2_store_{c}" not in st.session_state:
            st.session_state[f"w2_store_{c}"] = 0.0
    
    modo_pesos = st.selectbox(
        "Metodo multivariable",
        [
            "Coordenadas estrella",
            "Análisis de Componentes Principales (PCA)",
            "Análisis de Discriminantes Lineales (LDA)"
        ],
        key="modo_pesos"
    )
    # valor por defecto
    sistema_pesos = "Cartesiano"
    if st.session_state.modo_pesos == "Coordenadas estrella":
    # mostrar pesos
        sistema_pesos = st.radio(
            "Sistema pesos",
            ["Cartesiano","Polar"]
        )

        # ============================================================
        # DIMENSION ACTIVA
        # ============================================================

        if (
            "dimension_activa" not in st.session_state
            and
            len(columnas_mv) > 0
        ):
            st.session_state.dimension_activa = columnas_mv[0]
        if len(columnas_mv) > 0:

            dimension_activa = st.radio(
                "Dimension activa",
                columnas_mv,
                key="dimension_activa"
            )

        # ============================================================
        # PESOS
        # ============================================================

        for c in columnas_mv:

            col1, col2 = st.columns(2)

            col1.number_input(
                f"Peso1 {c}",
                key=f"w1_widget_{c}",
                value=float(st.session_state[f"w1_store_{c}"])
            )

            col2.number_input(
                f"Peso2 {c}",
                key=f"w2_widget_{c}",
                value=float(st.session_state[f"w2_store_{c}"])
            )
            st.session_state[f"w1_store_{c}"] = (
                st.session_state[f"w1_widget_{c}"]
            )

            st.session_state[f"w2_store_{c}"] = (
                st.session_state[f"w2_widget_{c}"]
            )
    # ============================================
    # VISUALIZACIÓN PCA
    # ============================================

    if st.session_state.get("modo_pesos") == "Análisis de Componentes Principales (PCA)":

        df_vec = st.session_state.get("pca_vectors", None)
        var_exp = st.session_state.get("pca_variance", None)

        if df_vec is not None:

            st.markdown("### Vectores PCA (loadings)")
            st.dataframe(df_vec.style.format("{:.3f}"))

        if var_exp is not None:

            st.markdown("### Varianza explicada")

            df_var = pd.DataFrame({
                "Componente": ["PC1", "PC2"],
                "Varianza": var_exp[:2]
            })

            st.dataframe(df_var.style.format({"Varianza": "{:.3f}"}))

# ============================================================
# GENERACION GRAFICO
# ============================================================

inicio = time.time()


df_z_global = df_original.copy()

for col in columnas_num:
    df_z_global[col] = estandarizar(df_original[col])


df_base = st.session_state.df_editado.copy()

df_render, x_field, y_field, dominio, vectores = construir_df_render(
    df_base,
    modo,
    col_x,
    col_y,
    columnas_mv,
    sistema_pesos
)
chart_total=0
if not hay_numericas:

    empty_df = pd.DataFrame({"x": [], "y": []})

    chart_total = alt.Chart(empty_df).mark_circle().encode(
        x=alt.X("x:Q", title="Sin datos"),
        y=alt.Y("y:Q", title="Sin datos")
    ).properties(
        width=MAIN_WIDTH,
        height=HEIGHT
    )
    chart_total = empty_chart(MAIN_WIDTH, HEIGHT, "Ejes no numéricos")
    
elif df_render.empty:
    chart_total = empty_chart(MAIN_WIDTH, HEIGHT, "Sin datos seleccionados")
else:

    x_field = col_x
    y_field = col_y
    vectores = []
    dominio = None

    if modo == "Representación estandarizada":

        df_render[f"{x_field}_z"] = df_z_global.loc[df_render.index, x_field]
        df_render[f"{y_field}_z"] = df_z_global.loc[df_render.index, y_field]

        x_field = f"{x_field}_z"
        y_field = f"{y_field}_z"

        dominio = dominio_simetrico(df_render, x_field, y_field)

    elif modo == "Representación multivariable":

        df_base = df_original.copy()

        # ---------------------------------------
        # PCA
        # ---------------------------------------
        if st.session_state.modo_pesos == "Análisis de Componentes Principales (PCA)":

            df_z = df_base[columnas_mv].apply(estandarizar).dropna()

            if df_z.empty or len(df_z) < 2:
                chart_total = empty_chart(MAIN_WIDTH, HEIGHT, "Datos insuficientes para PCA")
            else:
                pca = PCA(n_components=2)
                componentes = pca.fit_transform(df_z)

                vectores = []

                loadings = pca.components_.T

                for i, var in enumerate(columnas_mv):

                    vx = loadings[i, 0]
                    vy = loadings[i, 1]

                    vectores.append((vx, vy))
                
                loadings = pca.components_.T

                df_pca_vectors = pd.DataFrame(
                    loadings,
                    index=columnas_mv,
                    columns=["PC1", "PC2"]
                )

                # guardar en session_state para usar en sidebar
                st.session_state["pca_vectors"] = df_pca_vectors

                # varianza explicada (muy útil)
                st.session_state["pca_variance"] = pca.explained_variance_ratio_
                
                df_base.loc[df_z.index, "X_proj"] = componentes[:, 0]
                df_base.loc[df_z.index, "Y_proj"] = componentes[:, 1]

                df_render["X_proj"] = df_base.loc[df_render.index, "X_proj"]
                df_render["Y_proj"] = df_base.loc[df_render.index, "Y_proj"]
                
                x_field = "X_proj"
                y_field = "Y_proj"

                dominio = dominio_simetrico(df_render, x_field, y_field)

        # ---------------------------------------
        # LDA
        # ---------------------------------------
        elif st.session_state.modo_pesos == "Análisis de Discriminantes Lineales (LDA)":
            try:
                col_clase = st.session_state.col_color

                if col_clase not in df_base.columns:
                    chart_total = empty_chart(MAIN_WIDTH, HEIGHT, "Columna de clase inválida")

                else:

                    # ============================================================
                    # DATOS BASE
                    # ============================================================

                    df_z = df_base[columnas_mv].apply(estandarizar)
                    df_z["__class__"] = df_base[col_clase]

                    df_z = df_z.dropna()

                    n_clases = df_z["__class__"].nunique()

                    if n_clases < 2:
                        chart_total = empty_chart(MAIN_WIDTH, HEIGHT, "LDA requiere ≥ 2 clases")

                    else:

                        # ============================================================
                        # LDA
                        # ============================================================

                        n_comp = min(2, n_clases - 1)

                        lda = LinearDiscriminantAnalysis(n_components=n_comp)

                        X = df_z[columnas_mv]
                        y_raw = df_z["__class__"]

                        # ------------------------------------------------------------
                        # asegurar variable categórica
                        # ------------------------------------------------------------
                        y_num = pd.to_numeric(y_raw, errors="coerce")

                        if y_num.notna().all():
                            y = pd.qcut(
                                y_num,
                                q=min(4, y_num.nunique()),
                                duplicates="drop"
                            ).astype(str)
                        else:
                            y = y_raw.astype(str)

                        componentes = lda.fit_transform(X, y)

                        # ============================================================
                        # PROYECCION
                        # ============================================================

                        df_base["X_proj"] = 0.0
                        df_base["Y_proj"] = 0.0

                        df_base.loc[df_z.index, "X_proj"] = componentes[:, 0]

                        if componentes.shape[1] >= 2:
                            df_base.loc[df_z.index, "Y_proj"] = componentes[:, 1]
                        else:
                            df_base.loc[df_z.index, "Y_proj"] = 0.0
                            
                        df_render["X_proj"] = df_base.loc[df_render.index, "X_proj"]
                        df_render["Y_proj"] = df_base.loc[df_render.index, "Y_proj"]

                        x_field = "X_proj"
                        y_field = "Y_proj"

                        dominio = dominio_simetrico(df_render, x_field, y_field)

                        # ============================================================
                        # TABLA LDA (SIDEBAR)
                        # ============================================================

                        try:
                            coef = lda.coef_

                            df_lda = pd.DataFrame(
                                coef.T,
                                index=columnas_mv,
                                columns=[f"LD{i+1}" for i in range(coef.shape[0])]
                            )

                            st.sidebar.subheader("Vectores LDA")
                            st.sidebar.dataframe(
                                df_lda.style.format("{:.3f}")
                            )

                        except Exception as e:
                            st.sidebar.warning(f"No se puede mostrar LDA: {e}")

                        # ============================================================
                        # VECTORES PARA DIBUJAR
                        # ============================================================

                        vectores = []

                        try:
                            coef = lda.coef_

                            for i in range(min(2, coef.shape[0])):

                                v = coef[i]

                                # normalizar (solo visual)
                                norm = np.linalg.norm(v)
                                if norm == 0:
                                    continue

                                v_norm = v / norm

                                # tomar primeras dimensiones para dibujar
                                vx = v_norm[0] if len(v_norm) > 0 else 0
                                vy = v_norm[1] if len(v_norm) > 1 else 0

                                vectores.append((vx, vy))

                        except Exception:
                            vectores = []
            except Exception:
                st.session_state["proyeccion_valida"] = False
                # intentar mostrar excepción
                chart_total = empty_chart(
                    MAIN_WIDTH,
                    HEIGHT,
                    "No se ha podido calcular la proyección LDA"
                )

                vectores = [] 
        # ---------------------------------------
        # COORDENADAS ESTRELLA
        # ---------------------------------------
        else:

            df_z = df_z_global[columnas_mv]

            x_proj = 0
            y_proj = 0

            for c in df_z.columns:

                w1 = st.session_state.get(f"w1_store_{c}",0.0)
                w2 = st.session_state.get(f"w2_store_{c}",0.0)

                if sistema_pesos == "Cartesiano":

                    vx = w1
                    vy = w2

                else:

                    vx = w1*np.cos(w2*np.pi)
                    vy = w1*np.sin(w2*np.pi)

                x_proj += df_z.loc[df_render.index, c] * vx
                y_proj += df_z.loc[df_render.index, c] * vy

                vectores.append((vx,vy))

            df_render["X_proj"] = x_proj
            df_render["Y_proj"] = y_proj

            x_field = "X_proj"
            y_field = "Y_proj"

            dominio = dominio_simetrico(df_render,x_field,y_field)

    elif modo == "Representación expandida":

        dominio_x = dominio_expandido(df_render[x_field])
        dominio_y = dominio_expandido(df_render[y_field])

    # ============================================================
    # PERMUTACION ALEATORIA VISUAL + FILTRADO
    # ============================================================

    # Para plotly
    df_render_plotly = df_render.copy()
    
    df_render = df_render[df_render["Seleccionar"]].copy()

    # ORDEN VISUAL ALEATORIO GLOBAL (FUNCIONA DE VERDAD)
    df_render["_orden_visual"] = np.random.permutation(len(df_render))

    df_render = df_render.sort_values(
        "_orden_visual"
    )


    col_color = st.session_state.col_color

    # detectar tipo automáticamente
    if col_color in columnas_num:
        tipo = "Q"   # cuantitativo
    else:
        tipo = "N"   # categórico

    campo_color = f"{col_color}:{tipo}"

    # construir escala
    if st.session_state.esquema_color == "Automático":
        color_encoding = alt.Color(campo_color)
    else:
        color_encoding = alt.Color(
            campo_color,
            scale=alt.Scale(scheme=st.session_state.esquema_color)
        )


# ============================================================
# SELECCIONES INTERACTIVAS
# ============================================================

    hover = alt.selection_single(
        fields=["__id__"],
        nearest=True,
        on="mouseover",
        empty="none",
        clear="mouseout"
    )

    brush = alt.selection_interval(
        encodings=["x","y"],
        name="brush_sel"
    )

# ============================================================
# GRAFICO PRINCIPAL
# ============================================================

    if chart_total==0:
        if dominio:

            x_axis = alt.X(
                f"{x_field}:Q",
                scale=alt.Scale(domain=dominio, nice=False)
            )

            y_axis = alt.Y(
                f"{y_field}:Q",
                scale=alt.Scale(domain=dominio, nice=False)
            )

        elif modo == "Representación expandida":

            x_axis = alt.X(
                f"{x_field}:Q",
                scale=alt.Scale(domain=dominio_x)
            )

            y_axis = alt.Y(
                f"{y_field}:Q",
                scale=alt.Scale(domain=dominio_y)
            )

        else:

            x_axis = alt.X(f"{x_field}:Q")
            y_axis = alt.Y(f"{y_field}:Q")

    # ------------------------------------------------------------
    # BASE SCATTER
    # ------------------------------------------------------------

        chart = alt.Chart(df_render).mark_circle(
            size=70
        ).encode(
            x=x_axis,
            y=y_axis,
            detail="__id__:N",
            color=alt.condition(brush, color_encoding, alt.value("lightgray")),
            tooltip=["__id__"] + columnas
        )
    # ------------------------------------------------------------
    # HIGHLIGHT (brush)
    # ------------------------------------------------------------
        # selected_scatter = alt.Chart(df_render).mark_circle(size=70).encode(
        #     x=x_axis,
        #     y=y_axis,
        #     color=color_encoding
        # ).transform_filter(brush)

        selected_scatter = alt.Chart(df_render).mark_circle(
            size=70
        ).encode(
            x=x_axis,
            y=y_axis,
            detail="__id__:N",
            color=color_encoding,
            tooltip=["__id__"] + columnas
        ).transform_filter(brush)

    # ------------------------------------------------------------
    # HIGHLIGHT (hover)
    # ------------------------------------------------------------

        highlight_scatter = alt.Chart(df_render).mark_circle(
            size=200,
            stroke="black",
            strokeWidth=2
        ).encode(
            x=x_axis,
            y=y_axis,
            color=alt.condition(brush, color_encoding, alt.value("lightgray")),
            tooltip=["__id__"] + columnas
        ).transform_filter(hover)

        chart_total = chart + highlight_scatter

    # ------------------------------------------------------------
    # CIRCUNFERENCIA
    # ------------------------------------------------------------

        if modo in [
            "Representación estandarizada",
            "Representación multivariable"
        ]:

            chart_total = chart_total + semicircunferencias(st.session_state.color_circunferencia)

    # ------------------------------------------------------------
    # VECTORES
    # ------------------------------------------------------------

        if modo == "Representación multivariable":

            for vx, vy in vectores:

                chart_total = chart_total + vector_con_flecha(vx, vy)


        chart_total = chart_total.add_selection(
            hover,
            brush
        ).properties(
            width=MAIN_WIDTH,
            height=HEIGHT
        )

# ============================================================
# COORDENADAS PARALELAS
# ============================================================

cols_para = columnas_mv

chart_para_total = empty_chart(PARALLEL_WIDTH, HEIGHT)
if df_render.empty:
    chart_total = empty_chart(MAIN_WIDTH, HEIGHT, "Sin datos seleccionados")
elif len(cols_para) >= 2:

    df_para = df_render.copy()

    
    if modo in [
        "Representación estandarizada",
        "Representación multivariable"
    ]:

        df_para_z = df_z_global.loc[df_para.index, cols_para].copy()

        # mantener __id__ para interacción
        df_para_z["__id__"] = df_para["__id__"]

        df_long = paralelas_rapidas(df_para_z, cols_para)

    else:

        df_long = paralelas_rapidas(df_para, cols_para)

    df_long = df_long.merge(
        df_original,
        on="__id__"
    )
    df_long = df_long.merge(

        df_para[
            ["__id__", "_orden_visual"]
        ],

        on="__id__",
        how="left"
    )
    df_long = df_long.sort_values(
        "_orden_visual"
    )

    df_long["Variable"] = df_long["Variable"].astype(str)
    df_long["Valor"] = pd.to_numeric(df_long["Valor"], errors="coerce")

    # ============================================================
    # EJES
    # ============================================================

    df_axes = pd.DataFrame({
        "Variable": cols_para
    })

    axes = alt.Chart(df_axes).mark_rule(
        color="lightgray",
        strokeWidth=1
    ).encode(
        x=alt.X("Variable:N", sort=cols_para)
    )
    
    # ------------------------------------------------------------
    # SELECTOR POR EJE PARALELAS (invisible)
    # ------------------------------------------------------------

    selectors = alt.Chart(df_long).mark_rect(
        opacity=0
    ).encode(
        x=alt.X("Variable:N", sort=cols_para),
        y="Valor:Q"
    ).add_selection(brush)

    # ============================================================
    # CAPA INTERACTIVA (CLAVE)
    # ============================================================

    puntos_hover = alt.Chart(df_long).mark_circle(
        size=80,
        opacity=0
    ).encode(

        x=alt.X("Variable:N", sort=cols_para),
        y="Valor:Q",

        detail="__id__:N",

        tooltip=["__id__"] + columnas

    ).add_selection(hover)

    # ============================================================
    # LINEAS BASE
    # ============================================================

    base_para = alt.Chart(df_long).mark_line(
        opacity=0.15,
        strokeWidth=stroke_paralelas
    ).encode(
        x=alt.X("Variable:N", sort=cols_para),
        y="Valor:Q",
        detail="__id__:N",
        color=alt.condition(brush, color_encoding, alt.value("lightgray")),
    )

    # ============================================================
    # MULTIPLES LINEAS (brush)
    # ============================================================
    
    selected_para = alt.Chart(df_long).mark_line(
        strokeWidth=stroke_paralelas
    ).encode(
        x=alt.X("Variable:N", sort=cols_para),
        y="Valor:Q",
        detail="__id__:N",
        color=color_encoding
    ).transform_filter(brush)
    
    # ============================================================
    # LINEA RESALTADA (hover)
    # ============================================================

    highlight_para = alt.Chart(df_long).mark_line(
        strokeWidth=stroke_paralelas * 5,
    ).encode(
        x=alt.X("Variable:N", sort=cols_para),
        y="Valor:Q",
        detail="__id__:N",
        tooltip=["__id__"] + columnas,
        color=alt.condition(brush, color_encoding, alt.value("lightgray")),
    ).transform_filter(hover)

    # ============================================================
    # COMBINACION FINAL
    # ============================================================

    chart_para_total = (
        axes +
        base_para +
        selected_para +
        highlight_para +
        puntos_hover
        # + selectors
    ).add_selection(
        brush
    ).properties(
        width=PARALLEL_WIDTH,
        height=HEIGHT
    )
else:
    # chart_para_total = alt.Chart(pd.DataFrame({"x":[], "y":[]})).mark_line()
    chart_para_total = empty_chart(PARALLEL_WIDTH, HEIGHT)

# ============================================================
# PERMUTACION ALEATORIA VISUAL
# ============================================================

df_render = df_render.sample(
    frac=1
).reset_index(drop=True)
# ============================================================
# LAYOUT STREAMLIT (GRAFICOS LADO A LADO)
# ============================================================

chart_combined = alt.hconcat(
    chart_total,
    chart_para_total
).resolve_scale(
)

event = st.vega_lite_chart(
    chart_combined.to_dict(),
    use_container_width=True
)

# ============================================================
# BOTONES INTERCAMBIO COLUMNAS
# ============================================================


if chart_para_total and len(cols_para) >= 2:

    botones = st.columns(len(cols_para)*2-1)

    def swap(i):
        nuevas = list(st.session_state.columnas_mv)
        nuevas[i], nuevas[i+1] = nuevas[i+1], nuevas[i]
        st.session_state.columnas_mv = nuevas

    b = 0
    for i in range(len(cols_para)):

        botones[b].markdown(cols_para[i])
        b += 1

        if i < len(cols_para)-1:
            botones[b].button(
                "<->",
                key=f"swap{i}",
                on_click=swap,
                args=(i,)
            )
            b += 1

# ============================================================
# SCATTER INTERACTIVO PLOTLY
# ============================================================

st.subheader("Interacción Plotly")

PLOTLY_WIDTH = 600
PLOTLY_HEIGHT = 600

# ============================================================
# VALIDACIONES
# ============================================================

if df_render.empty or not st.session_state.get("proyeccion_valida", True):

    st.info("Sin datos para interacción")

elif x_field not in df_render.columns or y_field not in df_render.columns:

    st.error(f"Columnas no encontradas: {x_field}, {y_field}")

else:

    # ========================================================
    # USAR TODOS LOS DATOS
    # ========================================================

    df_plotly = df_original.copy()

    # ========================================================
    # RECONSTRUIR REPRESENTACIÓN
    # ========================================================

    # --------------------------------------------
    # REPRESENTACIÓN ESTANDARIZADA
    # --------------------------------------------

    if modo == "Representación estandarizada":

        df_plotly["_plot_x"] = estandarizar(
            df_original[col_x]
        )

        df_plotly["_plot_y"] = estandarizar(
            df_original[col_y]
        )

        plot_x = "_plot_x"
        plot_y = "_plot_y"

    # --------------------------------------------
    # MULTIVARIABLE
    # --------------------------------------------

    elif modo == "Representación multivariable":
        
        if (
            "X_proj" in df_render_plotly.columns
            and
            "Y_proj" in df_render_plotly.columns
        ):

            df_plotly["X_proj"] = np.nan
            df_plotly["Y_proj"] = np.nan

            df_plotly.loc[
                df_render_plotly.index,
                "X_proj"
            ] = df_render_plotly["X_proj"]

            df_plotly.loc[
                df_render_plotly.index,
                "Y_proj"
            ] = df_render_plotly["Y_proj"]

        plot_x = "X_proj"
        plot_y = "Y_proj"

    # --------------------------------------------
    # REPRESENTACIÓN NORMAL
    # --------------------------------------------

    else:

        plot_x = x_field
        plot_y = y_field

    # ========================================================
    # LIMPIEZA
    # ========================================================

    df_plotly = df_plotly.dropna(
        subset=[plot_x, plot_y]
    ).copy()

    df_plotly = df_plotly.reset_index(drop=True)

    # ========================================================
    # ARRAYS
    # ========================================================

    x_vals = (
        pd.to_numeric(
            df_plotly[plot_x],
            errors="coerce"
        )
        .tolist()
    )

    y_vals = (
        pd.to_numeric(
            df_plotly[plot_y],
            errors="coerce"
        )
        .tolist()
    )

    ids_vals = (
        df_plotly["__id__"]
        .astype(str)
        .tolist()
    )

    # ========================================================
    # FILTRAR NaN
    # ========================================================

    datos_validos = []

    for x, y, i in zip(x_vals, y_vals, ids_vals):

        if pd.notna(x) and pd.notna(y):

            datos_validos.append((x, y, i))

    if not datos_validos:

        st.warning("No hay puntos válidos")

    else:

        x_vals = [d[0] for d in datos_validos]
        y_vals = [d[1] for d in datos_validos]
        ids_vals = [d[2] for d in datos_validos]

        # ====================================================
        # PUNTOS VISIBLES EN ALTAIR
        # ====================================================

        ids_visibles = set(

            st.session_state.df_editado.loc[
                st.session_state.df_editado["Seleccionar"],
                "__id__"
            ].astype(str)

        )

        # ====================================================
        # ESTILO VISUAL
        # ====================================================

        sizes = []
        opacities = []

        for pid in ids_vals:

            if str(pid) in ids_visibles:

                sizes.append(10)
                opacities.append(0.9)

            else:

                sizes.append(4)
                opacities.append(0.5)

        # ====================================================
        # FIGURA INTERACTIVA
        # ====================================================

        fig_interact = go.Figure()

        fig_interact.add_trace(

            go.Scatter(

                x=x_vals,
                y=y_vals,

                mode="markers",

                marker=dict(

                    size=sizes,

                    opacity=opacities
                ),

                text=ids_vals,

                customdata=ids_vals,

                hovertemplate=
                (
                    "<b>ID:</b> %{customdata}<br>"
                    "<b>X:</b> %{x:.4f}<br>"
                    "<b>Y:</b> %{y:.4f}"
                    "<extra></extra>"
                )
            )
        )
        # ====================================================
        # VECTORES INTERACTIVOS
        # ====================================================

        if modo == "Representación multivariable":

            for i, (vx, vy) in enumerate(vectores):

                fig_interact.add_annotation(

                    x=vx,
                    y=vy,

                    ax=0,
                    ay=0,

                    xref="x",
                    yref="y",

                    axref="x",
                    ayref="y",

                    showarrow=True,

                    arrowhead=3,

                    arrowsize=1.5,

                    arrowwidth=2,

                    arrowcolor="black",

                    text=f"V{i+1}"
                )
            # ============================================================
            # CIRCUNFERENCIA UNIDAD
            # ============================================================

            if modo == "Representación multivariable":

                theta = np.linspace(0, 2*np.pi, 400)

                fig_interact.add_trace(

                    go.Scatter(

                        x=np.cos(theta),
                        y=np.sin(theta),

                        mode="lines",

                        line=dict(
                            color="gray",
                            width=2
                        ),

                        hoverinfo="skip",

                        showlegend=False
                    )
                )

            # ============================================================
            # MALLA INVISIBLE CLICKABLE
            # ============================================================

            if modo == "Representación multivariable":

                DOM = 2.5

                grid = np.linspace(-DOM, DOM, 80)

                gx, gy = np.meshgrid(grid, grid)

                grid_x = gx.flatten()
                grid_y = gy.flatten()

                fig_interact.add_trace(

                    go.Scattergl(

                        x=grid_x,
                        y=grid_y,

                        mode="markers",

                        marker=dict(
                            size=10,
                            opacity=0.001
                        ),

                        hoverinfo="skip",

                        hovertemplate="<extra></extra>",

                        showlegend=False
                    )
                )
        # ====================================================
        # LAYOUT INTERACTIVO
        # ====================================================

        fig_interact.update_layout(

            width=PLOTLY_WIDTH,
            height=PLOTLY_HEIGHT,

            dragmode="select",

            clickmode="event+select",

            margin=dict(
                l=40,
                r=40,
                t=40,
                b=80
            ),

            xaxis_title=str(x_field),
            yaxis_title=str(y_field)
        )
        fig_interact.update_xaxes(
            title_text=str(x_field),
            automargin=True
        )

        fig_interact.update_yaxes(
            title_text=str(y_field),
            automargin=True
        )

        # ====================================================
        # EVENTOS PLOTLY
        # ====================================================

        selected = plotly_events(

            fig_interact,

            select_event=True,
            click_event=modo == "Representación multivariable",
            hover_event=False,

            override_height=PLOTLY_HEIGHT,
            override_width=PLOTLY_WIDTH,

            key="plotly_select"
        )

        # ====================================================
        # IDS SELECCIONADOS
        # ====================================================

        ids_seleccionados = []

        if selected:

            # ========================================================
            # CLICK MULTIVARIABLE
            # ========================================================

            if modo == "Representación multivariable" and len(selected) == 1:

                p = selected[0]

                x_real = round(float(p["x"]), 6)
                y_real = round(float(p["y"]), 6)

                click_actual = (x_real, y_real)

                if (
                    st.session_state.get(
                        "ultimo_click_plotly"
                    )
                    !=
                    click_actual
                ):

                    st.session_state[
                        "ultimo_click_plotly"
                    ] = click_actual

                    dim = st.session_state.dimension_activa

                    if sistema_pesos == "Cartesiano":

                        st.session_state[f"w1_store_{dim}"] = float(x_real)
                        st.session_state[f"w2_store_{dim}"] = float(y_real)

                    else:

                        # Conversión cartesiano -> polar

                        radio = np.sqrt(
                            x_real**2 +
                            y_real**2
                        )

                        angulo = np.arctan2(
                            y_real,
                            x_real
                        )

                        # El programa utiliza ángulo en múltiplos de π

                        st.session_state[f"w1_store_{dim}"] = float(radio)
                        st.session_state[f"w2_store_{dim}"] = float(angulo / np.pi)

            # ========================================================
            # SELECCION NORMAL
            # ========================================================

            else:

                ids_seleccionados = []

                for p in selected:

                    if "pointIndex" in p:

                        idx = p["pointIndex"]

                        if 0 <= idx < len(ids_vals):

                            ids_seleccionados.append(
                                str(ids_vals[idx])
                            )

                ids_seleccionados = list(
                    dict.fromkeys(ids_seleccionados)
                )

                st.session_state.plotly_selected_ids = (
                    ids_seleccionados
                )
        else:

            ids_seleccionados = (
                st.session_state.plotly_selected_ids
            )


        # ====================================================
        # BOTONES
        # ====================================================

        c1, c2, c3 = st.columns(3)

        # ----------------------------------------------------
        # OCULTAR
        # ----------------------------------------------------

        if c1.button(
            "Ocultar selección",
            key="btn_ocultar_plotly"
        ):

            if ids_seleccionados:

                mask = (
                    st.session_state.df_editado["__id__"]
                    .astype(str)
                    .isin(ids_seleccionados)
                )

                st.session_state.df_editado.loc[
                    mask,
                    "Seleccionar"
                ] = False

                st.rerun()

        # ----------------------------------------------------
        # MOSTRAR
        # ----------------------------------------------------

        if c2.button(
            "Mostrar selección",
            key="btn_mostrar_plotly"
        ):

            if ids_seleccionados:

                mask = (
                    st.session_state.df_editado["__id__"]
                    .astype(str)
                    .isin(ids_seleccionados)
                )

                st.session_state.df_editado.loc[
                    mask,
                    "Seleccionar"
                ] = True

                st.rerun()

        # ----------------------------------------------------
        # INVERTIR
        # ----------------------------------------------------

        if c3.button(
            "Invertir selección",
            key="btn_invertir_plotly"
        ):

            if ids_seleccionados:

                mask = (
                    st.session_state.df_editado["__id__"]
                    .astype(str)
                    .isin(ids_seleccionados)
                )

                valores_actuales = (
                    st.session_state.df_editado.loc[
                        mask,
                        "Seleccionar"
                    ]
                )

                st.session_state.df_editado.loc[
                    mask,
                    "Seleccionar"
                ] = ~valores_actuales

                st.rerun()


# ============================================================
# ESTADISTICAS
# ============================================================

st.subheader("Estadisticas")

df_stats = estadisticas_originales(
    df_original,
    cols_para
)

if st.session_state.stats_transpose:

    df_stats = df_stats.T

st.dataframe(
    df_stats.style.format("{:.3f}"),
    width=MAIN_WIDTH+PARALLEL_WIDTH
)

fin = time.time()

st.write(
    "Tiempo ejecucion:",
    round(fin-inicio,4),
    "segundos"
)

# ============================================================
# CHECKBOX
# ============================================================

st.subheader("Seleccion de datos")

c1, c2, c3 = st.columns(3)

if c1.button("Marcar todo"):
    st.session_state.df_editado["Seleccionar"] = True

if c2.button("Desmarcar todo"):
    st.session_state.df_editado["Seleccionar"] = False

if c3.button("Invertir todo"):
    st.session_state.df_editado["Seleccionar"] = ~st.session_state.df_editado["Seleccionar"]

df_editado = st.data_editor(
    st.session_state.df_editado,
    use_container_width=True,
    key="editor_final"
)

st.session_state.df_editado = df_editado

if "plotly_selected_ids" not in st.session_state:
    st.session_state.plotly_selected_ids = []


