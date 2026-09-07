# ==========================================================
# SIMULADOR DE ESCENARIOS DE EBT
# APLICACIÓN PRODUCTIVA
# ==========================================================

import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# 1. CONFIGURACIÓN DE LA APLICACIÓN
# ==========================================================

st.set_page_config(
    page_title="Simulador de escenarios de EBT",
    page_icon="📊",
    layout="wide"
)

st.title(
    "Simulador de escenarios de EBT"
)

st.write(
    """
    Herramienta de simulación desarrollada para estimar el EBT
    bajo diferentes escenarios financieros, macroeconómicos
    y sectoriales.
    """
)


# ==========================================================
# 2. CARGA DEL MODELO Y DATOS DE REFERENCIA
# ==========================================================

@st.cache_resource
def cargar_modelo():

    return joblib.load(
        "modelo_ebt.pkl"
    )


@st.cache_data
def cargar_datos():

    datos = pd.read_csv(
        "datos_referencia.csv",
        parse_dates=["Fecha"]
    )

    return (
        datos
        .sort_values("Fecha")
        .reset_index(drop=True)
    )


artefacto = cargar_modelo()

datos_referencia = cargar_datos()

modelo = artefacto["modelo"]
features_modelo = artefacto["features"]


# ==========================================================
# 3. FUNCIONES AUXILIARES
# ==========================================================

def nombre_mes_espanol(fecha):

    meses = {
        1: "enero",
        2: "febrero",
        3: "marzo",
        4: "abril",
        5: "mayo",
        6: "junio",
        7: "julio",
        8: "agosto",
        9: "septiembre",
        10: "octubre",
        11: "noviembre",
        12: "diciembre"
    }

    return (
        f"{meses[fecha.month]} "
        f"de {fecha.year}"
    )


def formato_cop(valor):

    valor_formateado = (
        f"{valor:,.0f}"
        .replace(",", ".")
    )

    return f"${valor_formateado}"


def cop_para_input(valor):

    return (
        f"${valor:,.0f}"
        .replace(",", ".")
    )


def texto_a_cop(texto):

    texto = str(texto)

    texto_limpio = (
        texto
        .replace("$", "")
        .replace("COP", "")
        .replace("cop", "")
        .replace(".", "")
        .replace(",", "")
        .replace(" ", "")
        .strip()
    )

    if texto_limpio == "":

        raise ValueError(
            "Valor vacío"
        )

    return float(
        texto_limpio
    )


def formato_decimal_es(
    valor,
    decimales=2
):

    texto = (
        f"{valor:,.{decimales}f}"
    )

    texto = (
        texto
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    return texto


def formato_entero_es(valor):

    return (
        f"{valor:,.0f}"
        .replace(",", ".")
    )


# ==========================================================
# 4. CONSTRUCCIÓN DEL ESCENARIO
# ==========================================================

def construir_escenario(
    cartera,
    patrimonio,
    deuda,
    tpm,
    ibc,
    vehiculos,
    datos_referencia,
    features_modelo
):

    # ------------------------------------------------------
    # EBT del último mes observado
    # ------------------------------------------------------

    ebt_anterior = (
        datos_referencia
        .iloc[-1]["EBT"]
    )


    # ------------------------------------------------------
    # Ratio deuda / patrimonio
    # ------------------------------------------------------

    ratio_deuda_patrimonio = (
        deuda / patrimonio
    )


    # ------------------------------------------------------
    # TPM acumulada 6 meses
    # ------------------------------------------------------

    serie_tpm = pd.concat(
        [
            datos_referencia[
                "tasa_politica_Monetaria"
            ],
            pd.Series([tpm])
        ],
        ignore_index=True
    )

    dif_tpm = (
        serie_tpm
        .diff()
    )

    tpm_acum_6m = (
        dif_tpm
        .rolling(6)
        .sum()
        .iloc[-1]
    )


    # ------------------------------------------------------
    # IBC acumulado 3 meses
    # ------------------------------------------------------

    serie_ibc = pd.concat(
        [
            datos_referencia[
                "interes_bancario_corriente"
            ],
            pd.Series([ibc])
        ],
        ignore_index=True
    )

    dif_ibc = (
        serie_ibc
        .diff()
    )

    ibc_acum_3m = (
        dif_ibc
        .rolling(3)
        .sum()
        .iloc[-1]
    )


    # ------------------------------------------------------
    # Fila final utilizada por LightGBM
    # ------------------------------------------------------

    escenario = pd.DataFrame(
        [{
            "cartera_total":
                cartera,

            "patrimonio":
                patrimonio,

            "deuda":
                deuda,

            "tasa_politica_Monetaria":
                tpm,

            "Vehiculos_Mensuales_Estimados":
                vehiculos,

            "TPM_acum_6m":
                tpm_acum_6m,

            "IBC_acum_3m":
                ibc_acum_3m,

            "ratio_deuda_patrimonio":
                ratio_deuda_patrimonio,

            "EBT_lag_1":
                ebt_anterior
        }]
    )


    # Mantener exactamente las variables
    # y el orden utilizado durante el entrenamiento

    escenario = escenario[
        features_modelo
    ]


    return escenario


# ==========================================================
# 5. FORMATO VISUAL DE LAS VARIABLES DEL MODELO
# ==========================================================

def preparar_variables_visuales(
    escenario
):

    fila = (
        escenario
        .iloc[0]
    )

    datos = []


    for variable in escenario.columns:

        valor = fila[
            variable
        ]


        # --------------------------------------------------
        # Variables monetarias
        # --------------------------------------------------

        if variable in [
            "cartera_total",
            "patrimonio",
            "deuda",
            "EBT_lag_1"
        ]:

            valor_visual = (
                f"{formato_cop(valor)} COP"
            )


        # --------------------------------------------------
        # Tasa actual
        # --------------------------------------------------

        elif variable == (
            "tasa_politica_Monetaria"
        ):

            valor_visual = (
                f"{formato_decimal_es(valor, 2)} %"
            )


        # --------------------------------------------------
        # Cambios acumulados de tasas
        # --------------------------------------------------

        elif variable in [
            "TPM_acum_6m",
            "IBC_acum_3m"
        ]:

            valor_visual = (
                f"{formato_decimal_es(valor, 2)} pp"
            )


        # --------------------------------------------------
        # Vehículos
        # --------------------------------------------------

        elif variable == (
            "Vehiculos_Mensuales_Estimados"
        ):

            valor_visual = (
                formato_entero_es(
                    valor
                )
            )


        # --------------------------------------------------
        # Ratio deuda / patrimonio
        # --------------------------------------------------

        elif variable == (
            "ratio_deuda_patrimonio"
        ):

            valor_visual = (
                formato_decimal_es(
                    valor,
                    4
                )
            )


        else:

            valor_visual = str(
                valor
            )


        datos.append(
            {
                "Variable":
                    variable,

                "Valor":
                    valor_visual
            }
        )


    return pd.DataFrame(
        datos
    )


# ==========================================================
# 6. INFORMACIÓN DEL ÚLTIMO PERIODO
# ==========================================================

ultimo_mes = (
    datos_referencia
    .iloc[-1]
)

fecha_ultimo_mes = (
    ultimo_mes[
        "Fecha"
    ]
)

fecha_prediccion = (
    fecha_ultimo_mes
    + pd.offsets.MonthEnd(1)
)

ebt_anterior = (
    ultimo_mes[
        "EBT"
    ]
)

mes_anterior_texto = (
    nombre_mes_espanol(
        fecha_ultimo_mes
    )
)

mes_prediccion_texto = (
    nombre_mes_espanol(
        fecha_prediccion
    )
)


# ==========================================================
# 7. INICIALIZACIÓN DE SESSION STATE
# ==========================================================

if "cartera_input" not in st.session_state:

    st.session_state[
        "cartera_input"
    ] = cop_para_input(
        ultimo_mes[
            "cartera_total"
        ]
    )


if "patrimonio_input" not in st.session_state:

    st.session_state[
        "patrimonio_input"
    ] = cop_para_input(
        ultimo_mes[
            "patrimonio"
        ]
    )


if "deuda_input" not in st.session_state:

    st.session_state[
        "deuda_input"
    ] = cop_para_input(
        ultimo_mes[
            "deuda"
        ]
    )


if "resultado_simulacion" not in st.session_state:

    st.session_state[
        "resultado_simulacion"
    ] = None


# ==========================================================
# 8. FUNCIÓN PREVIA AL ENVÍO
# ==========================================================

def preparar_envio():

    # Eliminar cualquier resultado anterior.
    # Así no permanece visible una predicción
    # correspondiente a un escenario anterior
    # cuando el nuevo escenario es inválido.

    st.session_state[
        "resultado_simulacion"
    ] = None


    # ------------------------------------------------------
    # Normalizar visualmente variables financieras
    # ------------------------------------------------------

    campos = [
        "cartera_input",
        "patrimonio_input",
        "deuda_input"
    ]


    for campo in campos:

        try:

            valor = texto_a_cop(
                st.session_state[
                    campo
                ]
            )

            st.session_state[
                campo
            ] = cop_para_input(
                valor
            )


        except ValueError:

            # Se conserva el texto introducido
            # para que posteriormente aparezca
            # el mensaje de validación.

            pass


# ==========================================================
# 9. CONTEXTO DE LA PREDICCIÓN
# ==========================================================

st.subheader(
    f"Escenario de predicción: "
    f"{mes_prediccion_texto.capitalize()}"
)


st.info(
    f"El último periodo observado es "
    f"{mes_anterior_texto}. "
    f"Su EBT conocido es "
    f"{formato_cop(ebt_anterior)} COP "
    f"y se utiliza automáticamente como "
    f"información del mes anterior."
)


st.caption(
    """
    Los campos se inicializan con los últimos valores observados
    únicamente como referencia. Modifíquelos para construir
    el escenario que desea evaluar.
    """
)


# ==========================================================
# 10. FORMULARIO DEL ESCENARIO
# ==========================================================

with st.form(
    "formulario_escenario"
):

    # ------------------------------------------------------
    # Variables financieras
    # ------------------------------------------------------

    st.markdown(
        "### Variables financieras"
    )

    st.caption(
        """
        Ingrese los valores monetarios en pesos colombianos (COP).
        Puede utilizar el signo $ y separadores de miles.
        """
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        cartera_texto = (
            st.text_input(
                "Cartera total (COP)",
                key="cartera_input",
                help=(
                    "Ejemplo: "
                    "$1.982.997.739.851"
                )
            )
        )


    with col2:

        patrimonio_texto = (
            st.text_input(
                "Patrimonio (COP)",
                key="patrimonio_input",
                help=(
                    "Ejemplo: "
                    "$232.977.241.244"
                )
            )
        )


    with col3:

        deuda_texto = (
            st.text_input(
                "Deuda (COP)",
                key="deuda_input",
                help=(
                    "Ejemplo: "
                    "$1.974.053.386.419"
                )
            )
        )


    # ------------------------------------------------------
    # Variables macroeconómicas y sectoriales
    # ------------------------------------------------------

    st.markdown(
        "### Variables macroeconómicas y sectoriales"
    )


    col4, col5, col6 = (
        st.columns(3)
    )


    with col4:

        tpm = (
            st.number_input(
                "Tasa de Política Monetaria (%)",
                value=float(
                    ultimo_mes[
                        "tasa_politica_Monetaria"
                    ]
                ),
                step=0.25,
                format="%.2f"
            )
        )


    with col5:

        ibc = (
            st.number_input(
                "Interés Bancario Corriente (%)",
                value=float(
                    ultimo_mes[
                        "interes_bancario_corriente"
                    ]
                ),
                step=0.10,
                format="%.2f"
            )
        )


    with col6:

        vehiculos = (
            st.number_input(
                "Vehículos mensuales estimados",
                value=float(
                    ultimo_mes[
                        "Vehiculos_Mensuales_Estimados"
                    ]
                ),
                step=100.0,
                format="%.0f"
            )
        )


    simular = (
        st.form_submit_button(
            "Simular EBT",
            type="primary",
            use_container_width=True,
            on_click=preparar_envio
        )
    )


# ==========================================================
# 11. EJECUCIÓN DE LA SIMULACIÓN
# ==========================================================

if simular:


    # ------------------------------------------------------
    # Conversión de variables financieras
    # ------------------------------------------------------

    try:

        cartera = texto_a_cop(
            st.session_state[
                "cartera_input"
            ]
        )

        patrimonio = texto_a_cop(
            st.session_state[
                "patrimonio_input"
            ]
        )

        deuda = texto_a_cop(
            st.session_state[
                "deuda_input"
            ]
        )


    except ValueError:

        st.error(
            """
            Revise los valores monetarios ingresados.
            Utilice únicamente números, el signo $
            y separadores de miles.
            """
        )

        st.stop()


    # ------------------------------------------------------
    # Validación de variables financieras
    # ------------------------------------------------------

    if cartera < 0:

        st.error(
            "La cartera total no puede ser negativa."
        )

        st.stop()


    if patrimonio <= 0:

        st.error(
            "El patrimonio debe ser mayor que cero."
        )

        st.stop()


    if deuda < 0:

        st.error(
            "La deuda no puede ser negativa."
        )

        st.stop()


    # ------------------------------------------------------
    # Validación de variables macroeconómicas y sectoriales
    # ------------------------------------------------------

    if tpm < 0:

        st.error(
            """
            La Tasa de Política Monetaria
            no puede ser negativa.
            """
        )

        st.stop()


    if ibc < 0:

        st.error(
            """
            El Interés Bancario Corriente
            no puede ser negativo.
            """
        )

        st.stop()


    if vehiculos < 0:

        st.error(
            """
            El número de vehículos mensuales
            estimados no puede ser negativo.
            """
        )

        st.stop()


    # ------------------------------------------------------
    # Construcción del escenario
    # ------------------------------------------------------

    escenario = (
        construir_escenario(
            cartera=cartera,
            patrimonio=patrimonio,
            deuda=deuda,
            tpm=tpm,
            ibc=ibc,
            vehiculos=vehiculos,
            datos_referencia=datos_referencia,
            features_modelo=features_modelo
        )
    )


    # ------------------------------------------------------
    # Validación de valores faltantes
    # ------------------------------------------------------

    if (
        escenario
        .isna()
        .any()
        .any()
    ):

        st.error(
            """
            No fue posible construir todas las variables
            necesarias para la predicción.
            """
        )

        st.stop()


    # ------------------------------------------------------
    # Predicción
    # ------------------------------------------------------

    ebt_estimado = (
        modelo
        .predict(
            escenario
        )[0]
    )


    # ------------------------------------------------------
    # Guardar resultado
    # ------------------------------------------------------

    st.session_state[
        "resultado_simulacion"
    ] = {
        "ebt_estimado":
            ebt_estimado,

        "escenario":
            escenario.copy()
    }


# ==========================================================
# 12. RESULTADO DE LA SIMULACIÓN
# ==========================================================

resultado = (
    st.session_state[
        "resultado_simulacion"
    ]
)


if resultado is not None:

    ebt_estimado = (
        resultado[
            "ebt_estimado"
        ]
    )

    escenario = (
        resultado[
            "escenario"
        ]
    )


    # ------------------------------------------------------
    # Comparación con último EBT observado
    # ------------------------------------------------------

    variacion_absoluta = (
        ebt_estimado
        - ebt_anterior
    )


    if ebt_anterior != 0:

        variacion_porcentual = (
            variacion_absoluta
            / abs(
                ebt_anterior
            )
        ) * 100

    else:

        variacion_porcentual = 0


    # ------------------------------------------------------
    # Presentación del resultado
    # ------------------------------------------------------

    st.divider()

    st.markdown(
        "## Resultado de la simulación"
    )


    col_resultado_1, col_resultado_2 = (
        st.columns(2)
    )


    with col_resultado_1:

        st.metric(
            label=(
                "EBT estimado para "
                f"{mes_prediccion_texto}"
            ),
            value=(
                f"{formato_cop(ebt_estimado)} COP"
            )
        )


    with col_resultado_2:

        st.metric(
            label=(
                "Variación frente a "
                f"{mes_anterior_texto}"
            ),
            value=(
                f"{formato_cop(variacion_absoluta)} COP"
            ),
            delta=(
                f"{variacion_porcentual:.1f}%"
            )
        )


    st.caption(
        f"EBT observado en "
        f"{mes_anterior_texto}: "
        f"{formato_cop(ebt_anterior)} COP"
    )


    # ======================================================
    # 13. VARIABLES UTILIZADAS POR EL MODELO
    # ======================================================

    with st.expander(
        "Ver variables utilizadas por el modelo"
    ):

        escenario_visual = (
            preparar_variables_visuales(
                escenario
            )
        )


        st.dataframe(
            escenario_visual,
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------
        # Explicación de variables calculadas automáticamente
        # --------------------------------------------------

        st.markdown(
            f"""
            #### Variables calculadas automáticamente

            El modelo utiliza **nueve variables** para generar cada
            predicción. Algunas se solicitan directamente en el formulario,
            mientras que otras se calculan automáticamente utilizando los
            valores ingresados y la información histórica disponible.

            Esto evita solicitar datos redundantes al usuario y garantiza
            que las variables se construyan de la misma forma utilizada
            durante el entrenamiento del modelo.

            **TPM_acum_6m**

            Representa el cambio acumulado de la **Tasa de Política
            Monetaria durante los últimos seis meses**. Se construye a
            partir de las variaciones mensuales de la TPM histórica e
            incorpora la tasa ingresada para el escenario actual.

            Por esta razón, el usuario únicamente necesita ingresar la
            TPM del escenario que desea evaluar.

            **IBC_acum_3m**

            Representa el cambio acumulado del **Interés Bancario Corriente
            durante los últimos tres meses**. Se calcula utilizando las
            variaciones mensuales recientes del IBC e incorpora el valor
            ingresado para el escenario actual.

            Esto permite al modelo considerar no solo el nivel actual de
            las tasas, sino también su evolución reciente.

            **ratio_deuda_patrimonio**

            Representa la relación entre la deuda y el patrimonio del
            escenario:

            **Deuda / Patrimonio**

            Como ambas variables ya son ingresadas en el formulario, la
            aplicación calcula automáticamente este indicador y evita
            solicitarlo nuevamente al usuario.

            **EBT_lag_1**

            Corresponde al **EBT observado en el último mes disponible**.

            Para esta simulación se utiliza automáticamente el EBT de
            **{mes_anterior_texto}**, equivalente a
            **{formato_cop(ebt_anterior)} COP**.

            Esta variable permite que el modelo incorpore información sobre
            el comportamiento reciente del EBT. No se solicita al usuario
            porque corresponde a información histórica ya conocida por la
            aplicación.

            En conjunto, estas variables derivadas permiten mantener la
            construcción del escenario consistente con el proceso utilizado
            durante el desarrollo y entrenamiento del modelo.
            """
        )


# ==========================================================
# 14. NOTA METODOLÓGICA
# ==========================================================

st.divider()

st.markdown(
    "### Nota metodológica"
)

st.caption(
    """
    La estimación corresponde a una predicción mensual condicional
    generada mediante un modelo LightGBM.

    El modelo utiliza información financiera, macroeconómica y
    sectorial del escenario ingresado, junto con información
    histórica disponible hasta el mes anterior.

    Los resultados deben interpretarse como simulaciones predictivas
    y no como estimaciones de efectos causales sobre el EBT.
    """
)