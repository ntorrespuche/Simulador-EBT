# Simulador de escenarios de EBT

Aplicación desarrollada como componente de productivización del Trabajo Fin de Máster (TFM).

El simulador permite estimar el **EBT (Earnings Before Taxes)** mensual de una compañía de financiamiento del sector automotor bajo diferentes escenarios financieros, macroeconómicos y sectoriales.

La aplicación utiliza el modelo **LightGBM** seleccionado durante el proceso de desarrollo y validación del proyecto.

## Objetivo

Facilitar el uso del modelo predictivo mediante una interfaz sencilla que permita evaluar distintos escenarios sin necesidad de ejecutar código o utilizar el notebook de desarrollo.

La estimación se realiza con un horizonte de **un mes hacia adelante**, utilizando información histórica disponible hasta el último periodo observado y los valores definidos por el usuario para el periodo que desea simular.

## Variables ingresadas por el usuario

El simulador permite modificar las siguientes variables:

- Cartera total
- Patrimonio
- Deuda
- Tasa de Política Monetaria (TPM)
- Interés Bancario Corriente (IBC)
- Vehículos mensuales estimados

A partir de estos valores y de la información histórica disponible, la aplicación construye automáticamente las demás variables requeridas por el modelo:

- `TPM_acum_6m`: variación acumulada de la TPM durante seis meses.
- `IBC_acum_3m`: variación acumulada del IBC durante tres meses.
- `ratio_deuda_patrimonio`: relación entre deuda y patrimonio.
- `EBT_lag_1`: EBT observado en el mes inmediatamente anterior.

Estas variables no se solicitan directamente al usuario porque pueden derivarse automáticamente a partir del escenario ingresado y de la información histórica de referencia.

## Modelo

El modelo utilizado en producción corresponde al **LightGBM final optimizado** seleccionado durante el proceso de comparación y validación temporal realizado en el TFM.

El modelo fue entrenado utilizando información financiera histórica de la compañía, variables macroeconómicas y una variable sectorial relacionada con el mercado automotor.

## Archivos de producción

El repositorio contiene los archivos necesarios para ejecutar la aplicación:

- `app.py`: interfaz y lógica de inferencia del simulador.
- `modelo_ebt.pkl`: modelo LightGBM entrenado y sus metadatos.
- `datos_referencia.csv`: información histórica reciente necesaria para construir las variables temporales utilizadas en la predicción.
- `requirements.txt`: dependencias necesarias para ejecutar la aplicación.

## Funcionamiento

El usuario define un escenario para el siguiente mes mediante las variables disponibles en el formulario.

La aplicación:

1. Valida los valores ingresados.
2. Construye las variables derivadas requeridas por el modelo.
3. Organiza las variables según la estructura utilizada durante el entrenamiento.
4. Ejecuta el modelo LightGBM previamente entrenado.
5. Presenta el EBT estimado para el escenario.
6. Compara la estimación con el último EBT observado.

Los últimos valores disponibles se muestran inicialmente como referencia y pueden ser modificados por el usuario para construir diferentes escenarios.

## Interpretación

Los resultados deben interpretarse como **simulaciones predictivas condicionales**.

La modificación de una variable permite observar cómo cambia la predicción del modelo bajo el nuevo conjunto de condiciones, pero no representa una estimación causal del efecto individual de dicha variable sobre el EBT.

Adicionalmente, al tratarse de un modelo basado en árboles de decisión, cambios pequeños en determinadas variables pueden no modificar inmediatamente la predicción hasta que se alcance alguno de los puntos de división aprendidos por el modelo.

## Nota metodológica

El desarrollo completo del proyecto —incluyendo construcción de la base histórica, análisis exploratorio, ingeniería y selección de variables, validación temporal, comparación de modelos, optimización e interpretabilidad mediante SHAP— se encuentra documentado en el TFM y en su notebook de desarrollo.

Este repositorio corresponde exclusivamente a la **capa de productivización e inferencia del modelo final**.