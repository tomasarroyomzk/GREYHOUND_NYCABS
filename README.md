# Proyecto de Análisis Ambiental para Greyhound

## Propuesta de Negocio

La empresa de servicios de transporte de pasajeros “Greyhound” ha contratado nuestro equipo de Data Science con el objetivo de analizar su inversión en transportes particulares en Nueva York y revisar su impacto en el medio ambiente.

Para este análisis, utilizaremos como referencia los datasets de los taxis de la ciudad de Nueva York, junto con los datos de calidad del aire y contaminación acústica de la misma ciudad.

## Objetivos

Nuestro objetivo principal es brindar a Greyhound información valiosa para la toma de decisiones, enfocándonos en dos aspectos clave:

1. **Asesoramiento Ambiental:** Proporcionaremos a Greyhound perspectivas ambientales relevantes, permitiéndoles entender el impacto ambiental de su inversión en transporte.

2. **Rentabilidad del Negocio:** Analizaremos la rentabilidad del negocio y entregaremos conclusiones que faciliten la toma de decisiones estratégicas.

## Acciones Planeadas

1. **Análisis Ambiental:**
   - Utilizaremos datos de calidad del aire para evaluar el impacto ambiental de la flota de transporte de Greyhound.
   - Analizaremos la contaminación acústica asociada a la operación de los vehículos.

2. **Rentabilidad del Negocio:**
   - Utilizaremos datasets de taxis para evaluar la eficiencia y rentabilidad de la flota de transporte.
   - Desarrollaremos un modelo de Machine Learning para prever tendencias futuras y proporcionar información predictiva.

3. **Productos Entregables:**
   - **Dashboard Ambiental:** Crearemos un dashboard interactivo que visualice el impacto ambiental y la rentabilidad del negocio.
   - **Modelo de Machine Learning:** Desarrollaremos un modelo predictivo que permita a Greyhound anticipar escenarios futuros.

## JIRA
Gracias al software Jira pudimos hacer un cronograma con las tareas que definimos llevar a cabo para ejecutar la propuesta y les asignamos roles de todos los colaboradores del equipo.
IMAGEN DE JIRAAA

# Desarrollo del Proyecto

## Marco de Referencia del Proyecto
En esta fase, se ha trabajado en la estructuración y planificación inicial del proyecto, definiendo objetivos y alcance.

## Objetivos Específicos del Proyecto
En este paso, hemos definido los objetivos específicos que guiarán nuestro trabajo.

## ETL (Extract, Transform, Load)
Realizamos la transformación y limpieza de datos de diversos archivos para garantizar calidad y representatividad. Se emplearon roles de Data Engineering para asegurar la correcta ejecución. También utilizamos diccionarios para la visualización de datos y compartimos análisis preliminares en el repositorio.

## Datasets
Definimos los conjuntos de datos a utilizar, junto con diccionarios, descripciones y documentación de las fuentes.
1) Dataset “Taxis de la ciudad de Nueva york”.
2) Dataset “Calidad del aire en la ciudad de Nueva York”.
3) Dataset “Contaminación acústica en la ciudad de Nueva york”.

## EDA (Exploratory Data Analysis)
Realizamos un análisis exploratorio de datos en los tres datasets, proporcionando una visión integral del impacto de la inversión en transporte de pasajeros en Nueva York en la calidad del aire y la contaminación sonora. Se proponen KPIs para evaluar el alcance y los efectos de la inversión.

## Data Engineering
Generamos los pasos finales del ETL, documentos, diccionarios, Pipeline, diseño de modelo Entidad Relacional, Pipelines para alimentar el Data Warehouse y validación de datos. También se realizó un análisis de datos de muestra y se presentó el MVP.

## Dashboard y Data Analytics
Desarrollaremos un dashboard interactivo con los análisis propuestos y los KPIs definidos una vez que se haya completado el ETL y los datasets estén disponibles. También devolvimos conclusiones del análisis.

## •	KPI´s:  
Buscamos identificar patrones de viajes, demandas, zonas y mediciones representativas en las métricas establecidas para verificar el cumplimiento de los objetivos definidos y su correspondiente fundamento.

1. Promedio de Contaminación de Aire y Acústica por viaje, verificar si disminuye un 30% los niveles del 2022 respecto al 2021.

2. Ratio de Viajes que Toman Taxi entre el año 2021 y el 2022, y revisar si hubo un aumento del 15% en Queens en el último año.

3. Precio Promedio de Viajes en el último año 2022, ver si aumentó un 10% respecto del 2021.

4. Cantidad de Viajes en Horas Pico del último año 2022, verificar si aumentó en un 15% respecto al año 2021.

## Modelo de Machine Learning
Se implementó un modelo de Machine Learning de Regresión Lineal, en el cuál se vincularon las variables dependientes Monóxido de Carbono, Dióxido de Azufre y Dióxido de Nitrógeno (contaminantes del aire) con la variable independiente Costo del servicio de Taxi (en USD).
El modelo ejemplifica el comportamiento de los contaminantes del aire para cada valor en USD del servicio de táxi en Nueva York, este modelo se puede consumir de manera sencilla en esta [calculadora](https://proyectogrupalhenry-mmljsfcrkxpkshzh7rwvsx.streamlit.app/)

## Stack Tecnológico
Optamos por un stack tecnológico mayormente gratuito y de fácil acceso, respaldado por nuestra amplia experiencia en la consultora.


