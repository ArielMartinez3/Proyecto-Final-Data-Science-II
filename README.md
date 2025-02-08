# README: Insight de Ingresos: Predicción con Datos Censales con ML

## Introducción
Este proyecto tiene como objetivo analizar la relación entre el nivel educativo, la ocupación, las horas trabajadas y los ingresos. Se buscará responder a preguntas clave sobre la distribución de ingresos y cómo se ven influenciados por diferentes factores socioeconómicos.

## Abstract
El “Adult” dataset, también conocido como “Census Income” dataset, es una colección de datos del censo que se utiliza para predecir si el ingreso de una persona supera los $50,000 dólares americanos al año. Este dataset contiene 48,842 instancias y 14 características, incluyendo edad, educación, ocupación, y horas trabajadas por semana. El objetivo de este proyecto es explorar y visualizar los datos para identificar patrones y relaciones que puedan ayudar a responder preguntas específicas sobre los factores que influyen en los ingresos de las personas. A través de visualizaciones univariadas, bivariadas y multivariadas, junto con resúmenes numéricos, se buscará proporcionar una comprensión profunda de las variables más influyentes en la determinación de los ingresos.

## Dataset
* **Nombre:** Adult
* **Fuente:** UCI Machine Learning Repository
* **Enlace:** Adult Dataset
* **Descripción:** Conjunto de datos del censo utilizado para predecir el ingreso de una persona.
* **Variables:**
  * **age**: Edad del individuo en años.
  * **workclass**: Tipo de empleo o clase laboral (ej. privado, gubernamental, autónomo).
  * **fnlwgt**: Peso final, que representa el número de personas que se estiman en la población a partir de la muestra. Es un ajuste de la muestra para que sea representativa.
  * **education**: Nivel educativo alcanzado (ej. secundaria, licenciatura, maestría).
  * **education_num**: Número asociado al nivel educativo (0-16), donde 1 es la educación menos avanzada y 16 representa el doctorado.
  * **marital_status**: Estado civil del individuo (ej. soltero, casado, divorciado).
  * **occupation**: Ocupación del individuo (ej. administrativo, técnico, trabajador manual).
  * **relationship**: Relación con el cabeza de familia (ej. cabeza de familia, cónyuge, hijo).
  * **race**: Raza del individuo (ej. blanco, negro, asiático, indígena).
  * **sex**: Género del individuo (masculino o femenino).
  * **capital_gain**: Ganancia de capital del individuo en el último año.
  * **capital_loss**: Pérdida de capital del individuo en el último año.
  * **hours_per_week**: Número de horas trabajadas por semana.
  * **native_country**: País de origen o nacionalidad del individuo.
  * **income**: Ingreso del individuo, clasificado como superior o inferior a $50,000 anuales.

## Objetivos

* El objetivo de este proyecto es analizar los factores sociodemográficos que influyen en los ingresos, específicamente, si una persona gana más de $50,000 al año. Además, se busca identificar los factores más relevantes que influyen en el ingreso, para luego construir modelos de clasificación para predecir si una persona tiene un ingreso superior a $50,000.

* Contexto Comercial: Las empresas y gobiernos buscan identificar los factores que influyen en los ingresos para diseñar políticas efectivas de empleo, educación y bienestar económico. Este análisis ayudará a comprender qué variables impactan más en el nivel de ingresos, permitiendo tomar decisiones informadas.

* Problema Comercial: Existe una disparidad en los ingresos de la población según varios factores como la educación, ocupación y género. El problema radica en cómo identificar y aprovechar estos factores para mejorar las oportunidades laborales y reducir la desigualdad económica.

* Contexto Analítico: Utilizando técnicas de Exploratory Data Analysis (EDA) y modelado predictivo, se analizará las relaciones entre las diferentes variables del dataset y el ingreso de las personas. Se explorarán visualizaciones clave y métricas de desempeño para evaluar la influencia de cada variable.

## Preguntas de Investigación
* **Pregunta 1:** ¿Cuál es la relación entre el nivel educativo y el ingreso?
* **Pregunta 2:** ¿Cómo influye la ocupación en el ingreso?
* **Pregunta 3:** ¿Existe una diferencia significativa en los ingresos entre hombres y mujeres?

## Metodología
1. **Carga de datos:** Cargar el dataset utilizando pandas.
2. **Exploración de datos:**
   * **Análisis univariado:** Histograma de edad, recuento de valores únicos para variables categóricas.
   * **Análisis bivariado:** Gráficos de dispersión para horas trabajadas vs. ingreso, boxplots para comparar ingresos por nivel educativo.
   * **Análisis multivariado:** Pairplots para visualizar relaciones entre múltiples variables.
3. **Limpieza de datos:**
   * Manejar valores faltantes (imputación o eliminación).
   * Codificar variables categóricas.
4. **Modelado:**
   * Selección de características relevantes.
   * Construcción de modelos de clasificación (regresión logística, árboles de decisión, etc.).
   * Evaluación del rendimiento de los modelos.
5. **Visualización de datos:**
   * Realizar 3 gráficos diferentes con Matplotlib.
   * Realizar 3 gráficos diferentes con Seaborn.
   * Usar al menos un parámetro adicional (grid, hue, etc.) para enriquecer la legibilidad de los gráficos.
   * Interpretar cada gráfico para obtener insights relevantes que permitan dar respuesta a las preguntas de investigación.

## Herramientas
* **Lenguaje de programación:** Python
* **Librerías:** pandas, NumPy, matplotlib, seaborn, scikit-learn, SMOTE, lightgbm, xgboost.

## Resultados Esperados
* Identificar los factores más importantes que influyen en el ingreso.
* Construir un modelo de clasificación preciso para predecir el ingreso.
* Visualizaciones claras y concisas que respalden los hallazgos.

## Conclusiones Gráficos

### Histograma del Nivel Educativo
- La mayoría de las personas tienen entre 9 y 12 años de educación, con el punto más alto en 10 años, que corresponde a la finalización de la secundaria (high school).
- En promedio, las personas en el dataset tienen poco más de 10 años de educación.
- La visualización muestra que la mayoría de los individuos han completado la secundaria, con una mayor representación de niveles educativos superiores que los de menor educación.

### Gráfico de Barras de Ingresos por Ocupación
- Las ocupaciones como "Ejecutivo" y "Profesional" tienen más personas con ingresos altos.
- Ocupaciones como "Limpiador" y "Servicios domésticos" tienen menos personas con ingresos altos.
- Algunas ocupaciones tienen una mezcla de ingresos altos y bajos, como "Ventas" y "Reparación".
- El gráfico muestra que las ocupaciones pueden influir en el nivel de ingresos, con algunos trabajos claramente asociados con mejores salarios que otros.

### Scatterplot: Años de Educación vs Capital Ganado
- No hay una tendencia clara en cómo el capital ganado cambia con más años de educación. La relación parece débil, como lo indica la baja correlación de 0.12. La mayoría de los puntos están cerca del eje horizontal, indicando que el capital ganado es bajo para muchos con distintos niveles educativos.
- Aunque el capital ganado a partir de los 9 años de educación es levemente superior al de los grupos de menor educación, lo que puede sugerir cierta influencia, la distribución de ingresos sigue siendo muy variada.
- No parece haber una relación fuerte entre los años de educación y el capital ganado en el dataset.

### Boxplot: Horas Trabajadas por Semana según el Sexo
- Los hombres tienden a trabajar más horas a la semana en comparación con las mujeres, con una media de 42.4 horas frente a 36.4 horas.
- El rango de horas trabajadas es más amplio para los hombres, con una mayor variabilidad.
- Las medianas (líneas dentro de las cajas) son iguales para ambos sexos (40 horas), pero las mujeres tienen una distribución más concentrada en torno a esta mediana.

### Pairplot de Variables Seleccionadas
Este gráfico muestra cómo se relacionan entre sí cuatro variables importantes: edad, años de educación, horas trabajadas por semana e ingresos.
- **Observaciones:**
  - **Edad y Educación:** No parece haber una relación clara entre la edad y los años de educación.
  - **Edad y Horas Trabajadas:** Los ingresos más altos tienden a estar asociados con un mayor número de horas trabajadas.
  - **Educación y Horas Trabajadas:** Las personas con más educación tienden a trabajar más horas.
  - **Ingreso:** Las personas con ingresos más altos tienden a tener más educación y trabajar más horas.

### Gráfico de Dispersión Multivariado
- Las personas con más edad tienden a trabajar menos horas por semana. Esto puede ser porque los trabajos suelen ser menos demandantes para los trabajadores mayores.
- Los puntos más grandes, que indican un mayor nivel educativo, están más relacionados con el aumento en las horas trabajadas y los ingresos más altos.
- Los puntos en colores más cálidos (indicando ingresos más altos) tienden a tener más educación y a trabajar más horas por semana.
- El gráfico muestra que los trabajadores más educados y con mayores ingresos tienden a trabajar más horas y tienen un rango de edad más variado. Además, los trabajadores mayores suelen trabajar menos horas.

### Distribución de Ocupaciones
- Las ocupaciones más frecuentes son "Prof-specialty" (profesionales especializados), "Craft-repair" (trabajos de reparación y mantenimiento), y "Exec-managerial" (gestión ejecutiva).
- Las ocupaciones menos comunes incluyen "Armed-Forces" (fuerzas armadas) y "Priv-house-serv" (servicios domésticos privados).

### Distribución de Ingresos por Género
- Hay una notable mayor proporción de hombres con ingresos superiores a 50K en comparación con las mujeres.
- La proporción de ingresos inferiores a 50K se distribuye de una manera mucho más similar en ambos géneros.

## Distribución de Ocupaciones por Género

1. **Ocupaciones con Alta Proporción de Hombres:**
   - Ocupaciones como "Exec-managerial", "Craft-repair", y "Transport-moving" tienen una representación significativamente mayor de hombres.
   - Estas ocupaciones suelen estar asociadas con mayores ingresos. Por ejemplo, "Exec-managerial" tiene una alta probabilidad de ingresos superiores a 50K, lo que coincide con la alta representación masculina.

2. **Ocupaciones con Alta Proporción de Mujeres:**
   - Ocupaciones como "Other-service" y "Priv-house-serv" tienen una mayor cantidad de mujeres.
   - Estas ocupaciones tienden a tener una menor proporción de ingresos superiores a 50K. La ocupación "Other-service", en particular, tiene una alta representación femenina y una baja proporción de ingresos altos.

3. **Ocupaciones con Baja Representación Femenina:**
   - "Craft-repair" y "Protective-serv" tienen muy pocas mujeres en comparación con los hombres.
   - Estas ocupaciones también están asociadas con una variedad de niveles de ingresos, pero "Craft-repair" en particular tiende a tener una mayor proporción de ingresos altos.

## Distribución de Ingresos según Años de Educación

- A más años de educación, mayor porcentaje de personas con ingresos >50K. Por ejemplo, el 74% de quienes tienen 16 años de educación ganan más de 50K, frente al 0% con solo 1 año.
- Los porcentajes de ingresos altos aumentan notablemente a partir de 9 años de educación, alcanzando niveles críticos en educación universitaria.
- Los niveles educativos avanzados (13 años o más) muestran los mayores porcentajes de ingresos >50K, destacando la educación universitaria como clave para mayores ingresos.
- Para niveles educativos bajos (hasta 7 años), la mayoría tiene ingresos menores a 50K, indicando que una educación mínima limita las oportunidades de altos ingresos.

## Resultados obtenidos 

### Diferencias en Ingresos entre Géneros
- Los hombres tienen una mayor probabilidad de tener ingresos superiores a 50K en comparación con las mujeres.
- Los hombres trabajan en promedio 42.43 horas por semana, mientras que las mujeres trabajan 36.41 horas por semana.
- El análisis muestra que hay una correlación entre la representación de género en diferentes ocupaciones y la diferencia salarial. Las ocupaciones dominadas por hombres tienden a ofrecer mayores ingresos, mientras que las ocupaciones dominadas por mujeres suelen ofrecer salarios más bajos. 
- Esta diferencia en la distribución de género en ocupaciones específicas contribuye a la brecha salarial observada entre hombres y mujeres en el dataset.

**Conclusión:** La diferencia en las horas trabajadas puede contribuir a las diferencias de ingresos observadas entre géneros, y también hay una correlación con la representación de ambos géneros en ciertas ocupaciones.

### Relación entre ocupación e ingreso
- Algunas ocupaciones, como "Exec-managerial" y "Prof-specialty", pueden mostrar una alta proporción de personas con ingresos mayores a 50K. Estas ocupaciones suelen estar asociadas con altos niveles de responsabilidad y especialización, lo que puede justificar los altos ingresos.
- Ocupaciones como "Adm-clerical" y "Other-service" pueden tener una baja proporción de personas con ingresos superiores a 50K. Estas ocupaciones a menudo están asociadas con roles de apoyo o servicios, que típicamente tienen menores salarios comparados con roles de gestión o profesionales especializados.
- La distribución de ingresos según ocupación puede resaltar disparidades salariales entre diferentes tipos de trabajos. Por los resultados obtenidos anteriormente, es probable que las ocupaciones que requieren más educación y experiencia estén mejor remuneradas en comparación con trabajos que no requieren tales calificaciones.
- El análisis y visualización ayudan a identificar patrones en cómo diferentes ocupaciones están relacionadas con el nivel de ingresos, proporcionando una visión clara sobre la disparidad salarial en el dataset.

### Relación entre Nivel Educativo e Ingresos
- Hay una estrecha relación entre el nivel educativo y la probabilidad de obtener ingresos superiores a 50K. A medida que aumentan los años de educación, también aumenta la probabilidad de tener ingresos más altos.
- Los niveles educativos avanzados (13 años o más) muestran los mayores porcentajes de ingresos >50K, destacando la educación universitaria como clave para mayores ingresos.
- Porcentaje de personas con ingresos mayores a 50K y más de 12 años de educación: 52.58%
- Porcentaje de personas con ingresos mayores a 50K que tienen más de 12 años de educación: 76.15%
- Porcentaje con Ingresos mayores a 50K: 24% de la población.

### Análisis Exploratorio
En este análisis exploratorio, hemos podido visualizar la distribución de las variables clave, identificar posibles valores atípicos y entender las relaciones entre las características más importantes del dataset. Los gráficos nos permitieron observar la influencia de factores como la educación, horas trabajadas y ocupación sobre el nivel de ingresos. Este proceso ha sido fundamental para tomar decisiones informadas sobre la preparación de los datos y la selección de características. Ahora, con una comprensión clara del comportamiento de los datos, estamos listos para avanzar a la siguiente etapa: la implementación de algoritmos de Machine Learning.

# Modelización
En la próxima fase, aplicaremos distintos modelos de clasificación, como la Regresión Logística, Random Forest y XGBoost, para predecir si una persona tiene un ingreso superior o inferior a $50,000 al año. Estos modelos se entrenarán con los datos procesados y serán evaluados mediante métricas como precisión, recall y F1-score, con el objetivo de seleccionar el modelo más adecuado para este problema de clasificación binaria.

Este proyecto se basa en aprendizaje supervisado, ya que el objetivo principal es predecir si una persona tiene un ingreso superior o inferior a $50,000 al año, utilizando una variable de ingreso binaria ('<=50K' o '>50K'). En este tipo de aprendizaje, el modelo tiene acceso a una variable objetivo o etiqueta, en este caso el nivel de ingresos, que será predicho en función de otras características o atributos del dataset (como la edad, la ocupación, las horas trabajadas por semana, el nivel educativo, etc.).

Los tres factores más determinantes a la hora de decidir el tipo de aprendizaje fueron:
1. **Disponibilidad de etiquetas:** En el dataset proporcionado, el ingreso ya está etiquetado como una variable categórica, lo que facilita la clasificación en grupos predeterminados.
2. **Clasificación:** El problema principal del proyecto es clasificar a las personas en una de dos categorías de ingreso ('<=50K' o '>50K'), lo cual es un problema de clasificación, uno de los subtipos más comunes del aprendizaje supervisado.
3. **Datos históricos:** Ya se dispone de datos previos con los que se puede entrenar al modelo para que aprenda a reconocer patrones entre las características de entrada (educación, horas trabajadas, ocupación, etc.) y la salida (ingresos).

### Posibles Modelos de Resolución
Al tratarse de un problema de clasificación supervisada, algunos modelos adecuados podrían ser: Regresión Logística, Árboles de Decisión, Random Forest, Support Vector Machines (SVM), Gradient Boosting Machines (GBM) (como XGBoost o LightGBM) y K-Nearest Neighbors (K-NN). Todos estos modelos pueden evaluarse utilizando métricas como la precisión, recall, F1-score y AUC-ROC, lo que permitirá seleccionar el modelo que mejor se ajuste al problema de clasificación binaria del ingreso.

### Distinción entre Regresión y Clasificación
La distinción entre regresión y clasificación en aprendizaje supervisado se basa en el tipo de variable objetivo o etiqueta que se intenta predecir:
- **Regresión:** Se utiliza cuando la variable objetivo es continua. En este caso, se predicen valores numéricos en un rango, como ingresos, temperaturas, o precios. Por ejemplo, predecir el salario exacto de una persona sería un problema de regresión.
- **Clasificación:** Se utiliza cuando la variable objetivo es categórica. Aquí se predicen clases o categorías, como "sí" o "no", "alto" o "bajo", o en este caso, "ingreso mayor a 50K" o "ingreso menor o igual a 50K".

En este proyecto, la variable objetivo es "income", que tiene dos posibles valores: <=50K (ingreso menor o igual a 50,000 dólares al año) y >50K (ingreso mayor a 50,000 dólares al año). Estos valores son categóricos y representan clases que dividen a las personas en dos grupos en función de su nivel de ingresos. Por lo tanto, estamos trabajando en un problema de clasificación (la variable objetivo no es continua, no se predice el ingreso exacto).

## Detección de Outliers
Utilizamos boxplots para identificar outliers en variables clave, como edad, horas trabajadas por semana y ganancias de capital. Esto es importante porque los outliers pueden influir negativamente en los modelos de Machine Learning, sesgando los resultados. Detectamos outliers para evitar que el modelo se ajuste demasiado a valores extremos, lo que podría provocar sobreajuste.

Los boxplots mostraron la presencia de valores extremadamente altos en variables como 'capital_gain' y 'hours_per_week'. A partir de esta observación, decidimos explorar métodos de preprocesamiento que nos permitan reducir el impacto de estos valores atípicos.

## Limpieza de Datos
Procedimos a limpiar los datos, eliminando o imputando valores faltantes en columnas relevantes. Este proceso es crucial para garantizar que el conjunto de datos esté en condiciones óptimas para el análisis y el entrenamiento de los modelos. Consideramos el desbalance en la variable objetivo, para evitar sesgos y asegurar que ambos grupos (ingresos >50K o <=50K) estén representados de manera adecuada con técnicas como SMOTE.

## Reducción de Dimensionalidad - Selección de Características
Implementamos un proceso de selección de características (Feature Selection) utilizando dos métodos: Recursive Feature Elimination (RFE) y SelectKBest con la prueba chi-cuadrado. Estos métodos nos permiten reducir la dimensionalidad del conjunto de datos, enfocándonos solo en las características más relevantes, como educación, ocupación y horas trabajadas, lo que mejora la eficiencia del modelo. Usamos validación cruzada para verificar que la selección de características no favorezca únicamente el conjunto de entrenamiento.

RFE y SelectKBest identificaron variables como 'education', 'occupation', y 'age' como las más importantes para predecir ingresos. Este proceso redujo la dimensionalidad del dataset sin perder información crítica, mejorando la velocidad y precisión de los algoritmos.

### Resultados Obtenidos

#### Matriz de Correlación
La mayoría de las variables presentan una correlación débil con la variable objetivo (income), con valores que oscilan entre -0.5 y 0.5. Las variables con mayor correlación son fnlwgt (0.0526), education_num (0.1421) y hours_per_week (0.1110). En contraste, las variables con menor correlación incluyen race (0.0863), sex (0.0789) y native_country (0.0769). Esto sugiere que, aunque algunas variables muestran cierta relación, ninguna presenta una correlación extremadamente fuerte. Procederemos a realizar análisis adicionales, como regresiones o modelos más complejos, para profundizar en estas relaciones.

#### Chi-cuadrado
El test de Chi-cuadrado se utiliza para evaluar la independencia entre las variables categóricas. Todos los p-values obtenidos son extremadamente bajos, lo que indica que todas las variables tienen una relación significativa con la predicción de los ingresos. Esto refuerza la idea de que cada variable aporta información valiosa para explicar las diferencias en los niveles de ingresos, aunque con diferentes grados de influencia. A pesar de la presencia de valores outliers, se decidió mantenerlos en el análisis, dado su potencial impacto para el análisis del modelo de ML.

#### RFE (Recursive Feature Elimination)
A través de la selección de variables con RFE, se obtuvo una precisión del modelo del 78.00%. Las variables seleccionadas parecen ser relevantes para el modelo, destacándose la importancia de características como education, age y hours_per_week en la predicción de ingresos.

#### Importancia de Variables según Random Forest
Los resultados del modelo de Random Forest indican que las variables más importantes son: fnlwgt (0.1636), age (0.1523), capital_gain (0.1128), relationship (0.1060), y education_num (0.0911), entre otras. Esto sugiere que ciertas variables tienen un impacto considerable en la predicción de ingresos.

#### ANOVA
En el análisis de varianza (ANOVA), las variables con p-value igual a 0, como age, education_num, relationship, sex, capital_gain, hours_per_week, marital_status y capital_loss, son estadísticamente significativas para predecir la variable objetivo. Otras variables, como education, occupation, race, workclass y native_country, presentan p-values menores a 0.05, lo que sugiere que también son significativas, aunque menos influyentes. Sin embargo, fnlwgt tiene un p-value de 0.0877, lo que indica que no es estadísticamente significativa al nivel de confianza del 95%.

#### Validación Cruzada (Con y Sin fnlwgt)
Debido a este resultado obtenido, se realizó una validación cruzada para evaluar la importancia de fnlwgt en el modelo. Con fnlwgt, la precisión promedio fue de 0.8243, mientras que sin esta variable fue de 0.8247. La inclusión de fnlwgt aporta una ligera mejora en la precisión del grupo minoritario (ingresos mayores a 50K), sugiriendo su valor en contextos específicos. La conclusión que obtenemos es que fnlwgt puede estar capturando interacciones complejas entre variables que ANOVA, al ser una técnica lineal, no puede detectar. En algunos modelos más complejos (como Random Forest), una variable puede ser importante por las interacciones que permite, pero no necesariamente tiene una fuerte relación directa con la variable objetivo.

#### SMOTE
La técnica SMOTE no mejoró el rendimiento del modelo y, en algunos casos, afectó negativamente la recall. Por lo tanto, se recomienda omitir esta técnica en el modelo final.

#### Random Forest con Ajuste de Umbral
Finalmente, al aplicar un ajuste de umbral en el modelo de Random Forest, se observó una mejora en la precisión para la clase 0, pero una disminución en el rendimiento para la clase 1. Esto sugiere que, aunque se logró un mejor balance entre las clases, esto fue a costa de la precisión general.

## Entrenamiento de Modelos - XGBoost y LightGBM
Optamos por entrenar los datos con los algoritmos XGBoost y LightGBM, debido a su capacidad para manejar grandes cantidades de datos, su rendimiento computacional eficiente y su habilidad para capturar relaciones complejas. Elegimos estos algoritmos después de probar otros modelos (Regresión Logística, Random Forest, KNN, SVM, AdaBoost, Decision Tree, Gradient Boosting) para comparar su desempeño y elegir el más adecuado, pero XGBoost y LightGBM ofrecieron mejores resultados en términos de precisión.

Con XGBoost, obtuvimos una precisión de aproximadamente 85%, mientras que LightGBM logró un 87%. Estos algoritmos son óptimos para este tipo de problemas, demostrando su capacidad para manejar relaciones no lineales y conjuntos de datos grandes.

## Optimización de Parámetros - GridSearchCV
Para maximizar el rendimiento de los modelos, utilizamos GridSearchCV, una técnica de optimización de hiperparámetros que evalúa múltiples combinaciones de parámetros para encontrar la configuración ideal. Esto asegura que los modelos funcionen con los parámetros más adecuados para el conjunto de datos específico.

Con la optimización de parámetros, tanto XGBoost como LightGBM mostraron mejoras en precisión y recall, lo que indica un ajuste fino del modelo. La mejor combinación de hiperparámetros permitió que el modelo de LightGBM lograra su mejor desempeño, con una precisión final del 88%.

### Resultados obtenidos
* Al determinar que education_num, age, y hours_per_week poseen una relación fuerte con la variante objetivo, probamos segmentando las mismas para ver si mejora la precisión del modelo.
* En muchos segmentos, especialmente aquellos con educación baja y horas extremas de trabajo (0-20 y 61-100 horas), se observa un fuerte desbalance de clases. Esto provoca que el modelo solo se enfoque en predecir correctamente la clase dominante (ingresos bajos), ignorando por completo a los individuos con ingresos altos, lo que se refleja en F1-scores bajos o inexistentes para la clase '>50K'.
* Educación y horas trabajadas influyen notablemente en el desempeño: Los segmentos con más horas de trabajo (21-40) y más años de educación (13-16) muestran mejor desempeño en la predicción de ingresos mayores a $50K. Por el contrario, aquellos con menos educación o en extremos de horas trabajadas tienden a sufrir de desbalance en el modelo.
* Muestras pequeñas impactan el desempeño: En segmentos pequeños, como 61-100 horas trabajadas o niveles educativos bajos, el modelo tiene dificultades para generalizar, lo que se traduce en predicciones poco fiables, especialmente para la clase con menos representación.

Por lo que llegamos a la conclusión de que es preferible no segmentar el dataset.
Utilizamos los modelos que mejores resultados mostraron, aplicamos GridSearchCV y RandomizedSearchCV para ajustar los hiperparámetros de los modelos LightGBM y XGBoost, 

## Evaluación del Modelo y Métricas
Creamos un Reporte de clasificación para evaluar la performance de los modelos, con parámetros como precisión (accuracy), recall, y F1-score. También generamos matrices de confusión para analizar el rendimiento del modelo en términos de falsos positivos y falsos negativos. Finalmente añadimos un gráfico de comparación de valores reales y predicciones, y una curva ROC para ambos modelos Estas métricas son clave para determinar la efectividad de los modelos en la predicción de ingresos, y comparar su eficacia en varios aspectos a la hora de realizar la predicción. (En nuestro caso realizamos varias evaluaciones debido a la ligera dificultad que presentaban todos los modelos para precedir casos del grupo minoritario)

Los resultados mostraron que LightGBM tenía un mejor balance entre precisión y recall, lo que lo convierte en el modelo más robusto. XGBoost también presentó buenos resultados, pero con una leve caída en el recall. Las métricas de evaluación confirman que LightGBM es el modelo más adecuado para este problema, debido a su capacidad de equilibrar correctamente las predicciones, minimizando tanto falsos positivos como negativos.

## Análisis de la Importancia de Características
Después de entrenar los modelos, analizamos la importancia de cada característica para determinar cuáles factores tienen mayor influencia en la predicción de ingresos superiores a $50,000. Usamos las herramientas de visualización integradas en XGBoost y LightGBM para obtener un ranking de las variables más influyentes en el modelo que mejores resultados obtuvo. De esta manera, podremos comprobar la veracidad de los insights observados en el EDA, centrándonos en las características mencionadas en las preguntas de investigación ('occupation', 'education', y 'sex').

Las variables más relevantes fueron 'occupation', 'education', y 'hours_per_week', mientras que 'sex' tuvo una influencia baja en ambos modelos. Esto sugiere que factores como el nivel educativo y el tipo de ocupación son más determinantes para predecir altos ingresos. El análisis de la importancia de características confirma que, aunque el género tiene un rol en la predicción de ingresos, las variables relacionadas con la educación, horas trabajadas y ocupación son mucho más influyentes.

# Conclusiones Finales
* Con base en los resultados obtenidos, concluimos que la combinación de reducción de dimensionalidad, elección de algoritmos avanzados como XGBoost y LightGBM, y optimización de hiperparámetros fue clave para obtener un modelo preciso y eficiente.  
* El modelo optimizado con LightGBM alcanzó la mejor precisión (88%) y mostró un buen balance en las métricas de evaluación. Los factores más influyentes para predecir ingresos superiores a $50,000 son la edad, el capital ganado y perdido, el nivel educativo, horas trabajadas y la ocupación. El análisis de este dataset permite obtener resultados a distintos niveles de profundidad, revelando patrones que pueden informar decisiones estratégicas en políticas públicas, optimización de recursos en el mercado laboral y estrategias de desarrollo personal para los individuos en búsqueda de empleo.  

   * **Políticas Gubernamentales:** Los hallazgos pueden orientar a los gobiernos en el diseño de políticas que fomenten el acceso a educación y capacitación en áreas con mayor potencial de ingresos. Invertir en programas educativos y de formación laboral que se alineen con las ocupaciones que generan mayores ingresos puede contribuir a la equidad económica y reducir la desigualdad en el acceso a oportunidades. 

   * **Mercado Laboral:** La identificación de ocupaciones y niveles educativos que correlacionan con ingresos más altos puede servir como guía para las empresas al desarrollar estrategias de contratación y capacitación. Las organizaciones pueden enfocarse en estos factores para atraer y retener talento, además de ofrecer programas de desarrollo profesional que potencien las habilidades relevantes en su fuerza laboral.  

   * **Personas en Búsqueda de Empleo:** Para quienes buscan trabajo, entender los factores que influyen en los ingresos puede ser clave en la toma de decisiones sobre su formación y trayectoria profesional. Al enfocarse en mejorar su nivel educativo, adquirir habilidades en ocupaciones demandadas y maximizar horas de trabajo, los individuos pueden aumentar sus posibilidades de acceder a empleos mejor remunerados.  
En resumen, este análisis no solo aporta información valiosa para la formulación de políticas y estrategias en el mercado laboral, sino que también empodera a los solicitantes de empleo al proporcionarles herramientas para optimizar sus oportunidades económicas.
