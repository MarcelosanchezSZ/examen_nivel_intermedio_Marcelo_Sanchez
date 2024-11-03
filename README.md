# EXAMEN_NIVEL_INTERMEDIO_MARCELO_SANCHEZ

## Descripción

Este proyecto contiene un conjunto de funciones diseñadas para evaluar el nivel de python

## Funciones

### Función 1: filter_dataframe(df: pd.DataFrame, col_name: str, umbral: float)

**Descripción**: 
Filtra el DataFrame para mantener solo las filas donde el valor en la columna especificada
    es mayor que el umbral dado.

**Parámetros**:
- df: DataFrame
- col_name: str
- umbral: float

**Retorno**: 
- df: DataFrame

**Ejemplo**:
```python
import pandas as pd
import numpy as np
from examen import filter_dataframe

data = {
    'nombre': ['Marcelo', 'Ashley', 'Pedro', 'Carmen'],
    'edad': [29, 25, 22, 28],
    'salario': [35250.80, 67834.0, 387362.0, 76254.0]}
df = pd.DataFrame(data)
print(df)
print('\n')

resultado = filter_dataframe(df, 'salario', 40000.0)
print(resultado) 
```

### Funcion 2: generar_datos_regresion(num_filas: int)

**Descripción**:
Devuelve un DataFrame con datos sintéticos y una serie para la variable dependiente.

**Parámetros**:
num_filas: DataFrame

**Retorno**:
- df: DataFrame

**Ejemplo**:
```python
import pandas as pd
import numpy as np
from examen import generate_datos_regresion

df_independientes, serie_dependiente = generate_datos_regresion(100)
print(df_independientes.head())
print('\n')
print(serie_dependiente.head()) 
```

## Funcion 3: train_multiple_linear_regression(df_independientes: pd.DataFrame , serie_dependiente: pd.Series)
**Descripción**:
Entrena un modelo de regresión lineal multiple.

**Parámetros**:
  df_independientes: DataFrame con las variables independientes.
  serie_dependiente: Serie con la variable dependiente.

**Retorno**:
  return: Modelo de regresión lineal entrenado.

**Ejemplo**:
```python
import pandas as pd
import numpy as np
from examen import train_multiple_linear_regression

modelo = train_multiple_linear_regression(df_independientes, serie_dependiente)
print(modelo.summary()) 
```

### Funcion 4: flatten_list(lista: list)
**Descripción**:
  Devuelve una lista unica a partir de una lista de listas.

**Parámetros**:
  lista: Lista de listas.
  
**Retorno**:
  return: Lista unidimensional.

**Ejemplo**:
```python
import pandas as pd
import numpy as np
from examen import flatten_list

lista_listas = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
lista_unica = flatten_list(lista_listas)
print(lista_unica) 
```

### Funcion 5: group_and_aggregate(dataframe: pd.DataFrame, group_by: str, column_agre: str)
**Descripción**:
Agrupa un DataFrame por una columna y calcula la media de otra columna.
  
**Parámetros**:
  dataframe: DataFrame a procesar.
  group_by: Nombre de la columna por la cual agrupar.
  column_agre: Nombre de la columna a promediar.

**Retorno**:
  return: DataFrame con la media de la columna especificada.

**Ejemplo**:
```python
import pandas as pd
import numpy as np
from examen import group_and_aggregate

data = {
    'Categoria': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Valor': [10, 20, 30, 40, 50, 60],
    'Otro': [1, 2, 3, 4, 5, 6]
}
df = pd.DataFrame(data)
print(df)
print('\n')

df_agrupado = group_and_aggregate(df, 'Categoria', 'Valor')
print(df_agrupado) 
```

### Funcion 6: train_logistic_regression(df_independientes: pd.DataFrame , serie_dependiente: pd.Series)
**Descripción**:
  Entrena un modelo de regresión logistica.
**Parámetros**:
  df_independientes: DataFrame con las variables independientes.
  serie_dependiente: Serie con la variable dependiente.

**Retorno**:
  return: Modelo de regresión lineal entrenado.

**Ejemplo**:
```python
import pandas as pd
import numpy as np
from examen import train_logistic_regression

# Crear un DataFrame de ejemplo
np.random.seed(42)  # Para reproducibilidad

data = {
    'Edad': np.random.randint(18, 70, size=100),  # Edad aleatoria entre 18 y 70
    'Ingreso': np.random.randint(20000, 100000, size=100),  # Ingreso aleatorio
}

df = pd.DataFrame(data)
y = pd.Series(np.random.choice([0, 1], size=100))
modelo = train_logistic_regression(df, y)
print(modelo.summary()) 
```


### Funcion 7: apply_function_to_column(df: pd.DataFrame, col_name:str, funcion):

**Descripción**:
  Aplica una función a una columna específica de un DataFrame.

**Parámetros**:
  df: DataFrame a procesar.
  col_name: Nombre de la columna a procesar.
  funcion: funcion especifica

**Retorno**:
  return: DataFrame modificado

**Ejemplo**:
```python
import pandas as pd
import numpy as np
from examen import apply_function_to_column

def cuadrado(x):
    """Devuelve el cuadrado de un número."""
    return x ** 2

# Crear un DataFrame de ejemplo
data = {
    'Números': [1, 2, 3, 4, 5],
    'Otras_columnas': ['a', 'b', 'c', 'd', 'e']
}
df = pd.DataFrame(data)

# Mostrar el DataFrame original
print("DataFrame Original:")
print(df)

# Llamar a la función para aplicar la función personalizada
df_modificado = apply_function_to_column(df, 'Números', cuadrado)

# Mostrar el DataFrame modificado
print("\nDataFrame Modificado:")
print(df_modificado) 
```

### Funcion 8: filter_and_square(lista: list):
   
**Descripción**:
  Filtra los numeros mayores que cinco y devuelve los cuadrados de esos numeros
  
**Parámetros**:
  lista: Lista de numeros

**Retorno**:
  return: Lista de numeros cuadrados

**Ejemplo**:
```python
import pandas as pd
import numpy as np
from examen import filter_and_square

lista = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
lista_filtrada = filter_and_square(lista)
print(lista_filtrada) 
```
