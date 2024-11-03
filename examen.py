## FUNCIONES
import pandas as pd
import numpy as np
from faker import Faker
import statsmodels.api as sm

def filter_dataframe(df: pd.DataFrame, col_name: str, umbral: float):
    """
    Filtra el DataFrame para mantener solo las filas donde el valor en la columna especificada
    es mayor que el umbral dado.
    Parametros: 
    df: DataFrame a procesar.
    columna: Nombre de la columna a evaluar.
    umbral: umbral para filtrar.
    return: DataFrame filtrado.
    """
    # Filtrar el DataFrame
    df_filtrado = df[df[col_name] > umbral]
    return df_filtrado


def generate_regresion_data(num_filas: int):
    """
    Devuelve un DataFrame con datos sintéticos y una serie para la variable dependiente.
    
    Parametros:
    num_filas: Número de filas a generar.
    return: DataFrame con datos sintéticos y serie para la variable dependiente.
    """
    fake = Faker()
    
    # Listas para almacenar los datos
    edades, ingresos, gastos = [], [], []
    
    # Generar datos
    for _ in range(num_filas):
        edad = fake.random_int(min=18, max=35)  # Edad entre 18 y 70
        ingreso = fake.random_int(min=10000, max=60000)  # Ingreso entre 20,000 y 100,000
        gasto = 0.1*ingreso + 10*edad + fake.random_int(-50, 50) 
        
        edades.append(edad)
        ingresos.append(ingreso)
        gastos.append(gasto)

    # Crear DataFrame
    df = pd.DataFrame({
        'Edad': edades,
        'Ingreso': ingresos,
    })
    
    # Crear la serie para la variable dependiente
    serie_gasto = pd.Series(gastos, name='Gasto')

    return df, serie_gasto

def train_multiple_linear_regression(df_independientes: pd.DataFrame , serie_dependiente: pd.Series):
  """
  Entrena un modelo de regresión lineal multiple.
  Parametros:
  df_independientes: DataFrame con las variables independientes.
  serie_dependiente: Serie con la variable dependiente.
  return: Modelo de regresión lineal entrenado.
  """
  modelo = sm.OLS(serie_dependiente, sm.add_constant(df_independientes))
  resultados = modelo.fit()
  return resultados

def flatten_list(lista: list):
  """
  Devuelve una lista unica a partir de una lista de listas.
  Parametros:
  lista: Lista de listas.
  return: Lista unidimensional.
  """
  lista_unica = [ elemento for sublista in lista_listas for elemento in sublista ]
  return lista_unica


def group_and_aggregate(dataframe: pd.DataFrame, group_by: str, column_agre: str):
  """
  Agrupa un DataFrame por una columna y calcula la media de otra columna.
  Parametros:
  dataframe: DataFrame a procesar.
  group_by: Nombre de la columna por la cual agrupar.
  column_agre: Nombre de la columna a promediar.
  return: DataFrame con la media de la columna especificada.
  """
  resultado = df.groupby(group_by)[column_agre].mean().reset_index()
  return resultado

def train_logistic_regression(df_independientes: pd.DataFrame , serie_dependiente: pd.Series):
  """
  Entrena un modelo de regresión logistica.
  Parametros:
  df_independientes: DataFrame con las variables independientes.
  serie_dependiente: Serie con la variable dependiente.
  return: Modelo de regresión lineal entrenado.
  """
  modelo = sm.Logit(serie_dependiente, df_independientes)
  resultados = modelo.fit()
  return resultados

def apply_function_to_column(df: pd.DataFrame, col_name:str, funcion):
  """
  Aplica una función a una columna específica de un DataFrame.
  Parametros:
  df: DataFrame a procesar.
  col_name: Nombre de la columna a procesar.
  funcion:
  """
  df[col_name] = df[col_name].apply(funcion)
  return df

def filter_and_square(lista: list):
  """
  Filtra los numeros mayores que cinco y devuelve los cuadrados de esos numeros
  Parametros:
  lista: Lista de numeros
  return: Lista de numeros cuadrados
  """
  lista_filtrada = [x**2 for x in lista if x > 5]
  return lista_filtrada