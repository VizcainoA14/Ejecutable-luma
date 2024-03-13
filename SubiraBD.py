from sqlalchemy import text
import pandas as pd
from sqlalchemy import create_engine
import os
from configparser import ConfigParser


CONFIG = ".config"


def set_up():

    env = os.getenv("ENV", CONFIG)

    if env == CONFIG:
        config = ConfigParser()
        config.read(CONFIG)
        config = config["DATA"]
    else:
        config = {
            "host": os.getenv("DB_HOST"),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": os.getenv("DB_PORT"),
        }
    return config



class SubirDB:

    def __init__(self):
        self.config = set_up()

        DBNAME = self.config['POSTGRES_DATABASE']
        USER = self.config['POSTGRES_USER']
        PASSWORD = self.config['POSTGRES_PASSWORD']
        HOST = self.config['POSTGRES_HOST']
        PORT = "5432"

        self.engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}')




    def max_date(self, table):
        query = f'SELECT MAX(date) FROM {table};'
        Max_date = pd.read_sql_query(query, self.engine)['max'][0]
        return Max_date

    def create_tables(self):
        tables = [
            'eit171', 'eit195', 'eit284', 'eit304', 'hmiigr', 'hmimag'
        ]
        
        for table in tables:
            # Leer el DataFrame desde el archivo CSV
            df_csv = pd.read_csv(f'./DATA/output_2024_{table}.csv')
            
            # Consultar la última fecha en la base de datos
            last_date_query = f'SELECT MAX(date) FROM {table};'
            last_date = pd.read_sql_query(last_date_query, self.engine)['max'][0]

            # Filtrar solo las filas con fechas mayores a la última fecha en la base de datos
            df_new_data = df_csv[df_csv['date'] > last_date]

            if not df_new_data.empty:
                # Si hay nuevas fechas, guardar en la base de datos
                df_new_data.to_sql(table, self.engine, if_exists='append', index=False)
                print(f'Se agregaron {len(df_new_data)} nuevas filas a la tabla {table}.')

x = SubirDB()
