import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.append("/opt/airflow")

from collect_data import WeatherDataCollector
from preprocess_data import WeatherDataProcessor

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def collect_weather_data():
    collector = WeatherDataCollector()
    weather_data = collector.collect_all_cities()
    if weather_data is not None:
        collector.save_data(weather_data, "raw_data.csv")

def process_weather_data():
    processor = WeatherDataProcessor()
    raw_data = processor.load_latest_data()
    if raw_data is not None:
        processed_data = processor.process_data(raw_data)
        if processed_data is not None:
            processor.save_processed_data(processed_data, "processed_data.csv")

# Define the DAG
dag = DAG(
    'weather_pipeline',
    default_args=default_args,
    description='A pipeline to collect and process weather data',
    schedule_interval=timedelta(hours=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['weather', 'data-pipeline'],
)

# Define the tasks
collect_task = PythonOperator(
    task_id='collect_weather_data',
    python_callable=collect_weather_data,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_weather_data',
    python_callable=process_weather_data,
    dag=dag,
)

# Set task dependencies
collect_task >> process_task
