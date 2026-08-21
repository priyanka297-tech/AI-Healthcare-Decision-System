import os
from pathlib import Path

import streamlit as st
import mysql.connector
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# LOAD CONFIGURATION
# LOCAL:
#     Uses .env
#
# STREAMLIT COMMUNITY CLOUD:
#     Uses st.secrets
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# Load .env for local development
load_dotenv(ENV_FILE, override=False)


def get_config(key, default=None):
    """
    Get configuration value.

    Priority:
    1. Streamlit Secrets
    2. Environment variable / .env
    3. Default value
    """

    try:
        # Streamlit Community Cloud
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # st.secrets may not be configured locally
        pass

    # Local .env / environment variable
    return os.getenv(key, default)


# ============================================================
# MYSQL DATABASE CONFIGURATION
# ============================================================

MYSQL_CONFIG = {
    "host": get_config("MYSQL_HOST", "localhost"),
    "port": int(get_config("MYSQL_PORT", "3306")),
    "user": get_config("MYSQL_USER", "root"),
    "password": get_config("MYSQL_PASSWORD", ""),
    "database": get_config("MYSQL_DATABASE", "hospital"),
}


# ============================================================
# CHECK DATABASE CONFIGURATION
# ============================================================

def check_database_config():

    required_values = [
        "host",
        "user",
        "database",
    ]

    missing_values = []

    for value in required_values:

        if not MYSQL_CONFIG.get(value):

            missing_values.append(value)

    if missing_values:

        raise ValueError(
            "Missing MySQL configuration: "
            + ", ".join(missing_values)
            + ". Check your .env file or Streamlit Secrets."
        )


# ============================================================
# CREATE DATABASE CONNECTION
# ============================================================

def connect_db():

    check_database_config()

    try:

        connection = mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            database=MYSQL_CONFIG["database"],
            connection_timeout=10,
        )

        return connection

    except mysql.connector.Error as e:

        raise RuntimeError(
            f"MySQL connection failed: {e}"
        )


# ============================================================
# CREATE PATIENT HISTORY TABLE
# ============================================================

def create_table():

    conn = None
    cursor = None

    try:

        conn = connect_db()

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_history (

                id INT AUTO_INCREMENT PRIMARY KEY,

                patient_name VARCHAR(255) NOT NULL,

                age INT,

                gender VARCHAR(50),

                disease_group VARCHAR(255),

                survival_probability FLOAT,

                death_probability FLOAT,

                prediction VARCHAR(255),

                created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP

            )
            """
        )

        conn.commit()

        return True

    except mysql.connector.Error as e:

        print(
            f"Database table creation error: {e}"
        )

        return False

    finally:

        if cursor is not None:
            cursor.close()

        if conn is not None and conn.is_connected():
            conn.close()


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    patient_name,
    age,
    gender,
    disease_group,
    survival_probability,
    death_probability,
    prediction,
):

    conn = None
    cursor = None

    try:

        conn = connect_db()

        cursor = conn.cursor()

        query = """
            INSERT INTO patient_history (
                patient_name,
                age,
                gender,
                disease_group,
                survival_probability,
                death_probability,
                prediction
            )

            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
        """

        values = (
            patient_name,
            age,
            gender,
            disease_group,
            survival_probability,
            death_probability,
            prediction,
        )

        cursor.execute(
            query,
            values,
        )

        conn.commit()

    except mysql.connector.Error as e:

        if conn is not None:
            conn.rollback()

        raise RuntimeError(
            f"Unable to save prediction: {e}"
        )

    finally:

        if cursor is not None:
            cursor.close()

        if conn is not None and conn.is_connected():
            conn.close()


# ============================================================
# GET ALL PATIENT PREDICTIONS
# ============================================================

def get_all_predictions():

    conn = None

    try:

        conn = connect_db()

        query = """
            SELECT
                id,
                patient_name,
                age,
                gender,
                disease_group,
                survival_probability,
                death_probability,
                prediction,
                created_at

            FROM patient_history

            ORDER BY created_at DESC
        """

        df = pd.read_sql(
            query,
            conn,
        )

        return df

    except Exception as e:

        raise RuntimeError(
            f"Unable to retrieve patient history: {e}"
        )

    finally:

        if conn is not None and conn.is_connected():
            conn.close()


# ============================================================
# DELETE PREDICTION
# ============================================================

def delete_prediction(record_id):

    conn = None
    cursor = None

    try:

        conn = connect_db()

        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM patient_history
            WHERE id = %s
            """,
            (record_id,),
        )

        conn.commit()

        return cursor.rowcount

    except mysql.connector.Error as e:

        if conn is not None:
            conn.rollback()

        raise RuntimeError(
            f"Unable to delete prediction: {e}"
        )

    finally:

        if cursor is not None:
            cursor.close()

        if conn is not None and conn.is_connected():
            conn.close()


# ============================================================
# SEARCH PATIENT
# ============================================================

def search_patient(name):

    conn = None

    try:

        conn = connect_db()

        query = """
            SELECT
                id,
                patient_name,
                age,
                gender,
                disease_group,
                survival_probability,
                death_probability,
                prediction,
                created_at

            FROM patient_history

            WHERE patient_name LIKE %s

            ORDER BY created_at DESC
        """

        df = pd.read_sql(
            query,
            conn,
            params=(f"%{name}%",),
        )

        return df

    except Exception as e:

        raise RuntimeError(
            f"Unable to search patient: {e}"
        )

    finally:

        if conn is not None and conn.is_connected():
            conn.close()


# ============================================================
# CREATE TABLE
# ============================================================
#
# Call this from your main Streamlit application after
# the app has started, preferably with error handling.
#
# Example:
#
# try:
#     create_table()
# except Exception as e:
#     st.warning(f"Database initialization failed: {e}")
#
# ============================================================