from .config import DB_SETTINGS
# Author: wujiahang
from .db_pool import MySQLConnectionPool

POOL = MySQLConnectionPool(host=DB_SETTINGS["host"], port=DB_SETTINGS["port"], user=DB_SETTINGS["user"], password=DB_SETTINGS["password"], db=DB_SETTINGS["name"], charset="utf8mb4")


def init_db():
    conn = POOL.get_connection()
    cur = conn.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS posecoach CHARACTER SET utf8mb4")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS analyses(
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        user_id VARCHAR(64) DEFAULT 'guest',
        action_type VARCHAR(64),
        template_video VARCHAR(512),
        user_video VARCHAR(512),
        score FLOAT,
        advice TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS analysis_images(
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        analysis_id BIGINT,
        template_image VARCHAR(512),
        user_image VARCHAR(512),
        `desc` TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""")
    cur.close()
    conn.close()


def insert_analysis(user_id, action_type, template_video, user_video, score, advice):
    conn = POOL.get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO analyses(user_id, action_type, template_video, user_video, score, advice)
           VALUES(%s,%s,%s,%s,%s,%s)""",
        (user_id, action_type, template_video, user_video, float(score), advice),
    )
    cur.execute("SELECT LAST_INSERT_ID()")
    rid = cur.fetchone()[0]
    cur.close()
    conn.close()
    return rid


def insert_image(analysis_id, template_image, user_image, desc=""):
    """插入对比图，同时保存描述"""
    conn = POOL.get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO analysis_images(analysis_id, template_image, user_image, `desc`)
           VALUES(%s,%s,%s,%s)""",
        (analysis_id, template_image, user_image, desc),
    )
    cur.close()
    conn.close()


def list_analyses(limit=50):
    conn = POOL.get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, action_type, score FROM analyses ORDER BY id DESC LIMIT %s",
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_analysis_detail(analysis_id):
    conn = POOL.get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, action_type, template_video, user_video, score, advice "
        "FROM analyses WHERE id=%s",
        (analysis_id,),
    )
    head = cur.fetchone()
    cur.execute(
        "SELECT template_image, user_image, `desc` FROM analysis_images WHERE analysis_id=%s",
        (analysis_id,),
    )
    imgs = cur.fetchall()
    cur.close()
    conn.close()
    return head, imgs