# Author: wujiahang
from DBUtils.PooledDB import PooledDB
import pymysql


class MySQLConnectionPool:
    def __init__(self, host, port, user, password, db, charset="utf8mb4", mincached=1, maxcached=5, maxconnections=10):
        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=maxconnections,
            mincached=mincached,
            maxcached=maxcached,
            blocking=True,
            host=host,
            port=port,
            user=user,
            password=password,
            database=db,
            charset=charset,
            autocommit=True
        )

    def get_connection(self):
        return self.pool.connection()