import redis
from rq import Queue, Worker
from app.cache_redis.redis_engine import RedisEngine

redis_engine = RedisEngine()


def main():
    print("Hello from tfg-worker!")
    
    queue_list = [redis_engine.get_queue()]

    connection = redis_engine.get_connection()

    print(" Starting worker...")
    print(f" Connecting to Redis")

    worker = Worker(
        queue_list, 
        connection=connection, 
        name='tfg-worker'
    )

    worker.work()


if __name__ == "__main__":
    main()
