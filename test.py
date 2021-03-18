import time
import threading
import numpy as np
from milvus import Milvus, IndexType, Status
from milvus.client.types import MetricType

SERVER_ADDR = "127.0.0.1"
SERVER_PORT = '19531'

COLLECTION_NAME = 'TEST'
COLLECTION_DIMENSION = 512

SEARCH_PARAM = {'ef': 256}

MILVUS = Milvus(host=SERVER_ADDR, port=SERVER_PORT)

def create_collection():
    MILVUS.drop_collection(collection_name=COLLECTION_NAME)
    tb_param = {'collection_name': COLLECTION_NAME, 'dimension': COLLECTION_DIMENSION, 'index_file_size': 1024,
                'metric_type': MetricType.L2}
    status = MILVUS.create_collection(tb_param)
    
    # create hnsw index
    status = MILVUS.create_index(COLLECTION_NAME, IndexType.HNSW, {'M': 24, 'efConstruction': 256})

def gen_vec_list(nb, seed=np.random.RandomState(1234)):
    xb = seed.rand(nb, COLLECTION_DIMENSION).astype("float32")
    vec_list = xb.tolist()
    return vec_list

def insert():
    batch_count = 10
    vec_list = gen_vec_list(batch_count)
    for k in range(100000):
        if k % 100 == 0:
            print("insert...", k)
        vec_ids = [k * batch_count + i for i in range(batch_count)]
        status, result = MILVUS.insert(collection_name=COLLECTION_NAME, records=vec_list, ids=vec_ids)

    # Use only autoflush (1s)
    #status = MILVUS.flush(collection_name_array=[COLLECTION_NAME])


def search():
    for k in range(10000):
        if k%100 == 0:
            print("search...", k)
        query_vectors = gen_vec_list(nb=1)
        status, results = MILVUS.search(collection_name=COLLECTION_NAME, query_records=query_vectors,
                                        top_k=3, params=SEARCH_PARAM)
        if status.code != Status.SUCCESS:
            print('search error', status.message)

if __name__ == "__main__":
    create_collection()

    threads = []

    for i in range(5):
        x = threading.Thread(target=insert, args=())
        threads.append(x)
        x.start()

    for i in range(10):
        x = threading.Thread(target=search, args=())
        threads.append(x)
        x.start()

    for th in threads:
        th.join()