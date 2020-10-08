from agents.sac_models import *
from agents.memory_dataloading import *
# from threading import Lock, Thread
from agents.util import collate, partition, partial

from copy import deepcopy
# import socket
import select
from requests import get
import pickle
from argparse import ArgumentParser
# import time
from gym_tmrl.envs.tmrl_env import *


# OBSERVATION PREPROCESSING ==================================

def obs_preprocessor_tmnf_act_in_obs(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    # print(f"DEBUG1: len(obs):{len(obs)}, obs[0]:{obs[0]}, obs[1].shape:{obs[1].shape}, obs[2]:{obs[2]}")
    # obs = (obs[0] / 1000.0, np.moveaxis(obs[1], -1, 0) / 255.0, obs[2])
    obs = (obs[0], np.moveaxis(obs[1], -1, 1), obs[2])
    # print(f"DEBUG2: len(obs):{len(obs)}, obs[0]:{obs[0]}, obs[1].shape:{obs[1].shape}, obs[2]:{obs[2]}")
    # exit()
    return obs


def obs_preprocessor_tmnf_lidar_act_in_obs(obs):
    """
    This takes the output of gym as input
    Therefore the output of the memory must be the same as gym
    """
    obs = (obs[0], np.ndarray.flatten(obs[1]), obs[2])
    return obs


# CONFIGURATION: ==========================================

PRAGMA_EDOUARD_YANN = False  # True if Edouard, False if Yann
PRAGMA_TM2020_TMNF = False  # True if TM2020, False if TMNF
PRAGMA_CUDA = False  # True if CUDA, False if CPU

TRAIN_MODEL = Tm_hybrid_1 if PRAGMA_TM2020_TMNF else Mlp
POLICY = TMPolicy if PRAGMA_TM2020_TMNF else MlpPolicy
OBS_PREPROCESSOR = obs_preprocessor_tmnf_act_in_obs if PRAGMA_TM2020_TMNF else obs_preprocessor_tmnf_lidar_act_in_obs

ACT_IN_OBS = True
BENCHMARK = False

public_ip = get('http://api.ipify.org').text
local_ip = socket.gethostbyname(socket.gethostname())
print(f"I: local IP: {local_ip}")
print(f"I: public IP: {public_ip}")
REDIS_IP = public_ip
LOCALHOST = False

PORT_TRAINER = 55555  # Port to listen on (non-privileged ports are > 1023)
PORT_ROLLOUT = 55556  # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 268435456  # 1048576  # 8192  # 32768  # socket buffer size (200 000 000 is large enough for 1000 images right now)
HEADER_SIZE = 12  # fixed number of characters used to describe the data length

SOCKET_TIMEOUT_CONNECT_TRAINER = 60.0
SOCKET_TIMEOUT_ACCEPT_TRAINER = 60.0
SOCKET_TIMEOUT_CONNECT_ROLLOUT = 60.0
SOCKET_TIMEOUT_ACCEPT_ROLLOUT = 60.0  # socket waiting for rollout workers closed and restarted at this interval
SOCKET_TIMEOUT_COMMUNICATE = 30.0
SELECT_TIMEOUT_OUTBOUND = 30.0
# PING_INTERVAL = 60.0  # interval at which the server pings the clients
SELECT_TIMEOUT_PING_PONG = 60.0
ACK_TIMEOUT_WORKER_TO_REDIS = 60.0
ACK_TIMEOUT_TRAINER_TO_REDIS = 60.0
ACK_TIMEOUT_REDIS_TO_WORKER = 60.0
ACK_TIMEOUT_REDIS_TO_TRAINER = 60.0
WAIT_BEFORE_RECONNECTION = 10.0
LOOP_SLEEP_TIME = 1.0

MODEL_PATH_WORKER = r"D:\cp\weights\expt.pth" if PRAGMA_EDOUARD_YANN else r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\weights\expt.pth"
MODEL_PATH_TRAINER = r"D:\cp\weights\exp.pth" if PRAGMA_EDOUARD_YANN else r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\weights\exp.pth"
CHECKPOINT_PATH = r"D:\cp\exp0" if PRAGMA_EDOUARD_YANN else r"C:\Users\Yann\Desktop\git\tmrl\checkpoint\chk\exp0"
DATASET_PATH = r"D:\data2020"  if PRAGMA_EDOUARD_YANN else r"C:\Users\Yann\Desktop\git\tmrl\data"

MEMORY = partial(MemoryTM2020 if PRAGMA_TM2020_TMNF else MemoryTMNFLidar,
                 path_loc=DATASET_PATH,
                 imgs_obs=4 if PRAGMA_TM2020_TMNF else 1,
                 act_in_obs=ACT_IN_OBS,
                 obs_preprocessor=OBS_PREPROCESSOR
                 )

CONFIG_DICT = {
    "interface": TM2020Interface if PRAGMA_TM2020_TMNF else partial(TMInterfaceLidar, img_hist_len=1),
    "time_step_duration": 0.05,
    "start_obs_capture": 0.04,
    "time_step_timeout_factor": 1.0,
    "ep_max_length": np.inf,
    "real_time": True,
    "async_threading": True,
    "act_in_obs": ACT_IN_OBS,
    "benchmark": BENCHMARK,

}


# NETWORK: ==========================================

def ping_pong(sock):
    """
    This pings and waits for pong
    All inbound select() calls must expect to receive PINGPING
    returns True if success, False otherwise
    closes socket if failed
    # FIXME: do not use, the pingpong mechanism interferes with transfers
    """
    _, wl, xl = select.select([], [sock], [sock], SELECT_TIMEOUT_OUTBOUND)  # select for writing
    if len(xl) != 0 or len(wl) == 0:
        print("INFO: socket error/timeout while sending PING")
        sock.close()
        return False
    send_ping(sock)
    rl, _, xl = select.select([sock], [], [sock], SELECT_TIMEOUT_PING_PONG)  # select for reading
    if len(xl) != 0 or len(rl) == 0:
        print("INFO: socket error/timeout while waiting for PONG")
        sock.close()
        return False
    obj = recv_object(sock)
    if obj == 'PINGPONG':
        return True
    else:
        print("INFO: PINGPONG received an object that is not PING or PONG")
        sock.close()
        return False


def send_ping(sock):
    return send_object(sock, None, ping=True, pong=False, ack=False)


def send_pong(sock):
    return send_object(sock, None, ping=False, pong=True, ack=False)


def send_ack(sock):
    return send_object(sock, None, ping=False, pong=False, ack=True)


def send_object(sock, obj, ping=False, pong=False, ack=False):
    """
    If ping, this will ignore obj and send the PING request
    If pong, this will ignore obj and send the PONG request
    If ack, this will ignore obj and send the ACK request
    If raw, obj must be a binary string
    Call only after select on a socket with a (long enough) timeout.
    Returns True if sent successfully, False if connection lost.
    """
    if ping:
        msg = bytes(f"{'PING':<{HEADER_SIZE}}", 'utf-8')
    elif pong:
        msg = bytes(f"{'PONG':<{HEADER_SIZE}}", 'utf-8')
    elif ack:
        msg = bytes(f"{'ACK':<{HEADER_SIZE}}", 'utf-8')
    else:
        msg = pickle.dumps(obj)
        msg = bytes(f"{len(msg):<{HEADER_SIZE}}", 'utf-8') + msg
    try:
        nb_bytes = len(msg) - HEADER_SIZE
        # if nb_bytes > 0:
        #     print(f"DEBUG: sending object of {nb_bytes} bytes")
        t_start = time.time()
        sock.sendall(msg)
        # if nb_bytes > 0:
        #     print(f"DEBUG: finished sending after {time.time() - t_start}s")
    except OSError:  # connection closed or broken
        return False
    return True


def recv_object(sock):
    """
    If the request is PING or PONG, this will return 'PINGPONG'
    If the request is ACK, this will return 'ACK'
    If the request is PING, this will automatically send the PONG answer
    Call only after select on a socket with a (long enough) timeout.
    Returns the object if received successfully, None if connection lost.
    This sends the ACK request back to sock when an object transfer is complete
    """
    # first, we receive the header (inefficient but prevents collisions)
    msg = b''
    l = len(msg)
    while l != HEADER_SIZE:
        try:
            recv_msg = sock.recv(HEADER_SIZE - l)
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print(f"DEBUG: l:{l}")
    # print("DEBUG: data len:", msg[:HEADER_SIZE])
    # print(f"DEBUG: msg[:4]: {msg[:4]}")
    if msg[:4] == b'PING' or msg[:4] == b'PONG':
        if msg[:4] == b'PING':
            send_pong(sock)
        return 'PINGPONG'
    if msg[:3] == b'ACK':
        return 'ACK'
    msglen = int(msg[:HEADER_SIZE])
    # print(f"DEBUG: receiving {msglen} bytes")
    t_start = time.time()
    # now, we receive the actual data (no more than the data length, again to prevent collisions)
    msg = b''
    l = len(msg)
    while l != msglen:
        try:
            recv_msg = sock.recv(min(BUFFER_SIZE, msglen - l))  # this will not receive more bytes than required
            if len(recv_msg) == 0:  # connection closed or broken
                return None
            msg += recv_msg
        except OSError:  # connection closed or broken
            return None
        l = len(msg)
        # print(f"DEBUG2: l:{l}")
    # print("DEBUG: final data len:", l)
    # print(f"DEBUG: finished receiving after {time.time() - t_start}s.")
    send_ack(sock)
    return pickle.loads(msg)


def get_listening_socket(timeout, ip_bind, port_bind):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.bind((ip_bind, port_bind))
    s.listen(5)
    return s


def get_connected_socket(timeout, ip_connect, port_connect):
    """
    returns the connected socket
    returns None if connect failed
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip_connect, port_connect))
    except OSError:  # connection broken or timeout
        print(f"INFO: connect() timed-out or failed, sleeping {WAIT_BEFORE_RECONNECTION}s")
        s.close()
        time.sleep(WAIT_BEFORE_RECONNECTION)
        return None
    s.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
    return s


def accept_or_close_socket(s):
    """
    returns conn, addr
    None None in case of failure
    """
    conn = None
    try:
        conn, addr = s.accept()
        conn.settimeout(SOCKET_TIMEOUT_COMMUNICATE)
        return conn, addr
    except OSError:
        # print(f"INFO: accept() timed-out or failed, sleeping {WAIT_BEFORE_RECONNECTION}s")
        if conn is not None:
            conn.close()
        s.close()
        time.sleep(WAIT_BEFORE_RECONNECTION)
        return None, None


def select_and_send_or_close_socket(obj, conn):
    """
    Returns True if success
    False if disconnected (closes sockets)
    """
    _, wl, xl = select.select([], [conn], [conn], SELECT_TIMEOUT_OUTBOUND)  # select for writing
    if len(xl) != 0:
        print("INFO: error when writing, closing socket")
        conn.close()
        return False
    if len(wl) == 0:
        print("INFO: outbound select() timed out, closing socket")
        conn.close()
        return False
    elif not send_object(conn, obj):  # error or timeout
        print("INFO: send_object() failed, closing socket")
        conn.close()
        return False
    return True


def poll_and_recv_or_close_socket(conn):
    """
    Returns True, obj is success (obj is None if nothing was in the read buffer when polling)
    False, None otherwise
    """
    rl, _, xl = select.select([conn], [], [conn], 0.0)  # polling read channel
    if len(xl) != 0:
        print("INFO: error when polling, closing sockets")
        conn.close()
        return False, None
    if len(rl) == 0:  # nothing in the recv buffer
        return True, None
    obj = recv_object(conn)
    if obj is None:  # socket error
        print("INFO: error when receiving object, closing sockets")
        conn.close()
        return False, None
    elif obj == 'PINGPONG':
        return True, None
    else:
        # print(f"DEBUG: received obj:{obj}")
        return True, obj


# BUFFER: ===========================================

class Buffer:
    def __init__(self):
        self.memory = []

    def append_sample(self, sample):
        self.memory.append(sample)

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        return self


# REDIS SERVER: =====================================

class RedisServer:
    """
    This is the main server
    This lets 1 TrainerInterface and n RolloutWorkers connect
    This buffers experiences sent by RolloutWorkers
    This periodically sends the buffer to the TrainerInterface
    This also receives the weights from the TrainerInterface and broadcast them to the connected RolloutWorkers
    If localhost, the ip is localhost. Otherwise, it is the public ip and requires fort forwarding.
    """
    def __init__(self, samples_per_redis_batch=1000, localhost=True):
        self.__buffer = Buffer()
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.samples_per_redis_batch = samples_per_redis_batch
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.ip = '127.0.0.1' if localhost else self.local_ip

        print(f"INFO REDIS: local IP: {self.local_ip}")
        print(f"INFO REDIS: public IP: {self.public_ip}")
        print(f"INFO REDIS: IP: {self.ip}")

        Thread(target=self.__rollout_workers_thread, args=(), kwargs={}, daemon=True).start()
        Thread(target=self.__trainer_thread, args=(), kwargs={}, daemon=True).start()

    def __trainer_thread(self, ):
        """
        This waits for a TrainerInterface to connect
        Then, this periodically sends the local buffer to the TrainerInterface (when data is available)
        When the TrainerInterface sends new weights, this broadcasts them to all connected RolloutWorkers
        """
        ack_time = time.time()
        wait_ack = False
        while True:  # main redis loop
            s = get_listening_socket(SOCKET_TIMEOUT_ACCEPT_TRAINER, self.ip, PORT_TRAINER)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                print("DEBUG: accept_or_close_socket failed in trainer thread")
                continue
            # last_ping = time.time()
            print(f"INFO TRAINER THREAD: redis connected by trainer at address {addr}")
            # Here we could spawn a Trainer communication thread, but since there is only one trainer we move on
            i = 0
            while True:
                # ping client
                # if time.time() - last_ping >= PING_INTERVAL:
                #     print("INFO: sending ping to trainer")
                #     if ping_pong(conn):
                #         last_ping = time.time()
                #     else:
                #         print("INFO: ping to trainer client failed")
                #         break
                # send samples
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_redis_batch:
                    if not wait_ack:
                        obj = self.__buffer
                        print(f"INFO: sending obj {i}")
                        if select_and_send_or_close_socket(obj, conn):
                            wait_ack = True
                            ack_time = time.time()
                        else:
                            print("INFO: failed sending object to trainer")
                            self.__buffer_lock.release()
                            break
                        self.__buffer.clear()
                    else:
                        elapsed = time.time() - ack_time
                        print(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_REDIS_TO_TRAINER:
                            print("INFO: ACK timed-out, breaking connection")
                            self.__buffer_lock.release()
                            break
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for weights
                success, obj = poll_and_recv_or_close_socket(conn)
                if not success:
                    print("DEBUG: poll failed in trainer thread")
                    break
                elif obj is not None and obj != 'ACK':
                    print(f"DEBUG INFO: trainer thread received obj")
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                time.sleep(LOOP_SLEEP_TIME)  # TODO: adapt
                i += 1
            s.close()

    def __rollout_workers_thread(self):
        """
        This waits for new potential RolloutWorkers to connect
        When a new RolloutWorker connects, this instantiates a new thread to handle it
        """
        while True:  # main redis loop
            s = get_listening_socket(SOCKET_TIMEOUT_ACCEPT_ROLLOUT, self.ip, PORT_ROLLOUT)
            conn, addr = accept_or_close_socket(s)
            if conn is None:
                # print("DEBUG: accept_or_close_socket failed in workers thread")
                continue
            print(f"INFO WORKERS THREAD: redis connected by worker at address {addr}")
            Thread(target=self.__rollout_worker_thread, args=(conn, ), kwargs={}, daemon=True).start()  # we don't keep track of this for now
            s.close()

    def __rollout_worker_thread(self, conn):
        """
        Thread handling connection to a single RolloutWorker
        """
        # last_ping = time.time()
        ack_time = time.time()
        wait_ack = False
        while True:
            # ping client
            # if time.time() - last_ping >= PING_INTERVAL:
            #     print("INFO: sending ping to worker")
            #     if ping_pong(conn):
            #         last_ping = time.time()
            #     else:
            #         print("INFO: ping to trainer client failed")
            #         break
            # send weights
            self.__weights_lock.acquire()  # WEIGHTS LOCK...............................................................
            if self.__weights is not None:  # new weigths
                if not wait_ack:
                    obj = self.__weights
                    if select_and_send_or_close_socket(obj, conn):
                        ack_time = time.time()
                        wait_ack = True
                    else:
                        self.__weights_lock.release()
                        print("DEBUG: select_and_send_or_close_socket failed in worker thread")
                        break
                    self.__weights = None
                else:
                    elapsed = time.time() - ack_time
                    print(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                    if elapsed >= ACK_TIMEOUT_REDIS_TO_WORKER:
                        print("INFO: ACK timed-out, breaking connection")
                        self.__weights_lock.release()
                        break
            self.__weights_lock.release()  # END WEIGHTS LOCK...........................................................
            # checks for samples
            success, obj = poll_and_recv_or_close_socket(conn)
            if not success:
                print("DEBUG: poll failed in rollout thread")
                break
            elif obj is not None and obj != 'ACK':
                print(f"DEBUG INFO: rollout worker thread received obj")
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                self.__buffer += obj  # concat worker batch to local batch
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
            elif obj == 'ACK':
                wait_ack = False
                print(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
            time.sleep(LOOP_SLEEP_TIME)  # TODO: adapt


# TRAINER: ==========================================

class TrainerInterface:
    """
    This is the trainer's network interface
    This connects to the redis server
    This receives samples batches and sends new weights
    """
    def __init__(self,
                 redis_ip=None,
                 model_path=MODEL_PATH_TRAINER):
        self.__buffer_lock = Lock()
        self.__weights_lock = Lock()
        self.__weights = None
        self.__buffer = Buffer()
        self.model_path = model_path
        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.redis_ip = redis_ip if redis_ip is not None else '127.0.0.1'

        print(f"local IP: {self.local_ip}")
        print(f"public IP: {self.public_ip}")
        print(f"redis IP: {self.redis_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Trainer interface thread
        """
        ack_time = time.time()
        wait_ack = False
        while True:  # main client loop
            s = get_connected_socket(SOCKET_TIMEOUT_CONNECT_TRAINER, self.redis_ip, PORT_TRAINER)
            if s is None:
                print("DEBUG: get_connected_socket failed in TrainerInterface thread")
                continue
            while True:
                # send weights
                self.__weights_lock.acquire()  # WEIGHTS LOCK...........................................................
                if self.__weights is not None:  # new weights
                    if not wait_ack:
                        obj = self.__weights
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__weights_lock.release()
                            print("DEBUG: select_and_send_or_close_socket failed in TrainerInterface")
                            break
                        self.__weights = None
                    else:
                        elapsed = time.time() - ack_time
                        print(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_TRAINER_TO_REDIS:
                            print("INFO: ACK timed-out, breaking connection")
                            self.__weights_lock.release()
                            break
                self.__weights_lock.release()  # END WEIGHTS LOCK.......................................................
                # checks for samples batch
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print("DEBUG: poll failed in TrainerInterface thread")
                    break
                elif obj is not None and obj != 'ACK':  # received buffer
                    print(f"DEBUG INFO: trainer interface received obj")
                    self.__buffer_lock.acquire()  # BUFFER LOCK.........................................................
                    self.__buffer += obj
                    self.__buffer_lock.release()  # END BUFFER LOCK.....................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                time.sleep(LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def broadcast_model(self, model: ActorModule):
        """
        model must be an ActorModule (sac_models.py)
        broadcasts the model's weights to all connected RolloutWorkers
        """
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        torch.save(model.state_dict(), self.model_path)
        with open(self.model_path, 'rb') as f:
            self.__weights = f.read()
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................

    def retrieve_buffer(self):
        """
        returns a copy of the TrainerInterface's local buffer, and clears it
        """
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        buffer_copy = deepcopy(self.__buffer)
        self.__buffer.clear()
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        return buffer_copy


# ROLLOUT WORKER: ===================================

class RolloutWorker:
    def __init__(self,
                 env_id,
                 actor_module_cls,
                 # obs_space,
                 # act_space,
                 device="cpu",
                 redis_ip=None,
                 samples_per_worker_batch=1000,
                 model_path=MODEL_PATH_WORKER,
                 obs_preprocessor: callable = None
                 ):
        self.obs_preprocessor = obs_preprocessor
        self.env = UntouchedGymEnv(id=env_id, gym_kwargs={"config": CONFIG_DICT})
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.model_path = model_path
        self.actor = actor_module_cls(obs_space, act_space)
        self.actor.load_state_dict(torch.load(self.model_path))
        self.device = device
        self.buffer = Buffer()
        self.__buffer = Buffer()  # deepcopy for sending
        self.__buffer_lock = Lock()
        self.__weights = None
        self.__weights_lock = Lock()
        self.samples_per_worker_batch = samples_per_worker_batch

        self.public_ip = get('http://api.ipify.org').text
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.redis_ip = redis_ip if redis_ip is not None else '127.0.0.1'

        print(f"local IP: {self.local_ip}")
        print(f"public IP: {self.public_ip}")
        print(f"redis IP: {self.redis_ip}")

        Thread(target=self.__run_thread, args=(), kwargs={}, daemon=True).start()

    def __run_thread(self):
        """
        Redis thread
        """
        ack_time = time.time()
        wait_ack = False
        while True:  # main client loop
            s = get_connected_socket(SOCKET_TIMEOUT_CONNECT_ROLLOUT, self.redis_ip, PORT_ROLLOUT)
            if s is None:
                print("DEBUG: get_connected_socket failed in worker")
                continue
            while True:
                # send buffer
                self.__buffer_lock.acquire()  # BUFFER LOCK.............................................................
                if len(self.__buffer) >= self.samples_per_worker_batch:  # a new batch is available
                    if not wait_ack:
                        obj = self.__buffer
                        if select_and_send_or_close_socket(obj, s):
                            ack_time = time.time()
                            wait_ack = True
                        else:
                            self.__buffer_lock.release()
                            print("DEBUG: select_and_send_or_close_socket failed in worker")
                            break
                        self.__buffer.clear()  # empty sent batch
                    else:
                        elapsed = time.time() - ack_time
                        print(f"WARNING: object ready but ACK from last transmission not received. Elapsed:{elapsed}s")
                        if elapsed >= ACK_TIMEOUT_WORKER_TO_REDIS:
                            print("INFO: ACK timed-out, breaking connection")
                            self.__buffer_lock.release()
                            break
                self.__buffer_lock.release()  # END BUFFER LOCK.........................................................
                # checks for new weights
                success, obj = poll_and_recv_or_close_socket(s)
                if not success:
                    print(f"INFO: rollout worker poll failed")
                    break
                elif obj is not None and obj != 'ACK':
                    print(f"DEBUG INFO: rollout worker received obj")
                    self.__weights_lock.acquire()  # WEIGHTS LOCK.......................................................
                    self.__weights = obj
                    self.__weights_lock.release()  # END WEIGHTS LOCK...................................................
                elif obj == 'ACK':
                    wait_ack = False
                    print(f"INFO: transfer acknowledgment received after {time.time() - ack_time}s")
                time.sleep(LOOP_SLEEP_TIME)  # TODO: adapt
            s.close()

    def act(self, obs, train=False):
        """
        converts inputs to torch tensors and converts outputs to numpy arrays
        """
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        obs = collate([obs], device=self.device)
        with torch.no_grad():
            action_distribution = self.actor(obs)
            action = action_distribution.sample() if train else action_distribution.sample_deterministic()
        action, = partition(action)
        return action

    def collect_n_steps(self, n, train=True):
        """
        collects n transitions (from reset)
        set train to False for test samples, True for train samples
        """
        # self.buffer.clear()
        act = self.env.default_action
        obs = self.env.reset()
        # print(f"DEBUG: rw init obs:{obs}, act:{act}")
        # print(f"DEBUG: init obs[0]:{obs[0]}")
        # print(f"DEBUG: init obs[1][-1].shape:{obs[1][-1].shape}")
        self.buffer.append_sample(get_buffer_sample(obs, act, 0.0, False, {}))
        for _ in range(n):
            act = self.act(obs, train)
            # print(f"DEBUG: rw act:{act}")
            obs, rew, done, info = self.env.step(act)
            # print(f"DEBUG: rw obs:{obs}, rew:{rew}, done:{done}")
            self.buffer.append_sample(get_buffer_sample(obs, act, rew, done, info))  # WARNING: in the buffer, act is for the PREVIOUS transition (act, obs(act))

    def run(self):
        while True:
            print("INFO: collecting samples")
            self.collect_n_steps(self.samples_per_worker_batch, train=True)
            print("INFO: copying buffer for sending")
            self.send_and_clear_buffer()
            print("INFO: checking for new weights")
            self.update_actor_weights()

    def send_and_clear_buffer(self):
        self.__buffer_lock.acquire()  # BUFFER LOCK.....................................................................
        self.__buffer = deepcopy(self.buffer)
        self.__buffer_lock.release()  # END BUFFER LOCK.................................................................
        self.buffer.clear()

    def update_actor_weights(self):
        """
        updates the model with new weights from the trainer when available
        """
        self.__weights_lock.acquire()  # WEIGHTS LOCK...................................................................
        if self.__weights is not None:  # new weights available
            with open(self.model_path, 'wb') as f:  # FIXME: check that this deletes the old file
                f.write(self.__weights)
            self.actor.load_state_dict(torch.load(self.model_path))
            print("INFO: model weights have been updated")
            self.__weights = None
        self.__weights_lock.release()  # END WEIGHTS LOCK...............................................................


# Environment-dependent interface ===================

def get_buffer_sample(obs, act, rew, done, info):
    """
    this creates the object that will actually be stored in the buffer
    # TODO
    """
    obs_mod = (obs[0], obs[1][-1])  # speed and most recent image only
    rew_mod = np.float32(rew)
    return obs_mod, act, rew_mod, done, info


# Main ==============================================

def main(args):
    worker = args.worker
    redis = args.redis
    trainer = args.trainer

    if redis:
        rs = RedisServer(samples_per_redis_batch=1000, localhost=LOCALHOST)
    elif worker:
        rw = RolloutWorker(env_id="gym_tmrl:gym-tmrl-v0",
                           actor_module_cls=partial(POLICY, act_in_obs=ACT_IN_OBS),
                           device='cuda' if PRAGMA_CUDA else 'cpu',
                           redis_ip=REDIS_IP,
                           samples_per_worker_batch=1000,
                           model_path=MODEL_PATH_WORKER,
                           obs_preprocessor=OBS_PREPROCESSOR)
        rw.run()
    else:
        main_train()
    while True:
        time.sleep(1.0)


def main_train():
    from agents import TrainingOffline, run_wandb_tm
    from agents.util import partial
    from agents.sac import Agent
    from agents.envs import UntouchedGymEnv

    Sac_tm = partial(
        TrainingOffline,
        Env=partial(UntouchedGymEnv,
                    id="gym_tmrl:gym-tmrl-v0",
                    gym_kwargs={"config": CONFIG_DICT}),
        epochs=50,
        rounds=10,
        steps=1000,
        update_model_interval=1000,
        update_buffer_interval=1000,
        max_training_steps_per_env_step=1.0,
        Agent=partial(Agent,
                      OutputNorm=partial(beta=0., zero_debias=False),
                      Memory=MEMORY,
                      device='cuda' if PRAGMA_CUDA else 'cpu',
                      Model=partial(TRAIN_MODEL,
                                    act_in_obs=ACT_IN_OBS),
                      memory_size=500000,
                      batchsize=64,
                      lr=0.0003,  # default 0.0003
                      discount=0.99,
                      target_update=0.005,
                      reward_scale=0.5,  # default: 5.0
                      entropy_scale=1.0),  # default: 1.0
    )

    print("--- NOW RUNNING: SAC trackmania ---")
    interface = TrainerInterface(redis_ip=REDIS_IP, model_path=MODEL_PATH_TRAINER)
    run_wandb_tm(None,
                 None,
                 None,
                 interface,
                 run_cls=Sac_tm,
                 checkpoint_path=CHECKPOINT_PATH)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--redis', dest='redis', action='store_true')
    parser.set_defaults(redis=False)
    parser.add_argument('--trainer', dest='trainer', action='store_true')
    parser.set_defaults(trainer=False)
    parser.add_argument('--worker', dest='worker', action='store_true')  # not used
    parser.set_defaults(worker=False)
    args = parser.parse_args()
    main(args)
