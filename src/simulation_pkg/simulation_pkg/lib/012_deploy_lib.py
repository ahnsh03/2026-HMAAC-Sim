import os
import types
import marshal

import random
import numpy as np
from datetime import datetime
import time

# from simulation_pkg import get_pyc

random.seed(time.time())
"""
파일 수정 후 그냥 실행하면 pyc 파일 옮겨짐
"""


# huggingface.py의 sim.pt 파일 다운로드 코드
def check_and_download_model(file, destination_path):
    import shutil
    from huggingface_hub import hf_hub_download
    if not os.path.exists(destination_path):    
        model_path = hf_hub_download(repo_id='gogoring/simulation_ws', filename=file)
        shutil.copy(model_path, destination_path)
    else:
        pass


    
HOME = os.path.expanduser("~")

def _find_workspace_root():
    """Locate the colcon workspace that contains src/simulation_pkg/models."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(12):
        if os.path.isdir(os.path.join(d, "src", "simulation_pkg", "models")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    for candidate in (
        "/root/hmobility_ws",
        os.path.join(HOME, "hmobility_ws"),
        os.path.join(HOME, "H-Mobility-Autonomous-Advanced-Course-Simulation"),
        os.getcwd(),
    ):
        if os.path.isdir(os.path.join(candidate, "src", "simulation_pkg", "models")):
            return candidate
    raise FileNotFoundError(
        "Cannot locate workspace with src/simulation_pkg/models"
    )

def get_base_path(extra_dirs=None, repeat_last=False):
    ws = _find_workspace_root()
    BASE_PATH = os.path.join(ws, "src", "simulation_pkg")
    if repeat_last:
        BASE_PATH = os.path.join(BASE_PATH, "simulation_pkg")
    if extra_dirs:
        BASE_PATH = os.path.join(BASE_PATH, *extra_dirs)
    return BASE_PATH

def get_pkg(): # 패키지 경로
    return get_base_path()

def get_path(): # 노드 파일 경로
    return get_base_path(repeat_last=True)

def get_model(file_name=None): # model.sdf 파일 경로
    return get_base_path(["models", file_name, "model.sdf"])

def get_data(file_name=None): # data 폴더 안의 파일 경로
    return get_base_path(["data", file_name], repeat_last=True)

def get_lib(file_name=None): # lib/pyc 폴더 안의 파일 경로
    return get_base_path(["lib", "pyc", file_name], repeat_last=True)

def old_get_pyc(module_file):
    file_path = get_lib(module_file)
    print('\n파일명:', file_path, '\n\n')
    pyc = open(file_path, 'rb').read()
    code = marshal.loads(pyc[16:])
    module = types.ModuleType('module_name')
    exec(code, module.__dict__)
    return module

def get_time(is_img=True):
    now = datetime.now()
    now = now.strftime('%y%m%d_%H%M%S')
    if is_img:
        result = now + '.png' 
    return result    



# 시뮬레이션 세팅
def load_model(entity_name, model_name, random_coordinates):
    
    x, y, z, roll, pitch, yaw = random_coordinates
    
    # 사용한 좌표를 출력
    #print(f"Spawning model '{entity_name}' at coordinates: x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}")
    
    model_file = get_model(model_name)
    os.system(f"ros2 run gazebo_ros spawn_entity.py -file {model_file} -entity {entity_name} -x {x} -y {y} -z {z} -R {roll} -P {pitch} -Y {yaw}")

def driving_ego():  
    x_min = -5.092991 # 2
    x_max = -5.092991 # 2

    y_min = -22.694952 # 22.73
    y_max = -22.694952 # 22.73

    random_x = random.uniform(x_min, x_max)
    random_y = random.uniform(y_min, y_max)

    z = 0.011641

    r = -0.0
    p = 0.0
    y = 1.559726 # -1.563693

    return random_x, random_y, z, r, p, y

def old_obstacle_stand(): # 신호등
    x = 23.185000
    
    y = 4.388550
    
    y_min = 0.383322
    y_max = 5.920192    
    random_y = random.uniform(y_min, y_max)

    z = 0.0
    r = 0.0
    p = 0.0
    y = 0.0

    return x, y, z, r, p, y

def traffic_light_stand(): # 신호등   
    x_min = -5.913965
    x_max = -5.337101
    
    y_min = 17.857704
    y_max = 17.949493 

    random_x = random.uniform(x_min, x_max)
    random_y = random.uniform(y_min, y_max)
    
    z = 0.0

    r = 0.0
    p = 0.0
    y = 1.568773

    return random_x, random_y, z, r, p, y

def hmobility_traffic_light_stand(): # 신호등 (H-모빌리티 환경 - 고정 위치)
    x = 4.863314

    y_pos = -17.906038

    z = 0.0

    r = 0.0
    p = 0.0
    y = -1.565107

    return x, y_pos, z, r, p, y

def hmobility_obstacle_car(): # 장애물 차량 (H-모빌리티 환경 - 최종평가 유사, 고정 위치)
    x = 2.884013
    y_pos = 19.826846
    z = 0.011641

    r = 0.0
    p = 0.0
    y = -1.571629

    return x, y_pos, z, r, p, y

# 장애물 회피 차량 + 범위 지정 + 랜덤
model_types = ["prius_hybrid_ob1", "prius_hybrid_ob2", "prius_hybrid_ob3", "hatchback_green", "hatchback_yellow"] 

obstacle_coordinates1 = (12.718713, -15.719065, 0.00, 0.00, 0.00, 2.424641)
obstacle_coordinates_1= (-3.659642, 8.710748, 0.00, 0.00, 0.00, -0.013934)
obstacle_coordinates_2= (-3.659642, 2.037476, 0.00, 0.00, 0.00, -0.013934)

# 회피 장애물 후보 (x, y, yaw). 실차 배치를 그대로 측정한 값.
# obstacle2 (트랙 x≈12) / obstacle3 (트랙 x≈16), 각각 3후보 -> 조합 9가지
obstacle_coordinates2 = [(12.389278, 8.702062, -3.049152),
                         (12.236391, 10.351026, -3.049136),
                         (12.084256, 11.991503, -3.049116)]
obstacle_coordinates3 = [(16.984153, -3.086174, -3.041943),
                         (16.852462, -1.429280, -3.041441),
                         (16.682904, 0.257396, -3.041395)]

def obstacle_coord(coordinates):
    x_obstacle, y_obstacle, yaw = random.choice(coordinates)

    z = 0.0
    r = 0.0
    p = 0.0

    return x_obstacle, y_obstacle, z, r, p, yaw

parking_start = [(-1.672862, -16.311572, 0.011641, -0.000000, 0.00, -3.133789),
                 (-1.681217, -15.244810, 0.011641, -0.000000, 0.00, -3.133795),  
                 (-0.772971, -15.237810, 0.011641, -0.000000, 0.00, -3.133797),  
                 (-0.764668, -16.302988, 0.011641, -0.000000, 0.00, -3.133797),
                ]

parking_ego = random.choice(parking_start)


# 수직주차 4칸 (실차 배치를 그대로 측정한 값). 나란히 배열: x 거의 일정, y 만 다름.
# range 는 min=max 로 두어 정확히 그 자리에 놓이게 한다.
parking_zones = {
    1: {"x_range": (-4.619265, -4.619265), "y_range": (8.749825, 8.749825), "yaw_range": (-0.013934, -0.013934)},
    2: {"x_range": (-4.610268, -4.610268), "y_range": (5.387341, 5.387341), "yaw_range": (-0.013934, -0.013934)},
    3: {"x_range": (-4.601163, -4.601163), "y_range": (1.984500, 1.984500), "yaw_range": (-0.013934, -0.013934)},
    4: {"x_range": (-4.592220, -4.592220), "y_range": (-1.357908, -1.357908), "yaw_range": (-0.013934, -0.013934)}
}

def parking_coord(zone):  
    x_obstacle = random.uniform(zone["x_range"][0], zone["x_range"][1])
    y_obstacle = random.uniform(zone["y_range"][0], zone["y_range"][1])
    z = 0.0
    r = 0.0
    p = 0.0
    y = random.uniform(zone["yaw_range"][0], zone["yaw_range"][1])
    
    return x_obstacle, y_obstacle, z, r, p, y

# 주차 가능한 조합: 1번과 3번 / 2번과 4번에만 주차
valid_parking_pairs = [(1, 3), (2, 4)]
parking_pair = random.choice(valid_parking_pairs)
selected_zone1, selected_zone2 = parking_pair

# 주차할 칸을 무작위로 선택하고, 좌표를 생성
selected_zone_num = random.choice(parking_pair)
parking_car1 = parking_coord(parking_zones[selected_zone1])
parking_car2 = parking_coord(parking_zones[selected_zone2])
    


    
if __name__ == "__main__":
    import os
    import py_compile

    # 경로 설정
    username = os.getlogin()
    base_path = f"/home/{username}/H-Mobility-Autonomous-Advanced-Course-Simulation/src/simulation_pkg/simulation_pkg/lib"

    pyc_folder = os.path.join(base_path, "pyc")
    target_file = "012_deploy_lib.py"

    # 1. __pycache__ 폴더 삭제
    pycache_path = os.path.join(base_path, "__pycache__")
    if os.path.exists(pycache_path):
        import shutil
        shutil.rmtree(pycache_path)
        #print(f"Deleted: {pycache_path}")
        print("pyc 폴더 삭제")

    # 2. pyc 폴더 내 기존 pyc 파일 삭제
    pyc_file_to_delete = os.path.join(pyc_folder, "012_deploy_lib.cpython-310.pyc")
    if os.path.exists(pyc_file_to_delete):
        os.remove(pyc_file_to_delete)
        #print(f"Deleted: {pyc_file_to_delete}")
        print("기존 pyc 파일 삭제")

    # 3. 특정 파일 컴파일
    source_file_path = os.path.join(base_path, target_file)
    if os.path.exists(source_file_path):
        py_compile.compile(source_file_path, cfile=pyc_file_to_delete)
        #print(f"Compiled: {source_file_path} -> {pyc_file_to_delete}")
        print("끝")
    else:
        print(f"File not found: {source_file_path}")
