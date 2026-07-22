import random
from simulation_pkg import basic

def main():
    # 수직주차: (1,3) 또는 (2,4) 세트로 무작위 배치 (basic 안에서 이미 선택됨)
    basic.load_model( "obstacle4", basic.model_types[3] , basic.parking_car1 )
    basic.load_model( "obstacle5", basic.model_types[4] , basic.parking_car2 )
    
    # For obstacle
    basic.load_model( "obstacle1", basic.model_types[2] , basic.obstacle_coordinates1 )
    basic.load_model( "obstacle2", basic.model_types[0] , basic.obstacle_coord(basic.obstacle_coordinates2))
    basic.load_model( "obstacle3", basic.model_types[1] , basic.obstacle_coord(basic.obstacle_coordinates3))

if __name__ == "__main__":
    main()