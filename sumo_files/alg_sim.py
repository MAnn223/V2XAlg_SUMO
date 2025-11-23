import traci
import math
import numpy as np

COMM_RANGE = 50.0
POS_NOISE = 2.0
SENSOR_RANGE = 60.0

np.random.seed(0)

def detect_objects(ego_id, ego_pos, veh_ids):
    detections = []
    for veh_id in veh_ids:
        if veh_id == ego_id:
            continue
        veh_pos = traci.vehicle.getPosition(veh_id)
        dist = math.dist(ego_pos, veh_pos)
        
        if dist > SENSOR_RANGE:
            continue
        
        noise = np.random.normal(0, POS_NOISE, 2)
        veh_pos_w_noise = (veh_pos[0] + noise[0], veh_pos[1] + noise[1])
        
        confidence = max(0.1, 1.0 - (dist/SENSOR_RANGE))
        
        detection = {
            'obj_id': veh_id,
            'pos': veh_pos_w_noise,
            'confidence': confidence,
            'detected_by': ego_id
        }
        
        detections.append(detection)
    return detections

def find_neighbors(veh_id, positions):
    cur_pos = positions[veh_id]
    neighbors = []
    
    for n_id, n_pos in positions.items():
        if n_id == veh_id:
            continue
        if math.dist(cur_pos, n_pos) <= COMM_RANGE:
            neighbors.append(n_id)
    return neighbors

def broadcast_detections(veh_detections, n_detections):
    all_detections = []
    all_detections.extend(veh_detections)
    for n in n_detections:
        all_detections.extend(n)
    return all_detections

def nms_fusion(detections, dist_threshold = 5.0):
    sorted_detections = sorted(detections, key = lambda d: d['confidence'], reverse = True)
    
    fused = []
    used = [False] * len(sorted_detections)
    
    for i in range (len(sorted_detections)):
        if used[i]:
            continue
        
        # keep detection with highest confidence
        fused.append(sorted_detections[i])
        used[i] = True
        
        # remove closeby detectios
        for j in range(i + 1, len(sorted_detections)):
            if used[j]:
                continue
            dist = math.dist(sorted_detections[i]['pos'], sorted_detections[j]['pos'])
            if dist < dist_threshold:
                used[j] = True
    
    return fused
        

def run_sim():
    traci.start(["sumo-gui", "-n", "net.net.xml", "-r", "routes.rou.xml"])

    # run sim as long as there are vehicles expected
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        sim_time = traci.simulation.getTime()
        
        # get all vehicles
        veh_ids = traci.vehicle.getIDList()
        
        # get all positions
        positions = {}
        for veh_id in veh_ids:
            positions[veh_id] = traci.vehicle.getPosition(veh_id)
        
        # each vehicle detects other vehicles
        all_detections = {}
        for veh_id in veh_ids:
            cur_pos = positions[veh_id]
            detections = detect_objects(veh_id, cur_pos, veh_ids)
            all_detections[veh_id] = detections
        
            if detections:
                    print(f"[t={sim_time:.1f}] {veh_id} detected {len(detections)} objects")
        
        #each vehicle shares with neighbors
        for veh_id in veh_ids:
            neighbors = find_neighbors(veh_id, positions)
            
            if neighbors:
                print(f" {veh_id} can communicate with: {neighbors}")
                n_detections = [all_detections[i] for i in neighbors]
                combined_detections = broadcast_detections(all_detections[veh_id], n_detections)
                combined_detections = [d for d in combined_detections if d['obj_id'] != veh_id]
                print(f"  {veh_id}: {len(combined_detections)} total detections")
                
                fused_detections = nms_fusion(combined_detections)
                print(f"  {veh_id}: {len(fused_detections)} fused detections")
                
#                for detection in fused_detections:
#                    print(f"    -> Object at ({detection['pos'][0]:.1f}, {detection['pos'][1]:.1f}), conf={detection['confidence']:.2f}")
        print()
    
    traci.close()

if __name__ == "__main__":
    run_sim()
