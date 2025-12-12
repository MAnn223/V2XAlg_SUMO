import traci
import math
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.affinity import rotate

COMM_RANGE = 50.0
POS_NOISE = 2.0
SENSOR_RANGE = 60.0
LENGTH_NOISE = 0.2
WIDTH_NOISE = 0.2
ANGLE_NOISE = 5.0
REAL_LENGTH = 4.5
REAL_WIDTH = 1.8

np.random.seed(0)

# simulates detection of surrounding vehicles with noise, returns list of detection dicts
def detect_objects(ego_id, ego_pos, veh_ids):

    detections = []

    for veh_id in veh_ids:
        if veh_id == ego_id:
            continue # skip itself

        veh_pos = traci.vehicle.getPosition(veh_id)
        veh_ang = traci.vehicle.getAngle(veh_id)

        # ignore if outside sensor detection range
        dist = math.dist(ego_pos, veh_pos)
        if dist > SENSOR_RANGE:
            continue
        
        # add noise to measured position
        pos_noise = np.random.normal(0, POS_NOISE, 2)
        veh_pos_w_noise = (veh_pos[0] + pos_noise[0], veh_pos[1] + pos_noise[1])
        
        # add noise to box
        noisy_length = np.random.normal(REAL_LENGTH, LENGTH_NOISE)
        noisy_width = np.random.normal(REAL_WIDTH, WIDTH_NOISE)
        noisy_angle = np.random.normal(veh_ang, ANGLE_NOISE)

        # decrease condfidence as distance increases
        confidence = max(0.1, 1.0 - (dist/SENSOR_RANGE))

        # detection dictionary 
        detection = {
            'obj_id': veh_id,
            'pos': veh_pos_w_noise,
            'length': noisy_length,
            'width': noisy_width,
            'angle': noisy_angle,
            'confidence': confidence,
            'covariance': {
                'x': POS_NOISE**2,
                'y': POS_NOISE**2,
                'length': LENGTH_NOISE**2,
                'width': WIDTH_NOISE**2,
                'angle': ANGLE_NOISE**2
            },
            'detected_by': ego_id
        }
        
        detections.append(detection)
    return detections

# find all vehicles within communication range 
def find_neighbors(veh_id, positions):
    cur_pos = positions[veh_id]
    neighbors = []
    
    for n_id, n_pos in positions.items():
        if n_id == veh_id:
            continue
        if math.dist(cur_pos, n_pos) <= COMM_RANGE:
            neighbors.append(n_id)
    return neighbors

# combine vehicle detections with neighbors
def broadcast_detections(veh_detections, n_detections):
    all_detections = []
    all_detections.extend(veh_detections)
    for n in n_detections:
        all_detections.extend(n)
    return all_detections

# approx IOU using center distance and max diagonal
# TODO: update to more accurate IOU similarity calc
def calculate_iou(box1, box2):
    x1, y1, l1, w1, a1 = box1['pos'][0], box1['pos'][1], box1['length'], box1['width'], box1['angle']
    x2, y2, l2, w2, a2 = box2['pos'][0], box2['pos'][1], box2['length'], box2['width'], box2['angle']

    center_dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    max_diagonal = max(math.sqrt(l1**2 + w1**2), math.sqrt(l2**2 + w2**2))
    if center_dist > max_diagonal:
        return 0.0
    
    iou = max(0, 1.0 - (center_dist / max_diagonal))
    return iou

# for each fused detection, get iou compared to gnd truth
def get_iou_accuracy(fused_det, ego_id, positions):
    iou_scores = {}
    
    for det in fused_det:
        if isinstance(det['obj_id'], list):
            obj_ids = det['obj_id']
        else:
            obj_ids = [det['obj_id']]
        for obj_id in obj_ids:
            if obj_id != ego_id:
                if math.dist(positions[ego_id], positions[obj_id]) <= COMM_RANGE:
                    true_pos = positions[obj_id]
                    true_angle = traci.vehicle.getAngle(obj_id)
                    ground_truth_box = {
                        'pos': (true_pos[0], true_pos[1]),
                        'length': REAL_LENGTH,
                        'width': REAL_WIDTH,
                        'angle': true_angle
                    }
                    iou = calculate_iou(det, ground_truth_box)
                    iou_scores[obj_id] = iou

    return iou_scores

def compute_average_precision(iou_scores, threshold=0.5):
    sorted_scores = sorted(iou_scores, reverse = True) 

    precisions = []
    recalls = []

    tp = 0
    fp = 0
    total_gt = len(iou_scores)

    for score in sorted_scores:
        if score >= threshold:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / total_gt 

        precisions.append(precision)
        recalls.append(recall)
    
    recall_precision = sorted(zip(recalls, precisions))
    recalls_sorted, precisions_sorted = zip(*recall_precision)

    # get area under precision-recall curve
    ap = np.trapz(precisions_sorted, recalls_sorted)

    return ap

def no_fusion(detections, ego_id):
    ego_detections = [d for d in detections if d['detected_by'] == ego_id]
    return ego_detections

# sort by confidence and keep most confident detection, remove all detections with IOU > threshold
def nms_fusion(detections, iouu_threshold = 0.5):
    sorted_detections = sorted(detections, key = lambda d: d['confidence'], reverse = True)
    
    fused = []
    used = [False] * len(sorted_detections)
    
    for i in range (len(sorted_detections)):
        if used[i]:
            continue
        
        # keep detection with highest confidence
        fused.append(sorted_detections[i])
        used[i] = True
        
        # remove closeby detections
        for j in range(i + 1, len(sorted_detections)):
            if used[j]:
                continue
            iou = calculate_iou(sorted_detections[i], sorted_detections[j])
            if iou > iouu_threshold:
                used[j] = True
    
    return fused

# group detections with IOU > threshold, find weighted average
def weighted_nms_fusion(detections, iou_threshold = 0.5):
    sorted_detections = sorted(detections, key = lambda d: d['confidence'], reverse = True)
    
    fused = []
    used = [False] * len(sorted_detections)
    
    for i in range (len(sorted_detections)):
        if used[i]:
            continue
        
        group = [sorted_detections[i]]
        used[i] = True
        
        
        # create group of detections close to primary detection being looked at
        for j in range(i+1, len(sorted_detections)):
            if used[j]:
                continue
            iou = calculate_iou(sorted_detections[i], sorted_detections[j])
            if iou > iou_threshold:
                group.append(sorted_detections[j])
                used[j] = True
        
        # find weights
        weights = []
        for det in group:
            w = det['confidence'] / det['covariance']['x']
            weights.append(w)
        
        total_weight = sum(weights)
        # find fused postiions
        x_fused = sum(det['pos'][0] * w for det, w in zip(group, weights)) / sum(weights)
        y_fused = sum(det['pos'][1] * w for det, w in zip(group, weights)) / sum(weights)
        length_fused = sum(det['length'] * w for det, w in zip(group, weights)) / total_weight
        width_fused = sum(det['width'] * w for det, w in zip(group, weights)) / total_weight
        angle_fused = sum(det['angle'] * w for det, w in zip(group, weights)) / total_weight
        
        fused.append({
            'pos': (x_fused, y_fused),
            'length': length_fused,
            'width': width_fused,
            'angle': angle_fused,
            'confidence': max(det['confidence'] for det in group),
            'obj_id': [det['obj_id'] for det in group] 
            })
        
    return fused

def plot_results(no_fusion_per_vehicle, nms_per_vehicle, weighted_per_vehicle):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1.5])
    
    ax_box = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])  
    ax_summary = fig.add_subplot(gs[1, :])

    # Extract all IOUs for box plot and statistics
    no_fusion_all_ious = []
    nms_all_ious = []
    weighted_all_ious = []
    
    for veh_id in nms_per_vehicle:
        no_fusion_all_ious.extend(no_fusion_per_vehicle[veh_id].values())
        nms_all_ious.extend(nms_per_vehicle[veh_id].values())
        weighted_all_ious.extend(weighted_per_vehicle[veh_id].values())

    # Box plot
    box_data = [no_fusion_all_ious, nms_all_ious, weighted_all_ious]
    bp = ax_box.boxplot(box_data, labels=['No Fusion', 'NMS', 'Weighted NMS'], patch_artist=True)
    
    # Color the boxes
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax_box.set_ylabel('IOU Score', fontsize=12)
    ax_box.set_title('IOU Distribution Comparison', fontsize=14, fontweight='bold')
    ax_box.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax_box.grid(axis='y', alpha=0.3)
    ax_box.legend()

    # IOU over time - per vehicle
    colors_per_veh = ['#1f77b4', '#ff7f0e', '#2ca02c']  
    markers = ['o', 's', '^']
    
    for idx, veh_id in enumerate(sorted(nms_per_vehicle.keys())):
        # No fusion data for this vehicle
        no_fusion_times = sorted(no_fusion_per_vehicle[veh_id].keys())
        no_fusion_ious = [no_fusion_per_vehicle[veh_id][t] for t in no_fusion_times]
        
        # NMS data for this vehicle
        nms_times = sorted(nms_per_vehicle[veh_id].keys())
        nms_ious = [nms_per_vehicle[veh_id][t] for t in nms_times]
        
        # Weighted NMS data for this vehicle
        weighted_times = sorted(weighted_per_vehicle[veh_id].keys())
        weighted_ious = [weighted_per_vehicle[veh_id][t] for t in weighted_times]
        
        # Plot with dotted line for no fusion, solid for NMS, dashed for weighted
        ax_time.plot(no_fusion_times, no_fusion_ious, 
                    color=colors_per_veh[idx], 
                    linestyle=':', 
                    marker=markers[idx],
                    markersize=4,
                    alpha=0.6,
                    linewidth=2,
                    label=f'{veh_id} (No Fusion)')
        
        ax_time.plot(nms_times, nms_ious, 
                    color=colors_per_veh[idx], 
                    linestyle='-', 
                    marker=markers[idx],
                    markersize=4,
                    alpha=0.7,
                    linewidth=2,
                    label=f'{veh_id} (NMS)')
        
        ax_time.plot(weighted_times, weighted_ious, 
                    color=colors_per_veh[idx], 
                    linestyle='--', 
                    marker=markers[idx],
                    markersize=4,
                    alpha=0.7,
                    linewidth=2,
                    label=f'{veh_id} (W-NMS)')
    
    ax_time.set_xlabel('Simulation Time (s)', fontsize=12)
    ax_time.set_ylabel('Average IOU Score', fontsize=12)
    ax_time.set_title('Per-Vehicle IOU Over Time', fontsize=14, fontweight='bold')
    ax_time.set_ylim([0, 1.05])
    ax_time.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax_time.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax_time.grid(alpha=0.3)
    
    # Statistical summary
    if len(nms_all_ious) > 0 and len(weighted_all_ious) > 0:
        avg_no_fusion = np.mean(no_fusion_all_ious)
        avg_nms = np.mean(nms_all_ious)
        avg_weighted = np.mean(weighted_all_ious)
        ap_no_fusion = compute_average_precision(no_fusion_all_ious)
        ap_nms = compute_average_precision(nms_all_ious)
        ap_weighted = compute_average_precision(weighted_all_ious)

        no_fusion_good = sum(1 for iou in no_fusion_all_ious if iou > 0.5) / len(no_fusion_all_ious) * 100
        nms_good = sum(1 for iou in nms_all_ious if iou > 0.5) / len(nms_all_ious) * 100
        weighted_good = sum(1 for iou in weighted_all_ious if iou > 0.5) / len(weighted_all_ious) * 100
        
        summary_text = f"""
    Statistical Summary (IOU Metric):
    
    No Fusion (Ego Only):
    - Mean IOU: {np.mean(no_fusion_all_ious):.3f}
    - Median IOU: {np.median(no_fusion_all_ious):.3f}
    - Std Dev: {np.std(no_fusion_all_ious):.3f}
    - Average Precision (AP): {ap_no_fusion:.3f}
    - Good detections (>0.5): {no_fusion_good:.1f}%
    
    NMS:
    - Mean IOU: {np.mean(nms_all_ious):.3f}
    - Median IOU: {np.median(nms_all_ious):.3f}
    - Std Dev: {np.std(nms_all_ious):.3f}
    - Average Precision (AP): {ap_nms:.3f}
    - Good detections (>0.5): {nms_good:.1f}%
    - Improvement over No Fusion: {((avg_nms - avg_no_fusion) / avg_no_fusion * 100):.1f}%
    
    Weighted NMS:
    - Mean IOU: {np.mean(weighted_all_ious):.3f}
    - Median IOU: {np.median(weighted_all_ious):.3f}
    - Std Dev: {np.std(weighted_all_ious):.3f}
    - Average Precision (AP): {ap_weighted:.3f}
    - Good detections (>0.5): {weighted_good:.1f}%
    - Improvement over No Fusion: {((avg_weighted - avg_no_fusion) / avg_no_fusion * 100):.1f}%
    - Improvement over NMS: {((avg_weighted - avg_nms) / avg_nms * 100):.1f}%
    
    Total Detections: {len(nms_all_ious)}
        """
    else:
        summary_text = "\n    No data collected during simulation."
    
    ax_summary.axis('off')
    ax_summary.text(0.02, 0.5, summary_text, fontsize=11, family='monospace', va='center')
    
    plt.tight_layout()
    plt.savefig('fusion_comparison_iou.png', dpi=300, bbox_inches='tight')
    print("\nGraph saved as 'fusion_comparison_iou.png'")
    plt.show()

def run_sim():
    traci.start(["sumo-gui", "-n", "net.net.xml", "-r", "routes.rou.xml"])
    
    # Store IOUs per vehicle per timestep
    # Structure: {veh_id: {timestep: avg_iou}}
    nms_per_vehicle = {}
    weighted_per_vehicle = {}
    no_fusion_per_vehicle = {}
    
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

        # Each vehicle shares with neighbors
        for veh_id in veh_ids:
            neighbors = find_neighbors(veh_id, positions)
            
            if neighbors:
                print(f" {veh_id} can communicate with: {neighbors}")
                n_detections = [all_detections[i] for i in neighbors]
                combined_detections = broadcast_detections(all_detections[veh_id], n_detections)
                combined_detections = [d for d in combined_detections if d['obj_id'] != veh_id]
                print(f"  {veh_id}: {len(combined_detections)} total detections")
                
                no_fusion_detections = no_fusion(combined_detections, veh_id)

                nms_fused_detections = nms_fusion(combined_detections)
                print(f"  {veh_id}: {len(nms_fused_detections)} nms fused detections")
                weighted_nms_fused_detections = weighted_nms_fusion(combined_detections)
                print(f"  {veh_id}: {len(weighted_nms_fused_detections)} w-nms fused detections")
                
                no_fusion_ious = get_iou_accuracy(no_fusion_detections, veh_id, positions)
                nms_ious = get_iou_accuracy(nms_fused_detections, veh_id, positions)
                weighted_ious = get_iou_accuracy(weighted_nms_fused_detections, veh_id, positions)
                
                # Store average IOU for this vehicle at this timestep
                if nms_ious:
                    avg_no_fusion_iou = np.mean(list(no_fusion_ious.values()))
                    avg_nms_iou = np.mean(list(nms_ious.values()))
                    avg_weighted_iou = np.mean(list(weighted_ious.values()))
                    
                    # Initialize vehicle dict if needed
                    if veh_id not in nms_per_vehicle:
                        no_fusion_per_vehicle[veh_id] = {}
                        nms_per_vehicle[veh_id] = {}
                        weighted_per_vehicle[veh_id] = {}
                    
                    no_fusion_per_vehicle[veh_id][sim_time] = avg_no_fusion_iou
                    nms_per_vehicle[veh_id][sim_time] = avg_nms_iou
                    weighted_per_vehicle[veh_id][sim_time] = avg_weighted_iou
                    
                    for car, iou in no_fusion_ious.items():
                        print(f"    No Fusion IOU for {car}: {iou:.3f}")
                    for car, iou in nms_ious.items():
                        print(f"    NMS IOU for {car}: {iou:.3f}")
                    for car, iou in weighted_ious.items():
                        print(f"    W-NMS IOU for {car}: {iou:.3f}")
        
        print()
    
    plot_results(no_fusion_per_vehicle, nms_per_vehicle, weighted_per_vehicle)
    traci.close()


if __name__ == "__main__":
    run_sim()