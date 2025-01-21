import rospy
from rosbag import Bag
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image

def calculate_iou(box1, box2):
    """
    Berechnet die Intersection over Union (IoU) zwischen zwei Bounding Boxen.
    :param box1: Tuple (center_x, center_y, size_x, size_y) der ersten Box
    :param box2: Tuple (center_x, center_y, size_x, size_y) der zweiten Box
    :return: IoU-Wert
    """
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    # Berechne die Koordinaten der Schnittbox
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    # Fläche der Schnittbox
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_width * inter_height

    # Flächen der Boxen
    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Fläche der Vereinigung
    union_area = area_box1 + area_box2 - intersection_area

    # Vermeide Division durch Null
    if union_area == 0:
        return 0

    return intersection_area / union_area

def process_yolo_result(msg, result_data):
    """
    Verarbeitet die YoloResult-Nachricht und extrahiert Bounding Boxen.
    Speichert die Daten für jede Rosbag in einem Dictionary.
    """
    rospy.loginfo(f"Header: {msg.header}")
    detections = msg.detections

    rospy.loginfo(f"Erhalte {len(detections.detections)} Detektionen.")

    for i, detection in enumerate(detections.detections):
        bbox = detection.bbox
        center = bbox.center  # Pose2D: x, y und theta (Rotation)
        size_x = bbox.size_x  # Breite der Bounding Box
        size_y = bbox.size_y  # Höhe der Bounding Box

        # Speichern der Daten für jede Rosbag-Datei
        result_data['boxes'].append((center.x, center.y, size_x, size_y))

        rospy.loginfo(
            f"Detection {i}: Label={detection.results[0].id if detection.results else 'Unbekannt'}, "
            f"Score={detection.results[0].score if detection.results else 0.0}, "
            f"Center=({center.x}, {center.y}), "
            f"Size=({size_x}, {size_y}), "
            f"Rotation={center.theta}"
        )

def process_rosbag(bag_file, topic_name, result_data):
    """
    Öffnet eine Rosbag und verarbeitet das angegebene Topic.
    """
    rospy.loginfo(f"Öffne Rosbag {bag_file} und lese Topic {topic_name}")

    with Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            if topic == topic_name:
                process_yolo_result(msg, result_data)

    rospy.loginfo(f"Verarbeitung von {bag_file} abgeschlossen.")

def main():
    """
    Hauptfunktion: Verarbeitet mehrere Rosbag-Dateien und berechnet die IoU.
    """
    rospy.init_node('process_yolo_rosbag', anonymous=True)

    # Daten für jede Rosbag-Datei speichern
    result_data_bag1 = {'boxes': []}
    result_data_bag2 = {'boxes': []}

    # Rosbag-Dateien und Topic-Name definieren
    bag_file_1 = '/mnt/c/Users/felix/Studium/Projekt/YoloImageSimulation.bag'
    bag_file_2 = '/mnt/c/Users/felix/Studium/Projekt/YoloImageReal.bag'  # Beispiel für zweite Datei
    topic_name = '/yolo_result'

    # Verarbeite beide Rosbags
    process_rosbag(bag_file_1, topic_name, result_data_bag1)
    process_rosbag(bag_file_2, topic_name, result_data_bag2)

    # Berechne IoU für entsprechende Bounding Boxen
    min_boxes = min(len(result_data_bag1['boxes']), len(result_data_bag2['boxes']))
    for i in range(min_boxes):
        box1 = result_data_bag1['boxes'][i]
        box2 = result_data_bag2['boxes'][i]
        iou = calculate_iou(box1, box2)
        rospy.loginfo(f"IoU für Box {i}: {iou}")

    rospy.loginfo("Verarbeitung abgeschlossen.")

if __name__ == "__main__":
    main()
