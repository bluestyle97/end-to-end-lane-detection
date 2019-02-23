import numpy
import warnings
from sklearn.cluster import MeanShift

def ms_cluster(prediction, bandwidth):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(prediction)

    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    num_clusters = cluster_centers.shape[0]
    
    return num_clusters, labels, cluster_centers

def get_lane_area(binary_seg_ret, instance_seg_ret):
    idx = np.where(binary_seg_ret==1)

    lane_embedding_feats = []
    lane_coordinate = []
    for i in range(len(idx[0])):
        lane_embedding_feats.append(instance_seg_ret[idx[0][i], idx[1][i]])
        lane_coordinate.append([idx[0][i], idx[1][i]])
    
    return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)

def threshold_coord(coord):
    pts_x = coord[:, 0]
    mean_x = np.mean(pts_x)

    idx = np.where(np.abs(pts_x - mean_x) < mean_x)

    return coord[idx[0]]

def lane_fit(lane_pts):
    if not isinstance(lane_pts, np.ndarray):
        lane_pts = np.array(lane_pts, np.float32)

    x = lane_pts[:, 0]
    y = lane_pts[:, 1]
    x_fit = []
    y_fit = []

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            f1 = np.polyfit(x, y, 3)
            p1 = np.poly1d(f1)
            x_min = int(np.min(x))
            x_max = int(np.max(x))
            x_fit = []
            for i in range(x_min, x_max + 1):
                x_fit.append(i)
            y_fit = p1(x_fit)
        except Warning as e:
            x_fit = x
            y_fit = y
        finally:
            return zip(x_fit, y_fit)
