import numpy as np





def wh_iou(box1, box2):
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    inter_area = min(w1, w2) * min(h1, h2)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / (union_area + 1e-6)


class TwoStageClusterer:
    def __init__(self, k_init=20, k_final=8):
        self.k_init = k_init
        self.k_final = k_final
        self.centroids = None

    def fit(self, boxes):
        self.centroids = self._kmeans(boxes, self.k_init)
        self.centroids = self._density_pruning(self.centroids, self.k_final)

        return self.centroids

    def _kmeans(self, boxes, k, max_iter=300):
        indices = np.random.choice(len(boxes), k, replace=False)
        centroids = boxes[indices]

        for _ in range(max_iter):
            assignments = []
            for box in boxes:
                dists = [1 - wh_iou(box, c) for c in centroids]
                assignments.append(np.argmin(dists))
            assignments = np.array(assignments)

            new_centroids = []
            max_shift = 0
            for i in range(k):
                cluster_boxes = boxes[assignments == i]
                if len(cluster_boxes) > 0:
                    new_c = cluster_boxes.mean(axis=0)
                else:
                    new_c = boxes[np.random.choice(len(boxes))]

                max_shift = max(max_shift, np.linalg.norm(new_c - centroids[i]))
                new_centroids.append(new_c)

            centroids = np.array(new_centroids)
            if max_shift < 1e-4:
                break
        return centroids

    def _density_pruning(self, centroids, target_k):
        areas = centroids[:, 0] * centroids[:, 1]
        sorted_indices = np.argsort(areas)
        sorted_centroids = centroids[sorted_indices].tolist()

        while len(sorted_centroids) > target_k:
            scores = []
            # S_i = d(c_i, c_{i-1}) + d(c_i, c_{i+1})
            for i in range(1, len(sorted_centroids) - 1):
                prev_c = sorted_centroids[i - 1]
                curr_c = sorted_centroids[i]
                next_c = sorted_centroids[i + 1]

                score = (1 - wh_iou(curr_c, prev_c)) + (1 - wh_iou(curr_c, next_c))
                scores.append(score)

            if not scores:
                break

            min_idx = np.argmin(scores) + 1
            removed = sorted_centroids.pop(min_idx)
            # print(f"  Pruning: Removed centroid {removed}, Score: {scores[min_idx-1]:.4f}")

        return np.array(sorted_centroids)

