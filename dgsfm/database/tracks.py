import numpy as np
from collections import defaultdict
from tqdm import tqdm
from dgsfm.database.structures import ImageBatch, TrackBatch
from dgsfm.database.viewgraph import ViewGraph



class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}      # rank stores the approximate height of each tree

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Union by Rank: Attach the smaller tree to the larger one
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                # If ranks are equal, pick one and increment its rank
                self.parent[root_x] = root_y
                self.rank[root_y] += 1

    def clear(self):
        self.parent.clear()
        self.rank.clear()


class TrackManager:
    def __init__(self) -> None:
        pass

    def establish_full_tracks(self, view_graph: ViewGraph, images: ImageBatch) -> TrackBatch:
        registered = set(images.registered_indices.tolist())
        pairwise_tracks = []
        feature_tracks = defaultdict(list)
        invalid_matches = 0

        for pair in tqdm(view_graph.image_pairs.values(), desc="Building pairwise tracks", position=0):
            if not pair.is_valid:
                continue
            if pair.image_id1 not in registered or pair.image_id2 not in registered:
                continue

            for match_idx in pair.inliers:
                match_idx = int(match_idx)
                if match_idx < 0 or match_idx >= len(pair.matches):
                    invalid_matches += 1
                    continue

                point1, point2 = map(int, pair.matches[match_idx])
                if not (0 <= point1 < len(images[pair.image_id1].features)) or not (
                    0 <= point2 < len(images[pair.image_id2].features)
                ):
                    invalid_matches += 1
                    continue
                if not np.all(np.isfinite(images[pair.image_id1].features[point1])) or not np.all(
                    np.isfinite(images[pair.image_id2].features[point2])
                ):
                    invalid_matches += 1
                    continue

                track_id = len(pairwise_tracks)
                observations = {pair.image_id1: point1, pair.image_id2: point2}
                pairwise_tracks.append(observations)
                for observation in observations.items():
                    feature_tracks[observation].append(track_id)

        # Merge tracks at shared features unless they disagree on another image.
        uf = UnionFind()
        components = dict(enumerate(pairwise_tracks))
        for track_ids in tqdm(feature_tracks.values(), desc="Connecting multiview tracks", position=0):
            for i, track_id1 in enumerate(track_ids):
                for track_id2 in track_ids[i + 1:]:
                    root1, root2 = uf.find(track_id1), uf.find(track_id2)
                    if root1 == root2:
                        continue
                    observations1, observations2 = components[root1], components[root2]
                    if any(
                        image_id in observations1 and observations1[image_id] != feature_id
                        for image_id, feature_id in observations2.items()
                    ):
                        continue
                    merged = observations1 | observations2
                    uf.union(root1, root2)
                    components.pop(root1)
                    components.pop(root2)
                    components[uf.find(root1)] = merged

        # A feature belongs to only one final track; keep the longest candidate.
        candidates = sorted(
            components.values(),
            key=lambda track: (-len(track), tuple(sorted(track.items()))),
        )
        used_features = set()
        valid_tracks = []
        for candidate in tqdm(candidates, desc="Removing duplicate tracks", position=0):
            observations = set(candidate.items())
            if observations.isdisjoint(used_features):
                valid_tracks.append(np.asarray(sorted(observations), dtype=np.uint32))
                used_features.update(observations)

        tracks = TrackBatch(len(valid_tracks))
        tracks.observations = valid_tracks
        print(
            f"Initialized {len(tracks)} tracks from {len(pairwise_tracks)} valid matches "
            f"({len(candidates) - len(valid_tracks)} duplicate tracks and "
            f"{invalid_matches} invalid matches removed)."
        )
        return tracks
