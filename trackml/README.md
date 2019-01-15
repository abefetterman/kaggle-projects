# TrackML Particle Tracking Challenge

[Link to competition](https://www.kaggle.com/c/trackml-particle-identification)

Cluster ~100k 3D points into ~10k particle tracks.

## Model and Training

Triplets (three points in the same direction of space) are enumerated and used
to form a graph of point pairs. DBSCAN is run on the graph. 

In another model, CMA-ES was used to optimize relative distance of point
parameters for direct use of DBSCAN, but this was less accurate than the graph
approach above.

 