# TrackML Particle Tracking Challenge

[Link to competition](https://www.kaggle.com/c/trackml-particle-identification)

Cluster ~100k 3D points into ~10k particle tracks. The tracks are roughly circular 
in `x-y` and straight in `r-z` but difficult tracks contain soft collisions. 
The tracks roughly originate near center but about 10% originate outside the 
innermost detector. 10% of points are noise.

## Model and Training

Triplets (three points in the same direction of space) are enumerated and used
to form a graph of point pairs. DBSCAN is run on the graph. 

Here are the steps used:

* Triplet enumeration, O(n^2). For each point:
  * Select nearby points in `eta-phi` space (`eta` is the "pseudorapidity" `-ln(tan(theta/2))`, `phi` is azimuth `arctan(y/x)`) using a KD-tree
  * Divide into inner and outer groups based on distance to origin
  * Define inner pairs as pairs between reference point and inner point, outer pairs likewise
  * Calculate angle `theta` of inner and outer pair in `u-v` space, where `u=y/r^2` and `v=x/r^2`
  * Calculate angle `phi` of inner and outer pair in `r-z` space
  * Go through inner pairs and add to bins in `theta-phi` space
  * Go through outer pairs and find collisions where `theta-phi` matches
  * Score each pair match with D = (dtheta)^2 + (dphi)^2, with adjustment for distance from origin `rho`
  * Eliminate any triplet where neither pair belongs to another triplet
* Use mutual distance of pairs D as a distance for DBSCAN, O(n_pairs), with params `min_count` and `eps`:
  * Choose starting pair p
  * Find neighbors with D < `eps`
  * If count(neighbors) < `min_count` label the point as noise and start over
  * Else label the point as a new cluster, and for each neighbor:
    * If the point is unlabeled or labeled as noise, label it with the cluster
    * Count neighbors of this neighbor with D < `eps`
    * If count(neighbors) of neighbor >= `min_count`, add neighbors to queue for labeling
* Label points based on pairs with minimum group size `group_size`, O(n_pairs):
  * Label each point by the most common non-noise label of its pairs
  * Eliminate all labels with less than `group_size` points
  * Repeat labeling each point by the most common non-noise label until stable

