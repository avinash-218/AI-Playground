# Object-Tracker-KCF

KCF tracker

This repository is to use KCF algorithm to track objects.
It rectangulates object of interest when found and displays error message at top 
right when the object is lost.

Compare the result of this and https://github.com/avinash-218/Object-Tracker-CSRT and you can see that if 
the part from Bolt's head till chest is the object of interest to track , this 
tracker is efficient in result as KCF can't track even for two seconds.

That's the limitation of KCF algorithm and the advantage of CSRT.But as you could be able to see that while tracking 
CSRT seems to be lagging but it is due to the complex mathematical computations of this algorithm.
KCF - Slow moving object tracking, less objects in surroundings; Faster in tracking
CSRT - Fast moving object tracking, can do good in surroundings with many objects; Slower tracking than KCF
