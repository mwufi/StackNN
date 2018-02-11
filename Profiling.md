# Making it faster?
So I got tired after 10 epochs or so.... is this really the fastest it can go? It's just copying some strings...

![reverse.py task](/assets/profile_1.png)
So it spends 50% of the time in backprop, 30% of the time in forward prop.

![the actual work: neural stack](/assets/profile_2.png)
It hits 6000 forward passes in 17 seconds.... spends a DAMN long time in the loop (and adding up `r`)