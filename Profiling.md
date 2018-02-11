# Making it faster?
So I got tired after 10 epochs or so.... is this really the fastest it can go? It's just copying some strings...

# Before
-  6000 forward passes in 17 seconds.... 
-  50% of the time in backprop, 30% of the time in forward prop.
![a](/assets/profile_1.png)
![b](/assets/profile_2.png)

# After
-2000 forward passes per second
- same allocation for backprop, forward prop
![a](/assets/profile_11.png)
![b](/assets/profile_21.png)


