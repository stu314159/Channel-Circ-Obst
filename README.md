LBM simulation of 3D channel with a circular obstacle.  This code decomposes the domain for MPI with GPU acceleration; one GPU per MPI process.  

To be honest, I don't remember for sure that this works.  I think it does, and that it computes correct results but scaling is bad because I gave up before providing good computation/communication interleaving.