These are the codes that I developed for my PhD, where I studied Kardar-Parisi-Zhang signatures in exciton polariton condensates. They refer to the 2D system, although they can be formulated in 1D straightforwardly.
They include the following modules:
	utils.py: Various utility functions
	model_script: Class for split step numerical integration of the generalized Gross-Pitaevskii equation. Its separate functions for the time evolution serve different purposes, and I have three separate scripts depending on what I want to compute:
		g1_script: Calculates the spatiotemporal correlation function (see my papers for the definition)
 		theta_script: Gathers trajectories of the unwrapped phase of the condensate in space and time. "Unwrapped" means that θ it is not anymore bound in the interval [-π, π]. 
 		tests_script: Scans different parameter values as input for the Gross-Pitaevskii equation

 All the scripts have been annotated to some extent, but of course programming is very much dependent on personal taste; maybe what makes sense to me doesn't to you! Feel free to email me at konstantinos.delis5@gmail.com if you have any questions regarding my implementation :)

 Cheers, Konstantinos