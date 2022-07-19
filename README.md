Hello everyone:) 

The scripts in this repo were developed by me as part of my PhD, in which I investigated Kardar-Parisi-Zhang signatures in exciton polariton systems (the scripts here refer to the 2D case). 
Some words for the individual modules:
  -- utils.py : Various utility functions; isotropic average, ensemble average, unwrapping, etc. 
  -- model_script.py: Class for the Gross-Pitaevskii equation solver, using the split-step method in 2nd order (half step in real space -> full step in momentum space -> half step in real space). The class incorporates two functions which I used to extract different observables (correlations of the wavefunction, statistics of the phase), as well as a function which extracts a 3D dataset of the phase in space and time.
  -- three scripts for parallel calculations, each using one of the functions mentioned above.
 
I used some annotations to try to guide the reader to what I was trying to do, but of course they might not be enough! 
Feel free to write me at konstantinos.delis5@gmail.com to let me know if you used the scripts or some part of them, or if you have any kind of question!

Cheers,
Konstantinos
