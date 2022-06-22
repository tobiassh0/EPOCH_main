# EPOCH_main
Used for analysis of EPOCH 1d PIC simulations. Old one was deleted.

* batch_load        :: User interface with code. Locates simulation, reads and writes data to an class object called "Simulation". Calls heavily on list_new
* list_new          :: Main hub of functions used for interacting with code. Rewritten due to batch processing. Updated sparcely. 
* energy            :: Called from list_new, calls to calculate, reads and plots cahnge in energy densities through time
* power             :: Called from list_new, calls to calculate, reads and plots power spectra
* functions_basic   :: Copied from previous PhD student. Contains basic list of functions for analysing (_small_) EPOCH simulations
* charge_dens       :: Used for creating current density ($J_{x,y,z}$) and momentum ($p_{x,y,z}$) evolution plots
* dist_fn           :: Redundant, absorbed into list_new
* my_constants      :: Contains typical physical constants as well as common ion/electron mass ratios.

Should at least have batch_load, list_new, my_constants, power and energy in the same dir. Doesn't need to be in sim dir, can be anywhere so long as the sim dir is called correctly in batch_load. User should input values when prompted (adequate direction provided I think?). Should allow for most conditions and provide info on errors rather than just a flag.
