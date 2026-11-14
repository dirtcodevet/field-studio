"""fs-light (11-11): the shared light system's modules (rig, rig_controller, light_runtime, fs_lightcache, the probes).
A package so Field Studio can load them under one unique name (ui/lightpanel.py loads this folder as `fs_light`);
the scripts here still run directly (their imports fall back to the plain names)."""
