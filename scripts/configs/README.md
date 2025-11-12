# Configuration files for simulated training

- [training with dt=1e-3](./config_dte-3.yaml)
- [training with dt=1e-2](./config_dte-2.yaml)
- [training with dt=1e-1](./config_dte-1.yaml)
- [training with dt=1e0](./config_dte0.yaml)

> ⚠️ The base simulation used [here](./config_dte-3.yaml) write an output every 0.001 seconds, for 100 seconds of simulation => 1e5 field solutions, i.e 493GB ... So check first that you have this amount of storage on your system 😉