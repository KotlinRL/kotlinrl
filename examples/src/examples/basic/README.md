### Running these examples requires open-rl-gymnasium-grpc-server to be running

```
docker pull kotlinrl/open-rl-gymnasium-grpc-server:latest
docker run --rm -p 50051:50051 kotlinrl/open-rl-gymnasium-grpc-server:latest
```
The `Basic` environment notebooks use action space sampling to interact with the environment and render the results
in a display frame.  The examples are not showing agent training, they simple prove the gymnasium environment is
working as expected.  Extended example notebooks will show how to use KotlinRL to train an agent to solve the specific
environment.